import io
import pandas as pd
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import Dataset, DataLoader
import torch.hub
import torch
import torchvision
import torch.nn as nn
from torchvision import models
import torch.optim as optim
import torchvision
import torch
import pandas as pd
import numpy as np
import numbers
from skimage import io
from skimage.color import gray2rgb
import lightning as L
import torchvision.transforms.functional as F_t  
import torch.nn.functional as F  
import torchvision.transforms.v2 as T
from lightning.pytorch.strategies import DDPStrategy


class BinaryClassificationDataset(Dataset):
    def __init__(self, csv_file, augmentation=True, apply_padding=True):
        self.df = pd.read_csv(csv_file)
        self.imgs = self.df['image_path'].values
        self.do_augment = augmentation
        self.apply_padding = apply_padding

        # geometric data augmentation
        self.apply_augmentation = T.Compose([
                GammaCorrectionTransform(gamma=0.2),
                T.RandomApply(transforms=[T.ColorJitter(brightness=0.2, contrast=0.2)], p=0.6),
                T.RandomApply(transforms=[T.RandomAffine(degrees=30, scale=(0.8, 1.15))], p=0.5),
                T.RandomVerticalFlip(p=0.5),
                T.RandomHorizontalFlip(p=0.5),
            ])

    def __len__(self):
        return len(self.imgs)
    
    def pad_to_multiple(self, image, multiple):
        h, w = image.shape[-2:]
        pad_h = (multiple - h % multiple) % multiple
        pad_w = (multiple - w % multiple) % multiple
        
        if pad_h > 0 or pad_w > 0:
            image = F_t.pad(image, (0, 0, pad_w, pad_h), padding_mode='constant') 
        
        return image

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        row_df = self.df.loc[self.df['image_path'] == img_path]
        numfind = row_df['numfind'].values[0]
        if numfind > 0:
            label = 1
        else:
            label = 0
    
        to_tensor = torchvision.transforms.ToTensor()
        img = io.imread(img_path) # 16 bit range
        img = (img / img.max()) # [0, 1] --> gray
        img = gray2rgb(img) # --> to rgb, in [0, 1] in H,W,3
        image = to_tensor(img).float() # --> tensor in [0, 1] in 3,H,W

        # Resize image to 1024x768
        if image.shape[-2:] != (1024, 768):
            resize_transform = T.Resize((1024, 768))
            image = resize_transform(image)

        if self.apply_padding:
            image = self.pad_to_multiple(image, 14)

        if self.do_augment:
            image = self.apply_augmentation(image)

        return image, label
        

class GammaCorrectionTransform:
    """Apply Gamma Correction to the image"""
    def __init__(self, gamma=0.5):
        self.gamma = self._check_input(gamma, 'gammacorrection')   
        
    def _check_input(self, value, name, center=1, bound=(0, float('inf')), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError("If {} is a single number, it must be non negative.".format(name))
            value = [center - float(value), center + float(value)]
            if clip_first_on_zero:
                value[0] = max(value[0], 0.0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError("{} should be a single number or a list/tuple with length 2.".format(name))

        # if value is 0 or (1., 1.) for gamma correction do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    def __call__(self, img):
        """
        Args:
            img (PIL Image or Tensor): Input image.

        Returns:
            PIL Image or Tensor: gamma corrected image.
        """
        gamma_factor = None if self.gamma is None else float(torch.empty(1).uniform_(self.gamma[0], self.gamma[1]))
        if gamma_factor is not None:
            img = F_t.adjust_gamma(img, gamma_factor, gain=1) 
        return img


class ResNetBinaryClassifier(L.LightningModule):
    def __init__(self, checkpoint_path=None, learning_rate=7e-5, weight_decay=1e-5, freeze_layers=False):
        super(ResNetBinaryClassifier, self).__init__()
        self.model = models.resnet50(pretrained=False)
        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu')) 
            state_dict = checkpoint['state_dict']  
            self.model.load_state_dict(state_dict, strict=False)

        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 1)
        self.criterion = nn.BCELoss()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        if freeze_layers:
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.model.layer4.parameters():
                param.requires_grad = True
            for param in self.model.fc.parameters():
                param.requires_grad = True

    def forward(self, x):
        x = self.model(x)
        return torch.sigmoid(x)
    
    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels.float().view(-1, 1))
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels.float().view(-1, 1))
        self.log('val_loss', loss, prog_bar=True)
        
        # Calculate accuracy
        predicted = (outputs > 0.5).float()
        accuracy = (predicted == labels.view(-1, 1)).float().mean()
        self.log('val_accuracy', accuracy, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        return optimizer


def train():

    train_csv = '/vol/biomedic3/bglocker/mscproj24/mrm123/retinanet/csv_files/massfil_vindr_1:1/train.csv'
    val_csv = '/vol/biomedic3/bglocker/mscproj24/mrm123/retinanet/csv_files/massfil_vindr_1:1/val.csv'

    batch_size = 16
    train_dataset = BinaryClassificationDataset(csv_file=train_csv, augmentation=True)
    val_dataset = BinaryClassificationDataset(csv_file=val_csv, augmentation=False)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = ResNetBinaryClassifier()

    wandb_logger = WandbLogger(project='ResNet50', name='massfil_vindr_0ns')
    checkpoint_callback = ModelCheckpoint(filename="{epoch}")
    checkpoint_callback_best = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        filename="best_massfil_vindr_0ns",
    )

    print("Initializing trainer for initial training phase")
    trainer = L.Trainer(
        max_epochs=75,  
        logger=wandb_logger,
        devices=2,
        callbacks=[checkpoint_callback, checkpoint_callback_best]
    )

    # Train the model (Initial phase)
    print("Starting initial training phase")
    trainer.fit(model, train_dataloader, val_dataloader)
    trainer.validate(model=model, dataloaders=val_dataloader)


if __name__=="__main__":
    train()
