import torchvision
import torch
import pandas as pd
import numpy as np
import ast
import numbers

from skimage import io
from skimage.color import gray2rgb
from torch import optim, utils
from torch.utils.data import Dataset

from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models import ResNet50_Weights
from torchvision import tv_tensors
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead
from lightning.pytorch.loggers import WandbLogger
import torchvision.transforms.functional as F
import torchvision.transforms.v2 as T


# Define the dataset class
class RCNNDataset(Dataset):
    def __init__(self, csv_file, augmentation=True):
        self.df = pd.read_csv(csv_file)
        self.imgs = self.df['image_path'].values
        self.do_augment = augmentation

        self.photometric_augment = T.Compose([
            GammaCorrectionTransform(gamma=0.2),
            T.RandomApply(transforms=[T.ColorJitter(brightness=0.2, contrast=0.2)], p=0.6)
        ])

        # geometric data augmentation
        self.geometric_augment = T.Compose([
                T.RandomApply(transforms=[T.RandomAffine(degrees=20, scale=(0.9, 1.1))], p=0.5),
                T.RandomVerticalFlip(p=0.5),
                T.RandomHorizontalFlip(p=0.5),
            ])

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        # Extract Image and Bounding Box info
        img_path = self.imgs[idx]
        row_df = self.df.loc[self.df['image_path'] == img_path]
        numfind = row_df['numfind'].values[0]
        if numfind > 0:
            boxes = torch.tensor(ast.literal_eval(row_df['bbox'].values[0]))
            labels = torch.ones(numfind, dtype=torch.int64)
        else:
            boxes = torch.empty((0, 4))
            labels = torch.tensor([0], dtype=torch.int64)

        to_tensor = torchvision.transforms.ToTensor()
        img = io.imread(img_path) # 16 bit range
        img = (img / img.max()) # [0, 1] --> gray
        img = gray2rgb(img) # --> to rgb, in [0, 1] in H,W,3
        image = to_tensor(img).float() # --> tensor in [0, 1] in 3,H,W

        # Resize image to 1024x768
        if image.shape[-2:] != (1024, 768):
            resize_transform = T.Resize((1024, 768))
            image = resize_transform(image)

        boxes = tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=(1024, 768))

        if self.do_augment:
            image = self.photometric_augment(image)
            if numfind > 0:
                image, boxes = self.geometric_augment(image, boxes)
            else:
                image = self.geometric_augment(image)

        # Filter out invalid bounding boxes produced during augmentation (width or height <= 0)
        valid_boxes = []
        valid_labels = []
        for box, label in zip(boxes, labels):
            if (box[2] > box[0]) and (box[3] > box[1]):
                valid_boxes.append(box)
                valid_labels.append(label)
    
        if valid_boxes:
            boxes = torch.stack(valid_boxes)
            labels = torch.tensor(valid_labels, dtype=torch.int64)
        else:
            boxes = torch.empty((0, 4), dtype=torch.float32)
            labels = torch.tensor([0], dtype=torch.int64)

        target = {'boxes': boxes, 'labels': labels}

        return image, target
    

class FasterRCNNModel(L.LightningModule):
    def __init__(self, learning_rate=1e-5, weight_decay=1e-5, momentum=0.9):
        super().__init__()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.momentum = momentum

        optimized_ratios = [1.0, 1.1912243160876392, 0.83947245409187]
        optimized_scales = [0.6701667817494404, 0.43826872391648763, 1.0929571608034148]
        anchor_sizes = [32, 64, 128, 256, 512]

        self.anchor_generator = AnchorGenerator(
            sizes=tuple((size * np.array(optimized_scales)).tolist() for size in anchor_sizes),
            aspect_ratios=tuple([optimized_ratios] * len(anchor_sizes))
        )

        self.model = fasterrcnn_resnet50_fpn_v2(
            num_classes=2,
            weights_backbone=ResNet50_Weights.IMAGENET1K_V1,
        )

        # Replace the default anchor generator in the model
        self.model.rpn.anchor_generator = self.anchor_generator

        self.model.rpn.head = RPNHead(256, self.anchor_generator.num_anchors_per_location()[0])

    def forward(self, images, targets=None):
        return self.model(images, targets)

    def restructure_targets(self, targets):
    # Convert the existing target structure into a list of dictionaries
        batch_size = len(targets['boxes'])
        new_targets = []
        for i in range(batch_size):
            target = {
                'boxes': targets['boxes'][i],
                'labels': targets['labels'][i]
            }
            new_targets.append(target)
        
        return new_targets

    def training_step(self, batch, batch_idx):
        x, targets = batch
        targets = self.restructure_targets(targets)
        loss_dict = self.model(x, targets)
        loss = sum(loss for loss in loss_dict.values())
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, targets = batch
        targets = self.restructure_targets(targets)
        self.model.train()
        with torch.no_grad():
            loss_dict = self.model(x, targets)
            loss = sum(loss for loss in loss_dict.values())

        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)


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
            img = F.adjust_gamma(img, gamma_factor, gain=1)
        return img
    

def collate(batch):
    img_list = []
    bx_list = []
    label_list = []
    for element in batch:
        img = element[0]
        img_list.append(img)
        bx_list.append(element[-1]['boxes'])
        label_list.append(element[-1]['labels'])

    img_tensor = torch.stack(img_list)

    return img_tensor, {'boxes': bx_list, 'labels': label_list}


def main():

    train_csv = '/vol/biomedic3/bglocker/mscproj24/mrm123/retinanet/csv_files/massfil_vindr_0ns/train.csv'
    val_csv = '/vol/biomedic3/bglocker/mscproj24/mrm123/retinanet/csv_files/massfil_vindr_0ns/val.csv'

    batch_size=16
    train_dataset = RCNNDataset(csv_file=train_csv)
    train_loader = utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate)
    val_dataset = RCNNDataset(csv_file=val_csv, augmentation=False)
    val_loader = utils.data.DataLoader(val_dataset, batch_size, shuffle=False, collate_fn=collate)

    model = FasterRCNNModel()

    wandb_logger = WandbLogger(project='Faster R-CNN', name='massfil_vindr_0ns_final') 

    checkpoint_callback = ModelCheckpoint(filename="{epoch}")
    checkpoint_callback_best = ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            filename="best_massfil_vindr_0ns_final",
        )

    early_stopping_callback = EarlyStopping(
            monitor="val_loss", 
            patience=10,         
            mode="min",          
            verbose=True     
    )

    trainer = L.Trainer(max_epochs=80, logger=wandb_logger, devices=2, callbacks=[checkpoint_callback, checkpoint_callback_best, early_stopping_callback])
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    trainer.validate(model=model, dataloaders=val_loader)


if __name__ == "__main__":
    main()