from torch.utils.data import Dataset
from PIL import Image
import torchvision
import torch
import pandas as pd
import numpy as np
import ast
import numbers
import optuna
import cv2
from skimage import exposure

from skimage import io
from skimage.color import gray2rgb
import os
from torch import optim, nn, utils, Tensor
from collections import OrderedDict
from torchvision.models.detection.image_list import ImageList

from torchvision import tv_tensors
from torchvision.ops import boxes as box_ops
import lightning as L
from retinanet import retinanet_resnet50_fpn, RetinaNetHead, RetinaNetClassificationHead, RetinaNetRegressionHead
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from anchor_utils import AnchorGenerator
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.loggers import WandbLogger
import torchvision.transforms.functional as F
import torchvision.transforms.v2 as T
from torchvision.ops import FeaturePyramidNetwork, MultiScaleRoIAlign


# Define the dataset class
class RetinaDataset(Dataset):
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
        img_path = self.imgs[idx]
        row_df = self.df.loc[self.df['image_path'] == img_path]
        numfind = row_df['numfind'].values[0]
        
        if numfind > 0 and pd.notna(row_df['label'].values[0]):
            bbox = torch.tensor(ast.literal_eval(row_df['bbox'].values[0]), dtype=torch.float32)
            # labels = torch.tensor(ast.literal_eval(row_df['label'].values[0]), dtype=torch.long)
            labels = torch.tensor([int(row_df['label'].values[0])], dtype=torch.long)
        else:
            bbox = torch.empty((0, 4), dtype=torch.float32)
            labels = torch.empty((0,), dtype=torch.long)

        to_tensor = torchvision.transforms.ToTensor()
        img = io.imread(img_path)
        img = (img / img.max())
        img = gray2rgb(img)
        image = to_tensor(img).float()

        if image.shape[-2:] != (1024, 768):
            resize_transform = T.Resize((1024, 768))
            image = resize_transform(image)

        boxes = tv_tensors.BoundingBoxes(bbox, format="XYXY", canvas_size=(1024, 768))

        if self.do_augment:
            image = self.photometric_augment(image)
            if numfind > 0:
                image, boxes = self.geometric_augment(image, boxes)
            else:
                image = self.geometric_augment(image)

        target = {'bbox': boxes, 'labels': labels}
        return image, target

    

# Define the LightningModule
class RetinaNet(L.LightningModule):
    def __init__(self, learning_rate=1e-5, weight_decay=1e-5):
        super().__init__()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.val_map_metric = MeanAveragePrecision()

        optimized_ratios = [1.0, 1.1912243160876392, 0.83947245409187]
        optimized_scales = [0.6701667817494404, 0.43826872391648763, 1.0929571608034148]
        
        # Define the base sizes (this corresponds to the feature map levels)
        anchor_sizes = [32, 64, 128, 256, 512]
        
        # Custom anchor generator using the optimized ratios and scales
        self.anchor_generator = AnchorGenerator(
            sizes=tuple((size * np.array(optimized_scales)).tolist() for size in anchor_sizes),
            aspect_ratios=tuple([optimized_ratios] * len(anchor_sizes))
        )
        
        self.retina = retinanet_resnet50_fpn(
            num_classes=4, 
            pretrained=False, 
            pretrained_backbone=True,
            anchor_generator=self.anchor_generator
        )

    def forward(self, images):
        return self.retina(images)

    def training_step(self, batch, batch_idx):
        x, target = batch
        loss_dict, _ = self.retina(x, target)
        loss = sum(loss for loss in loss_dict.values())
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, target = batch
        loss_dict, detections = self.retina(x, target)
        loss = sum(loss for loss in loss_dict.values())
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        return optimizer
    
    
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
        bx_list.append(element[-1]['bbox'])
        label_list.append(element[-1]['labels'])

    img_tensor = torch.stack(img_list)

    return img_tensor, {'bbox': bx_list, 'labels': label_list}


def main():

    train_csv = '/vol/biomedic3/bglocker/mscproj24/mrm123/retinanet/csv_files/multi_class/train_em.csv'
    val_csv = '/vol/biomedic3/bglocker/mscproj24/mrm123/retinanet/csv_files/multi_class/val_em.csv'

    batch_size=32
    train_dataset = RetinaDataset(csv_file=train_csv)
    train_loader = utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate)
    val_dataset = RetinaDataset(csv_file=val_csv, augmentation=False)
    val_loader = utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate)

    model = RetinaNet()

    wandb_logger = WandbLogger(project='Retinanet Multi Class', name='multi_class_embed_final') 

    checkpoint_callback = ModelCheckpoint(filename="{epoch}")
    checkpoint_callback_best = ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            filename="best_multi_class_embed_final",
        )

    trainer = L.Trainer(max_epochs=100, logger=wandb_logger, devices=1, callbacks=[checkpoint_callback, checkpoint_callback_best])
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    trainer.validate(model=model, dataloaders=val_loader)

def train_models():

    num_models = 4

    for i in range(num_models):

        train_csv = '/vol/biomedic3/bglocker/mscproj24/mrm123/retinanet/csv_files/massfil_vindr_0.2:1/train.csv'
        val_csv = '/vol/biomedic3/bglocker/mscproj24/mrm123/retinanet/csv_files/massfil_vindr_0.2:1/val.csv'

        batch_size=32
        train_dataset = RetinaDataset(csv_file=train_csv)
        train_loader = utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate)
        val_dataset = RetinaDataset(csv_file=val_csv, augmentation=False)
        val_loader = utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate)

        model = RetinaNet()

        wandb_logger = WandbLogger(project='Retinanet Models MV1-5-1', name=f'massfil_vindr_0.2:1_{i}') 

        checkpoint_callback = ModelCheckpoint(filename="{epoch}")
        checkpoint_callback_best = ModelCheckpoint(
                monitor="val_loss",
                mode="min",
                filename=f"best_massfil_vindr_0.2:1_{i}",
            )
        
        early_stopping_callback = EarlyStopping(
            monitor="val_loss", 
            patience=10,         
            mode="min",          
            verbose=True     
        )

        trainer = L.Trainer(max_epochs=100, logger=wandb_logger, devices=1, callbacks=[checkpoint_callback, checkpoint_callback_best, early_stopping_callback])
        trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        trainer.validate(model=model, dataloaders=val_loader)

if __name__ == "__main__":
    main()