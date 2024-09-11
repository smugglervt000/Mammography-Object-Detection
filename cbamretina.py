from torch.utils.data import Dataset
import torchvision
import torch
import pandas as pd
import numpy as np
import ast
import numbers

import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

from skimage import io
from skimage.color import gray2rgb
from torch import optim, nn
from torchvision.ops.feature_pyramid_network import LastLevelP6P7

from torchvision import tv_tensors
from torchvision.ops import boxes as box_ops
from torchvision.models.detection._utils import Matcher
import lightning as L
from retinanet import retinanet_resnet50_fpn, RetinaNetHead, RetinaNetClassificationHead, RetinaNetRegressionHead
from lightning.pytorch.callbacks import ModelCheckpoint
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from anchor_utils import AnchorGenerator
from lightning.pytorch.loggers import WandbLogger
import torchvision.transforms.functional as F_t
import torchvision.transforms.v2 as T
from torchvision.ops import FeaturePyramidNetwork
from model import RetinaDataset
from model import collate


# Define CBAM blocks
class CBAMBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(CBAMBlock, self).__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        x = self.channel_attention(x) * x
        x = self.spatial_attention(x) * x
        return x

# Define Channel Attention Block
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // reduction, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // reduction, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


# Define Spatial Attention Block
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)
    

# Define RetinaNet Class
class RetinaNetLightning(L.LightningModule):
    def __init__(self, learning_rate=1e-5, weight_decay=1e-5):
        super().__init__()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.val_map_metric = MeanAveragePrecision()

        # Optimized ratios and scales for anchors
        optimized_ratios = [1.0, 1.1912243160876392, 0.83947245409187]
        optimized_scales = [0.6701667817494404, 0.43826872391648763, 1.0929571608034148]
        
        # Define the base sizes (this corresponds to the feature map levels)
        anchor_sizes = [32, 64, 128, 256, 512, 1024] 

        self.anchor_generator = AnchorGenerator(
            sizes=tuple((size * np.array(optimized_scales)).tolist() for size in anchor_sizes),  
            aspect_ratios=(optimized_ratios,) * len(anchor_sizes) 
        )
 
        # Initialize RetinaNet with CBAM using RetinaNetWithCBAM Class
        self.retina = RetinaNetWithCBAM(
            num_classes=1, 
            pretrained_backbone=True,
            anchor_generator=self.anchor_generator
        )

    def forward(self, images):
        return self.retina(images)

    def training_step(self, batch, batch_idx):
        x, target = batch
        loss_dict = self.retina(x, target)
        loss = sum(loss for loss in loss_dict.values())
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, target = batch
        self.train()
        with torch.no_grad():
            loss_dict = self.retina(x, target)
            loss = sum(loss for loss in loss_dict.values())
        self.log("val_loss", loss)
        return loss

    
    def postprocess(self, cls_logits, bbox_regression, anchors):
        """
        This function postprocesses the class logits, bbox regression, and anchor tensors.
        """
        results = []
        for logits, bbox_reg, anchor in zip(cls_logits, bbox_regression, anchors):
            scores = torch.sigmoid(logits) 

            scores, _ = scores.max(dim=1) # Reshape scores tensor

            boxes = self.decode_boxes(anchor, bbox_reg)

            # Apply NMS 
            keep = nms(boxes, scores, iou_threshold=0.5) 

            results.append({
                'boxes': boxes[keep],
                'scores': scores[keep]
            })
        
        return results

    
    def decode_boxes(self, anchors, bbox_regression):
        """
        Box decoder to get model predictions.
        """
        anchors = anchors.to(bbox_regression.device) 
        x_min_anchors = anchors[:, 0]
        y_min_anchors = anchors[:, 1]
        x_max_anchors = anchors[:, 2]
        y_max_anchors = anchors[:, 3]

        widths = x_max_anchors - x_min_anchors
        heights = y_max_anchors - y_min_anchors
        center_x_anchors = x_min_anchors + 0.5 * widths
        center_y_anchors = y_min_anchors + 0.5 * heights

        dx = bbox_regression[:, 0]
        dy = bbox_regression[:, 1]
        dw = bbox_regression[:, 2]
        dh = bbox_regression[:, 3]

        center_x = center_x_anchors + dx * widths
        center_y = center_y_anchors + dy * heights

        pred_widths = widths * torch.exp(dw)
        pred_heights = heights * torch.exp(dh)

        pred_x_min = center_x - 0.5 * pred_widths
        pred_y_min = center_y - 0.5 * pred_heights
        pred_x_max = center_x + 0.5 * pred_widths
        pred_y_max = center_y + 0.5 * pred_heights

        decoded_boxes = torch.stack([pred_x_min, pred_y_min, pred_x_max, pred_y_max], dim=1)

        return decoded_boxes

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        return optimizer
    
# RetinaNet with CBAM architecture
class RetinaNetWithCBAM(nn.Module):
    def __init__(self, num_classes=1, pretrained_backbone=True, anchor_generator=None):
        super(RetinaNetWithCBAM, self).__init__()

        # Use ResNet50 pretrained backbone
        backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1 if pretrained_backbone else None)

        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool

        self.layer1 = backbone.layer1 
        self.layer2 = backbone.layer2  
        self.layer3 = backbone.layer3 
        self.layer4 = backbone.layer4  

        # Define CBAM blocks for each feature map level
        self.cbam1 = CBAMBlock(256) 
        self.cbam2 = CBAMBlock(512)  
        self.cbam3 = CBAMBlock(1024) 
        self.cbam4 = CBAMBlock(2048) 

        in_channels_list = [256, 512, 1024, 2048] 
        out_channels = 256 
        # Include P6P7 layers
        self.fpn = FeaturePyramidNetwork(in_channels_list=in_channels_list, out_channels=out_channels, extra_blocks=LastLevelP6P7(256, 256))

        # Define the RetinaNet classification and regression heads
        self.classification_head = RetinaNetClassificationHead(out_channels, num_anchors=9, num_classes=num_classes)
        self.regression_head = RetinaNetRegressionHead(out_channels, num_anchors=9)

        self.proposal_matcher = Matcher(
            high_threshold=0.5,   
            low_threshold=0.4,   
            allow_low_quality_matches=True
        )

        # Set anchor generator
        self.anchor_generator = anchor_generator

    def forward(self, images, targets=None):
        """
        Forward function for CBAM implementation.
        """
        x = self.conv1(images)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        c2 = self.layer1(x)  
        c3 = self.layer2(c2)  
        c4 = self.layer3(c3)  
        c5 = self.layer4(c4)  

        c2 = self.cbam1(c2)  
        c3 = self.cbam2(c3)  
        c4 = self.cbam3(c4) 
        c5 = self.cbam4(c5) 

        fpn_features = self.fpn({
            'C2': c2,
            'C3': c3,
            'C4': c4,
            'C5': c5,
        })

        feature_maps = list(fpn_features.values())  

        # Get class logits and bbox regression
        cls_logits = self.classification_head(feature_maps) 
        bbox_regression = self.regression_head(feature_maps) 

        if self.training and targets is not None:
            anchors = self.anchor_generator(images, feature_maps)  

            loss_dict = self.compute_loss(cls_logits, bbox_regression, targets, anchors)
            return loss_dict
        else:
            return cls_logits, bbox_regression

    def compute_loss(self, cls_logits, bbox_regression, targets, anchors):
        """
        Use loss from classification and regression heads.
        """
        matched_idxs = []
        for anchors_per_image, targets_per_image in zip(anchors, targets['bbox']):
            if torch.all(targets_per_image == -1):
                matched_idxs.append(
                    torch.full((anchors_per_image.size(0),), -1, dtype=torch.int64, device=anchors_per_image.device)
                )
                continue
            match_quality_matrix = box_ops.box_iou(targets_per_image, anchors_per_image)
            matched_idxs.append(self.proposal_matcher(match_quality_matrix))
        classification_loss = self.classification_head.compute_loss(
            targets, {"cls_logits": cls_logits}, matched_idxs
        )

        regression_loss = self.regression_head.compute_loss(
            targets, {"bbox_regression": bbox_regression}, anchors, matched_idxs
        )

        return {'classification_loss': classification_loss, 'regression_loss': regression_loss}


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


def main():
    train_csv = '/vol/biomedic3/bglocker/mscproj24/mrm123/retinanet/csv_files/massfil_vindr_0ns/train.csv'
    val_csv = '/vol/biomedic3/bglocker/mscproj24/mrm123/retinanet/csv_files/massfil_vindr_0ns/val.csv'

    batch_size = 16
    train_dataset = RetinaDataset(csv_file=train_csv)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate)
    val_dataset = RetinaDataset(csv_file=val_csv, augmentation=False)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate)

    model = RetinaNetLightning()

    wandb_logger = WandbLogger(project='Retinanet CBAM', name='massfil_vindr_0ns_cbam') 

    checkpoint_callback = ModelCheckpoint(filename="{epoch}")
    checkpoint_callback_best = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        filename="best_massfil_vindr_0ns_cbam",
    )

    trainer = L.Trainer(
        max_epochs=100,
        logger=wandb_logger,
        devices=2,
        callbacks=[checkpoint_callback, checkpoint_callback_best]
    )
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    trainer.validate(model=model, dataloaders=val_loader)


if __name__ == "__main__":
    main()
