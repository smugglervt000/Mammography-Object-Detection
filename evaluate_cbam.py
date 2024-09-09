import torchvision
import torch
from tqdm import tqdm 

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.ops import nms, box_iou

from torchvision import tv_tensors
from torchvision.ops import boxes as box_ops
import lightning as L
from retinanet import retinanet_resnet50_fpn, RetinaNetHead, RetinaNetClassificationHead, RetinaNetRegressionHead
from lightning.pytorch.callbacks import ModelCheckpoint
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from model import RetinaDataset, collate
from cbamretina import RetinaNetLightning


def evaluate_model(model, test_loader):
    model.eval()
    map_metric = MeanAveragePrecision(iou_type='bbox')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating", leave=True):
            x, target = batch
            x = x.to(device) 

            new_targets = []
            for i in range(len(target['bbox'])):
                bbox_tensor = target['bbox'][i].to(device) if hasattr(target['bbox'][i], 'to') else torch.empty((0, 4)).to(device)

                labels_tensor = target['labels'][i].to(device)

                new_targets.append({'boxes': bbox_tensor, 'labels': labels_tensor})

            cls_logits, bbox_regression, feature_maps = model(x) 

            # Generate anchors and postprocess
            anchors = model.anchor_generator(x, feature_maps=feature_maps) 
            results = model.postprocess(cls_logits, bbox_regression, anchors)

            preds = [{'boxes': r['boxes'].cpu(), 'scores': r['scores'].cpu(), 'labels': torch.ones(r['boxes'].size(0)).cpu()} for r in results]
            targets = [{'boxes': t['boxes'].cpu(), 'labels': t['labels'].cpu()} for t in new_targets]

            # Update the mAP metric
            map_metric.update(preds, targets)

    # Compute mAP metrics
    final_map = map_metric.compute()

    print(f"mAP: {final_map['map']:.4f}")
    print(f"mAP_50: {final_map['map_50']:.4f}")

    return final_map


def evaluate():

    checkpoint_path = '/vol/biomedic3/bglocker/mscproj24/mrm123/slurm_scripts/Retinanet CBAM/f1pcteil/checkpoints/best_massfil_vindr_0ns_cbam.ckpt'
    model = RetinaNetLightning.load_from_checkpoint(checkpoint_path)

    test_dataset = RetinaDataset(csv_file='/vol/biomedic3/bglocker/mscproj24/mrm123/retinanet/csv_files/massfil_vindr_0ns/test.csv', augmentation=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=False, collate_fn=collate)

    final_map = evaluate_model(model, test_loader)


if __name__ == "__main__":
    evaluate()