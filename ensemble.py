import torch
import os
from torch import optim, nn, utils, Tensor
from torchvision.ops import nms, box_iou
from skimage import io
from skimage.color import gray2rgb
from torchvision.ops import nms 
import lightning as L
from retinanet import retinanet_resnet50_fpn

from model import RetinaDataset, RetinaNet, collate
from results import evaluate_model_retina


# Define RetinaNet Ensemble Class
class RetinaNetEnsemble(L.LightningModule):
    def __init__(self, models, model_weights=None):
        super().__init__()
        self.models = models
        # Set default weights if not provided
        self.model_weights = model_weights if model_weights else [1.0] * len(models)

    def forward(self, images):
        all_predictions = []
        
        # Get predictions from each model
        for model in self.models:
            model.eval()
            with torch.no_grad():
                predictions = model(images)
                all_predictions.append(predictions)
        
        # Combine predictions using weighted averaging
        combined_predictions = self.combine_predictions(all_predictions)
        
        return combined_predictions

    def combine_predictions(self, all_predictions):
        final_boxes = []
        final_scores = []
        final_labels = []

        for i in range(len(all_predictions[0][1])):
            boxes = []
            scores = []
            labels = []
            
            # Apply weights to the predictions from each model
            for model_idx, preds in enumerate(all_predictions):
                model_boxes = preds[1][i]['bbox']
                model_scores = preds[1][i]['scores']
                model_labels = preds[1][i]['labels']

                # Apply the weight of the current model to its predictions
                weight = self.model_weights[model_idx]
                
                if model_scores.numel() > 0:
                    boxes.append(model_boxes)
                    scores.append(model_scores * weight)  # Scale scores by the model weight
                    labels.append(model_labels)

            if len(boxes) > 0:
                # Combine the predictions from all models
                boxes = torch.cat(boxes, dim=0)
                scores = torch.cat(scores, dim=0)
                labels = torch.cat(labels, dim=0)

                # Perform NMS to reduce overlapping boxes
                keep = nms(boxes, scores, iou_threshold=0.5)

                boxes = boxes[keep]
                scores = scores[keep]
                labels = labels[keep]
            else:
                # Else no object 
                boxes = torch.empty((0, 4))
                scores = torch.empty((0,))
                labels = torch.empty((0,), dtype=torch.int64)

            final_boxes.append(boxes)
            final_scores.append(scores)
            final_labels.append(labels)

        return [{}, [{'bbox': boxes, 'scores': scores, 'labels': labels} for boxes, scores, labels in zip(final_boxes, final_scores, final_labels)]]

# Function to load models to ensemble
def load_models(checkpoint_paths, ratios, scales):
    models = []
    for path in checkpoint_paths:
        model = RetinaNet.load_from_checkpoint(path, ratios=ratios, scales=scales)
        models.append(model)
    return models


# Paths to model checkpoints
checkpoint_paths_1 = [
    '/vol/biomedic3/bglocker/mscproj24/mrm123/slurm_scripts/Retinanet Models/vsrt8qac/checkpoints/best_massfil_vindr_0ns_3.ckpt',
    '/vol/biomedic3/bglocker/mscproj24/mrm123/slurm_scripts/Retinanet/ggs8ys1v/checkpoints/best_massfil_vindr_2:1_ptFe.ckpt',
    '/vol/biomedic3/bglocker/mscproj24/mrm123/slurm_scripts/Retinanet Models M1-1/wr77ji8u/checkpoints/best_massfil_vindr_1:1_4.ckpt'
]

models = load_models(checkpoint_paths_1, ratios=[1.0, 1.1912243160876392, 0.83947245409187], scales=[0.6701667817494404, 0.43826872391648763, 1.0929571608034148])
test_dataset = RetinaDataset(csv_file='csv_files/massfil_vindr_1:1/test.csv', augmentation=False)
test_loader_vindr = utils.data.DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=collate)
model_weights = [0.5, 0.6, 0.6] # Optimized weights to val set
ensemble_model = RetinaNetEnsemble(models, model_weights=model_weights)


def combined_metric(mAP, specificity, alpha=0.5):
    """
    Function to determine combined metric of mAP and specificity
    """
    # Calculate the combined metric
    combined_score = alpha * mAP + (1 - alpha) * specificity
    return combined_score

alpha = 0.5  

best_combined_score = -float('inf')
best_weights = None

model_weights_values = [
    [0.5, 1, 1],
    [0.7, 0.8, 0.9],
    # Use other model weights here
]

val_dataset = RetinaDataset(csv_file='csv_files/massfil_vindr_1:1/val.csv', augmentation=False)
val_dataloader = utils.data.DataLoader(val_dataset, batch_size=8, shuffle=True, collate_fn=collate)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for weights in model_weights_values:
    ensemble_model.model_weights = weights
    
    # Calculate mAP and precision on negative samples
    mAP, iou, precision_on_negative_samples = evaluate_model_retina(ensemble_model, val_dataloader, device=device)
    
    # Compute the combined metric
    combined_score = combined_metric(mAP['map'], precision_on_negative_samples, alpha=alpha)
    
    # Track the best weights based on combined metric
    if combined_score > best_combined_score:
        best_combined_score = combined_score
        best_weights = weights

print(f"Best Weights: {best_weights}, Best Combined Score: {best_combined_score}")
