import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch.metrics as metrics
import torchmetrics

class MyIOU(torchmetrics.Metric):
    def __init__(self, mode='binary', threshold=0.5):
        super().__init__()
        self.threshold = threshold
        self.mode = mode

    
    def forward(self, pred, true):
        tp, fp, fn, tn = metrics.get_stats(pred, true, mode=self.mode, threshold=self.threshold)
        score = metrics.iou_score(tp, fp, fn, tn, reduction="micro")
        return score
    

class MyF1(nn.Module):
    def __init__(self, mode='binary', threshold=0.5):
        super().__init__()
        self.threshold = threshold
        self.mode = mode

    
    def forward(self, pred, true):
        tp, fp, fn, tn = metrics.get_stats(pred, true, mode=self.mode, threshold=self.threshold)
        score = metrics.f1_score(tp, fp, fn, tn, reduction="micro")
        return score


