import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch.losses as losses


class DiceLoss(nn.Module):
    def __init__(self, mode='binary', from_logits=True):
        super().__init__()
        self.loss = losses.DiceLoss(mode=mode, from_logits=from_logits)

    def forward(self, y_pred, y_true):
        return self.loss(y_pred, y_true)




