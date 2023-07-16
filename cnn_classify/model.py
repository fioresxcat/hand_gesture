import torch
torch.set_float32_matmul_precision('high')

import pdb
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from easydict import EasyDict
from typing import List, Tuple, Dict, Optional, Union
import pytorch_lightning as pl
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score
import torchmetrics
from copy import deepcopy
import torchvision


class CNNClassifier(pl.LightningModule):
    def __init__(
        self, 
        n_classes,
        learning_rate: float,
        reset_optimizer: bool,
        acc: torchmetrics.Accuracy,
        criterion: nn.Module,
    ):
        super().__init__()
        self.model = torchvision.models.efficientnet_b0(weights=torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.model.classifier[1] = nn.Linear(in_features=1280, out_features=n_classes, bias=True)
        self.learning_rate = learning_rate
        self.reset_optimizer = reset_optimizer
        
        self.criterion = criterion
        self.train_acc = acc
        self.val_acc = deepcopy(acc)
        self.test_acc = deepcopy(acc)


    def compute_logits_and_losses(self, imgs, labels):
        logits = self.model(imgs)
        loss = self.criterion(logits, labels)
        return logits, loss


    def forward(self, imgs):
        return self.model(imgs)
    

    def step(self, batch, batch_idx, split):
        imgs, labels = batch
        logits, loss = self.compute_logits_and_losses(imgs, labels)

        acc = getattr(self, f'{split}_acc')
        acc(logits, labels)


        self.log_dict({
            f'{split}_loss': loss,
            f'{split}_acc': acc,
        }, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss


    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, 'train')
    
    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, 'val')
    
    def test_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, 'test')
    

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode=self.trainer.callbacks[0].mode,
            factor=0.2,
            patience=7,
        )

        return {
            'optimizer': opt,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': self.trainer.callbacks[0].monitor,
                'frequency': 1,
                'interval': 'epoch'
            }
        }
    

    def on_train_start(self) -> None:
        if self.reset_optimizer:
            opt = type(self.trainer.optimizers[0])(self.parameters(), **self.trainer.optimizers[0].defaults)
            self.trainer.optimizers[0].load_state_dict(opt.state_dict())
            print('Optimizer reseted')

if __name__ == '__main__':
    model = CNNClassifier(
        n_classes=10,
        learning_rate=1e-3,
        reset_optimizer=True,
        acc=None,
        criterion=None,
    )
    imgs = torch.rand(2, 3, 224, 224)
    out = model(imgs)
    print(out.shape)
    pdb.set_trace()