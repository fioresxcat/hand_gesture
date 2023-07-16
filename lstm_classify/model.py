from typing import Any, List
from easydict import EasyDict
import pdb
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchmetrics
from copy import deepcopy
import pytorch_lightning as pl




class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_sizes, n_classes, dropout=0.2):
        super(LSTMModel, self).__init__()

        self.lstm1 = nn.LSTM(input_size=input_dim, hidden_size=hidden_sizes[0], num_layers=1, batch_first=True, dropout=dropout)
        self.lstm2 = nn.LSTM(input_size=hidden_sizes[0], hidden_size=hidden_sizes[1], num_layers=1, batch_first=True, dropout=dropout)
        self.lstm3 = nn.LSTM(input_size=hidden_sizes[1], hidden_size=hidden_sizes[2], num_layers=1, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_sizes[2], n_classes)

        
    def forward(self, x):
        x, (hn, cn) = self.lstm1(x)
        x, (hn, cn) = self.lstm2(x)
        x, (hn, cn) = self.lstm3(x)
        out = self.fc(x[:, -1, :])        # x[:, -1, :] is the last time step
        return out
    

class LSTMModule(pl.LightningModule):
    def __init__(
        self, 
        model: nn.Module,
        n_classes,
        learning_rate: float,
        reset_optimizer: bool,
        acc: torchmetrics.Accuracy,
        criterion: nn.Module,
    ):
        super().__init__()
        self.model = model
        self.n_classes = n_classes
        self.learning_rate = learning_rate
        self.reset_optimizer = reset_optimizer

        self.criterion = criterion
        self.train_acc = acc
        self.val_acc = deepcopy(acc)
        self.test_acc = deepcopy(acc)


    def compute_logits_and_losses(self, features, labels):
        """
            features: shape batch x 15 x 1024
        """
        logits = self.model(features)   # logits shape batch x 14
        loss = self.criterion(logits, labels)
        return logits, loss


    def forward(self, imgs):
        return self.model(imgs)
    

    def step(self, batch, batch_idx, split):
        features, labels = batch
        logits, loss = self.compute_logits_and_losses(features, labels)

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
    model = LSTMModel(1024, [256, 128, 64], 14)
    print(model)
    print('num params: ', sum(p.numel() for p in model.parameters()))
    x = torch.randn(2, 16, 1024)
    out = model(x)
    print(out.shape)


