import os
import json
from pathlib import Path
import torch
torch.set_float32_matmul_precision('high')
import numpy as np
import yaml
import shutil
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar, LearningRateMonitor, RichModelSummary
from pytorch_lightning.cli import LightningCLI
from dataset import HandDataModule
from model import CNNClassifier


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_argument("--save_config_overwrite", default=False)
        parser.add_argument("--save_config_callback", default='')


    def before_instantiate_classes(self) -> None:
        self.save_config_kwargs['overwrite'] = self.config[self.config.subcommand].save_config_overwrite
        if self.config[self.config.subcommand].save_config_callback == 'None':
            self.save_config_callback = None



def cli_main():
    cli = MyLightningCLI(
        CNNClassifier,
        HandDataModule,
        parser_kwargs={"parser_mode": "omegaconf"},
        save_config_overwrite=False
    )


if __name__ == '__main__':
    cli_main()
