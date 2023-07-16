import numpy as np
from pathlib import Path
import os
import pdb
from easydict import EasyDict
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import cv2
import torch
import pytorch_lightning as pl
from PIL import Image
import time
from turbojpeg import TurboJPEG
import albumentations as A



class HandDataset(Dataset):
    def __init__(
        self,
        data_dir, # train, val or test dir
        mode, # train, val or test
        transforms, # albumentations
        img_suffix, # jpg or png
        gesture_list, # dict
        n_frame_per_sample,
        input_size,
        augment_props
    ):
        super(HandDataset, self).__init__()
        self.data_dir = data_dir
        self.img_suffix = img_suffix
        self.mode = mode
        self.transforms = transforms
        self.gesture2idx = {gesture_name: i for i, gesture_name in enumerate(gesture_list)}
        self.n_frame_per_sample = n_frame_per_sample
        self.input_size = input_size
        self.augment_props = augment_props

        self.jpeg_reader = TurboJPEG()
        self._init_paths_and_labels(self.img_suffix)
    

    def _init_paths_and_labels(self, img_suffix):
        self.img_paths, self.labels = [], []
        for gesture_name in os.listdir(self.data_dir):
            gesture_dir = os.path.join(self.data_dir, gesture_name)
            label = self.gesture2idx[gesture_name]

            for sample_name in os.listdir(gesture_dir):
                sample_dir = os.path.join(gesture_dir, sample_name)
                img_paths = list(Path(sample_dir).glob(f'*.{img_suffix}'))

                if len(img_paths) > self.n_frame_per_sample:
                    img_paths = np.random.choice(img_paths, size=self.n_frame_per_sample, replace=False)

                self.img_paths.extend(sorted(img_paths))
                self.labels.extend([label]*len(img_paths))
        
        print(f'Dataset {self.mode} has {len(self.img_paths)} samples')
        return self.img_paths, self.labels
        


    def __len__(self):
        return len(self.img_paths)
    

    def __getitem__(self, index):
        img_fp = self.img_paths[index]
        cl = self.labels[index]

        img = self.jpeg_reader.decode(open(img_fp, 'rb').read(), 0)
        img = cv2.resize(img, self.input_size)
        img = img/255.

        return torch.from_numpy(img).permute(2, 0, 1).float(), cl


class HandDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_dir,
        val_dir, 
        test_dir,
        predict_dir,
        data_cfg: dict, 
        training_cfg: dict
    ):
        super(HandDataModule, self).__init__()
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir
        self.predict_dir = predict_dir
        self.data_cfg = EasyDict(data_cfg)
        self.training_cfg = EasyDict(training_cfg)

        self.transforms = A.Compose(
            A.SomeOf([
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(p=0.5, shift_limit=0.1, scale_limit=0.15, rotate_limit=0, border_mode=cv2.BORDER_CONSTANT, value=0),
                A.ColorJitter(p=0.5, brightness=0.15, contrast=0.15, saturation=0.15, hue=0.07, always_apply=False),
                A.SafeRotate(p=0.5, limit=7, border_mode=cv2.BORDER_CONSTANT, value=0),
            ], n=2),
        )

    
    def setup(self, stage):
        if stage == 'fit' or stage == 'validate':
            self.train_ds = HandDataset(data_dir=self.train_dir, mode='train', transforms=self.transforms, **self.data_cfg)
            self.val_ds = HandDataset(data_dir=self.val_dir, mode='val', transforms=None, **self.data_cfg)
        elif stage == 'test':
            self.test_ds = HandDataset(self.test_dir, transforms=None, mode='test', **self.data_cfg)
        elif stage == 'predict':
            self.predict_ds = HandDataset(self.predict_dir, transforms=None, mode='predict', **self.data_cfg)


    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.train_ds, 
            batch_size=self.training_cfg.bs, 
            shuffle=self.training_cfg.shuffle_train, 
            num_workers=self.training_cfg.num_workers,
            pin_memory=False
        )


    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.val_ds, 
            batch_size=self.training_cfg.bs, 
            shuffle=False, 
            num_workers=self.training_cfg.num_workers,
            pin_memory=False
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds, 
            batch_size=self.training_cfg.bs, 
            shuffle=False, 
            num_workers=self.training_cfg.num_workers,
            pin_memory=False
        )
    
    def predict_dataloader(self):
        return DataLoader(
            self.predict_ds, 
            batch_size=self.training_cfg.bs, 
            shuffle=False, 
            num_workers=self.training_cfg.num_workers,
            pin_memory=False
        )



if __name__ == '__main__':
    import yaml
    from easydict import EasyDict

    with open('config_3d.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = EasyDict(config)
    config.data.training_cfg.num_workers = 0

    ds_module = HandDataModule(**config.data)
    ds_module.setup('validate')

    for i, item in enumerate(ds_module.val_ds):
        imgs, cl = item
        print(imgs.shape)
        pdb.set_trace()