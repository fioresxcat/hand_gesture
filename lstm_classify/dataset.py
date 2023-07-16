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
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)
import albumentations as A



class HandDataset(Dataset):
    def __init__(
        self,
        data_dir, # train, val or test dir
        mode, # train, val or test
        transforms, # albumentations
        img_suffix, # jpg or png
        gesture_list, # dict
        feat_dim,
        n_input_frames,
        n_sample_limit,
        augment_props
    ):
        super(HandDataset, self).__init__()
        self.data_dir = data_dir
        self.img_suffix = img_suffix
        self.mode = mode
        self.transforms = transforms
        self.gesture2idx = {gesture_name: i for i, gesture_name in enumerate(gesture_list)}
        self.feat_dim = feat_dim
        self.n_input_frames = n_input_frames
        self.n_sample_limit = n_sample_limit
        self.augment_props = augment_props

        self._init_paths_and_labels(self.img_suffix)
    

    def _init_paths_and_labels(self, img_suffix):
        self.ls_feat_paths, self.labels = [], []
        for gesture_name in os.listdir(self.data_dir):
            gesture_dir = os.path.join(self.data_dir, gesture_name)
            label = self.gesture2idx[gesture_name]

            for sample_name in os.listdir(gesture_dir):
                sample_dir = os.path.join(gesture_dir, sample_name)
                feat_paths = list(Path(sample_dir).glob(f'*.{img_suffix}'))

                if len(feat_paths) < self.n_input_frames:
                    n_missing_frame = self.n_input_frames - len(feat_paths)
                    feat_paths.extend([feat_paths[-1]] * n_missing_frame)
                else:
                    dist = len(feat_paths) // self.n_input_frames
                    valid_indices = [i for i in range(0, len(feat_paths), dist)]
                    if len(valid_indices) < self.n_input_frames:
                        n_missing_frame = self.n_input_frames - len(valid_indices)
                        valid_indices.extend([valid_indices[-1]] * n_missing_frame)
                    elif len(valid_indices) > self.n_input_frames:
                        valid_indices = valid_indices[:self.n_input_frames]
                    feat_paths = [feat_paths[i] for i in valid_indices]
                assert(len(feat_paths) == self.n_input_frames)

                self.ls_feat_paths.append(sorted(feat_paths))
                self.labels.append(label)
        
        print(f'Dataset {self.mode} has {len(self.ls_feat_paths)} samples')
        return self.ls_feat_paths, self.labels
        


    def __len__(self):
        return len(self.ls_feat_paths)
    

    def __getitem__(self, index):
        feat_paths = self.ls_feat_paths[index]
        cl = self.labels[index]

        feats = np.empty((self.n_input_frames, self.feat_dim), dtype=np.float32)
        for i, feat_path in enumerate(feat_paths):
            feat = np.load(feat_path)
            feats[i] = feat

        return torch.from_numpy(feats), cl



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

        if self.data_cfg.n_input_frames == 3:
            add_target = {'image0': 'image', 'image1': 'image'}
        elif self.data_cfg.n_input_frames == 5:
            add_target = {'image0': 'image', 'image1': 'image', 'image2': 'image', 'image3': 'image'}
        elif self.data_cfg.n_input_frames == 9:
            add_target = {'image0': 'image', 'image1': 'image', 'image2': 'image', 'image3': 'image', 'image4': 'image', 'image5': 'image', 'image6': 'image', 'image7': 'image'}
        elif self.data_cfg.n_input_frames == 15:
            add_target = {'image0': 'image', 'image1': 'image', 'image2': 'image', 'image3': 'image', 'image4': 'image', 'image5': 'image', 'image6': 'image', 'image7': 'image', 'image8': 'image', 'image9': 'image', 'image10': 'image', 'image11': 'image', 'image12': 'image', 'image13': 'image'}
        elif self.data_cfg.n_input_frames == 20:
            add_target = {'image0': 'image', 'image1': 'image', 'image2': 'image', 'image3': 'image', 'image4': 'image', 'image5': 'image', 'image6': 'image', 'image7': 'image', 'image8': 'image', 'image9': 'image', 'image10': 'image', 'image11': 'image', 'image12': 'image', 'image13': 'image', 'image14': 'image', 'image15': 'image', 'image16': 'image', 'image17': 'image', 'image18': 'image'}
        
        
        self.transforms = A.Compose(
            A.SomeOf([
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(p=0.5, shift_limit=0.1, scale_limit=0.15, rotate_limit=0, border_mode=cv2.BORDER_CONSTANT, value=0),
                A.ColorJitter(p=0.5, brightness=0.15, contrast=0.15, saturation=0.15, hue=0.07, always_apply=False),
                A.SafeRotate(p=0.5, limit=7, border_mode=cv2.BORDER_CONSTANT, value=0),
            ], n=2),
            additional_targets=add_target,
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