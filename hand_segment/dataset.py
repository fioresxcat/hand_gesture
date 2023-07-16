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



class SegmentDataset(Dataset):
    def __init__(
        self,
        data_dir, # train, val or test dir
        mode, # train, val or test
        transforms, # albumentations
        n_sample_per_vid,
        input_size,
        augment_props
    ):
        super(SegmentDataset, self).__init__()
        self.data_dir = data_dir
        self.mode = mode
        self.transforms = transforms
        self.n_sample_per_vid = n_sample_per_vid
        self.input_size = input_size
        self.augment_props = augment_props

        self.jpeg_reader = TurboJPEG()
        self._init_paths_and_labels()
    

    def _init_paths_and_labels(self):
        """
            self.data_dir: dir / gesture_cl / vid_fn+start+end / frames.jpg
        """
        self.img_paths, self.mask_paths = [], []
        image_dir = os.path.join(self.data_dir, 'images')
        mask_dir = os.path.join(self.data_dir, 'masks')
        for vid_fn in os.listdir(image_dir):
            vid_dir = os.path.join(image_dir, vid_fn)
            img_paths = [os.path.join(vid_dir, fn) for fn in os.listdir(vid_dir) if fn.endswith('.jpg')]
            if len(img_paths) > self.n_sample_per_vid:
                img_paths = np.random.choice(img_paths, self.n_sample_per_vid, replace=False)
            mask_paths = [fp.replace('/images/', '/masks/').replace('.jpg', '.png') for fp in img_paths]

            invalid_indices = []
            for i in range(len(img_paths)):
                if not os.path.exists(img_paths[i]) or not os.path.exists(mask_paths[i]):
                    invalid_indices.append(i)
            img_paths = [img_paths[i] for i in range(len(img_paths)) if i not in invalid_indices]
            mask_paths = [mask_paths[i] for i in range(len(mask_paths)) if i not in invalid_indices]
            
            self.img_paths.extend(img_paths)
            self.mask_paths.extend(mask_paths)
        
        return self.img_paths, self.mask_paths


    def __len__(self):
        return len(self.img_paths)
    

    def __getitem__(self, index):
        img_fp = self.img_paths[index]
        mask_fp = self.mask_paths[index]

        img = self.jpeg_reader.decode(open(img_fp, 'rb').read(), 0)   # rgb images
        img = cv2.resize(img, self.input_size)
        img = img / 255.

        mask = cv2.imread(mask_fp, 0)   # gray images
        mask = cv2.resize(mask, self.input_size, interpolation=cv2.INTER_NEAREST)
        mask = mask / 255.

        return torch.from_numpy(img).permute(2, 0, 1).float(), torch.from_numpy(mask).unsqueeze(0).int()




class SegmentDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_dir,
        val_dir, 
        test_dir,
        predict_dir,
        data_cfg: dict, 
        training_cfg: dict
    ):
        super(SegmentDataModule, self).__init__()
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
            self.train_ds = SegmentDataset(data_dir=self.train_dir, mode='train', transforms=self.transforms, **self.data_cfg)
            self.val_ds = SegmentDataset(data_dir=self.val_dir, mode='val', transforms=None, **self.data_cfg)
        elif stage == 'test':
            self.test_ds = SegmentDataset(self.test_dir, transforms=None, mode='test', **self.data_cfg)
        elif stage == 'predict':
            self.predict_ds = SegmentDataset(self.predict_dir, transforms=None, mode='predict', **self.data_cfg)


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

    with open('config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = EasyDict(config)
    config.data.training_cfg.num_workers = 0

    ds_module = SegmentDataModule(**config.data)
    ds_module.setup('validate')

    for i, item in enumerate(ds_module.val_dataloader()):
        imgs, masks = item
        print(imgs.shape, masks.shape)
        img = imgs[-1]
        mask = masks[-1]

        img = img.permute(1, 2, 0).numpy()
        img = (img*255).astype(np.uint8)
        cv2.imwrite('img.jpg', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        mask = mask.squeeze(0).numpy()
        mask = (mask*255).astype(np.uint8)
        cv2.imwrite('mask.png', mask)

        pdb.set_trace()