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
        n_input_frames,
        n_sample_limit,
        input_size,
        augment_props
    ):
        super(HandDataset, self).__init__()
        self.data_dir = data_dir
        self.img_suffix = img_suffix
        self.mode = mode
        self.transforms = transforms
        self.gesture2idx = {gesture_name: i for i, gesture_name in enumerate(gesture_list)}
        self.n_input_frames = n_input_frames
        self.n_sample_limit = n_sample_limit
        self.input_size = input_size
        self.augment_props = augment_props
        self.normalize_video = NormalizeVideo(
            mean = [0.45, 0.45, 0.45],
            std = [0.225, 0.225, 0.225]
        )

        self.jpeg_reader = TurboJPEG()
        self._init_paths_and_labels(self.img_suffix)
    

    # def _init_paths_and_labels(self):
    #     """
    #         self.data_dir: dir / gesture_cl / vid_fn+start+end / frames.jpg
    #     """
    #     self.ls_img_paths, self.ls_txt_paths, self.labels = [], [], []
    #     for gesture_name in os.listdir(self.data_dir):
    #         gesture_cl = self.gesture2idx[gesture_name]
    #         gesture_dir = os.path.join(self.data_dir, gesture_name)
    #         for vid_fn_start_end in os.listdir(gesture_dir):
    #             vid_fr_dir = os.path.join(gesture_dir, vid_fn_start_end)
    #             img_paths = [os.path.join(vid_fr_dir, fn) for fn in os.listdir(vid_fr_dir) if fn.endswith('.jpg')]
    #             txt_paths = [os.path.join(vid_fr_dir, fn) for fn in os.listdir(vid_fr_dir) if fn.endswith('.txt')]

    #             valid_indices = []
    #             for i, txt_fp in enumerate(txt_paths):
    #                 with open(txt_fp) as f:
    #                     line = f.readlines()[0]
    #                 conf, x, y, w, h = line.split()
    #                 x, y, w, h = float(x), float(y), float(w), float(h)
    #                 if not (x==0 and y==0 and w==0 and h==0):
    #                     valid_indices.append(i)

    #             # pdb.set_trace()
    #             if len(valid_indices) < self.n_input_frames:
    #                 n_missing_frame = self.n_input_frames - len(valid_indices)
    #                 valid_indices.extend([valid_indices[-1]] * n_missing_frame)

    #             elif len(valid_indices) > self.n_input_frames:
    #                 final_indices = []
    #                 dist = len(valid_indices) // self.n_input_frames
    #                 for i, idx in enumerate(valid_indices):
    #                     if i == 0:
    #                         final_indices.append(idx)
    #                         continue

    #                     last_idx = final_indices[-1]
    #                     if idx - last_idx < dist and len(valid_indices) - i > self.n_input_frames:
    #                         continue

    #                     final_indices.append(idx)
    #                 valid_indices = final_indices[:self.n_input_frames]
                
    #             img_paths = [img_paths[i] for i in valid_indices]
    #             txt_paths = [txt_paths[i] for i in valid_indices]
    #             self.ls_img_paths.append(img_paths)
    #             self.ls_txt_paths.append(txt_paths)
    #             self.labels.append(gesture_cl)

    #     # # filter img paths so that each samples only contains n_input_frames images
    #     # for i, img_paths in enumerate(self.ls_img_paths):
    #     #     if len(img_paths) > self.n_input_frames:
    #     #         d = len(img_paths) // self.n_input_frames
    #     #         new_img_paths = [img_paths[i] for i in range(0, len(img_paths), d)]
    #     #         if len(new_img_paths) < self.n_input_frames:
    #     #             n_missing_frames = self.n_input_frames - len(new_img_paths)
    #     #             new_img_paths.extend(img_paths[-n_missing_frames:])
    #     #         elif len(new_img_paths) > self.n_input_frames:
    #     #             new_img_paths = new_img_paths[:self.n_input_frames]
    #     #         self.ls_img_paths[i] = new_img_paths
    #     #         # pdb.set_trace()
    #     #     elif len(img_paths) < self.n_input_frames:
    #     #         n_missing_frames = self.n_input_frames - len(img_paths)
    #     #         new_img_paths = img_paths
    #     #         for _ in range(n_missing_frames):
    #     #             new_img_paths.append(img_paths[-1])
    #     #         self.ls_img_paths[i] = new_img_paths


    #     # pdb.set_trace()
    #     assert all([len(img_paths)==self.n_input_frames for img_paths in self.ls_img_paths])
    #     self.ls_img_paths = [sorted(img_paths) for img_paths in self.ls_img_paths]

    def _init_paths_and_labels(self, img_suffix):
        self.ls_img_paths, self.labels = [], []
        for gesture_name in os.listdir(self.data_dir):
            gesture_dir = os.path.join(self.data_dir, gesture_name)
            label = self.gesture2idx[gesture_name]

            for sample_name in os.listdir(gesture_dir):
                sample_dir = os.path.join(gesture_dir, sample_name)
                img_paths = list(Path(sample_dir).glob(f'*.{img_suffix}'))

                if len(img_paths) < self.n_input_frames:
                    n_missing_frame = self.n_input_frames - len(img_paths)
                    img_paths.extend([img_paths[-1]] * n_missing_frame)
                else:
                    dist = len(img_paths) // self.n_input_frames
                    valid_indices = [i for i in range(0, len(img_paths), dist)]
                    if len(valid_indices) < self.n_input_frames:
                        n_missing_frame = self.n_input_frames - len(valid_indices)
                        valid_indices.extend([valid_indices[-1]] * n_missing_frame)
                    elif len(valid_indices) > self.n_input_frames:
                        valid_indices = valid_indices[:self.n_input_frames]
                    img_paths = [img_paths[i] for i in valid_indices]
                assert(len(img_paths) == self.n_input_frames)

                self.ls_img_paths.append(sorted(img_paths))
                self.labels.append(label)
        
        print(f'Dataset {self.mode} has {len(self.ls_img_paths)} samples')
        return self.ls_img_paths, self.labels
        


    def __len__(self):
        return len(self.ls_img_paths)
    

    def mask_img_from_txt(self, img_paths, txt_paths):
        # pdb.set_trace()
        imgs = []
        xmin, ymin, xmax, ymax = None, None, None, None
        for i, fp in enumerate(img_paths):
            with open(fp, 'rb') as in_file:
                orig_img = self.jpeg_reader.decode(in_file.read(), 0)  # already rgb images
            txt_fp  = txt_paths[i]
            with open(txt_fp) as f:
                line = f.readlines()[0]
            conf, x, y, w, h = line.split()
            x, y, w, h = float(x), float(y), float(w), float(h)
            if not (x==0 and y==0 and w==0 and h==0):
                xmin = int((x - w/2) * orig_img.shape[1])
                xmax = int((x + w/2) * orig_img.shape[1])
                ymin = int((y - h/2) * orig_img.shape[0])
                ymax = int((y + h/2) * orig_img.shape[0])
                mask = np.zeros(shape=orig_img.shape[:2], dtype=np.uint8)
                mask[ymin:ymax, xmin:xmax] = 1
                masked_img = orig_img * mask[..., None]
            else:
                masked_img = orig_img
                    
            imgs.append(masked_img)
        
        os.makedirs('test', exist_ok=True)
        for i, img in enumerate(imgs):
            img_fp = img_paths[i]
            cv2.imwrite(f'test/{Path(img_fp).stem}_{i}.jpg', img)
        pdb.set_trace()
        return imgs
            


    def __getitem__(self, index):
        img_paths = self.ls_img_paths[index]
        cl = self.labels[index]

        input_imgs = []
        for i, img_fp in enumerate(img_paths):
            # img = self.jpeg_reader.decode(open(str(img_fp), 'rb').read(), 0)    # rgb images
            img = cv2.imread(str(img_fp))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.input_size)
            input_imgs.append(img)
        
        # os.makedirs('test', exist_ok=True)
        # for i, img in enumerate(input_imgs):
        #     cv2.imwrite(f'test/a_{i}.jpg', img)

        transformed_imgs = np.stack(input_imgs, axis=0)

        # normalize
        transformed_imgs = torch.from_numpy(transformed_imgs)
        transformed_imgs = transformed_imgs.permute(3, 0, 1, 2) # shape 3 x 9 x h x w
        transformed_imgs = transformed_imgs / 255.0
        transformed_imgs = self.normalize_video(transformed_imgs)

        return transformed_imgs, cl



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