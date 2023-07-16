import torch
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path
import os
import shutil
import pdb

import numpy as np
from easydict import EasyDict
import cv2
from PIL import Image

from unet import UnetModule


if __name__ == '__main__':
    from sys import argv
    import yaml
    import segmentation_models_pytorch as smp

    data_dir = '/data3/users/tungtx2/hand_gesture/classification_data'
    ckpt_path = 'ckpt/exp2/epoch=29-train_loss=0.067-val_loss=0.143-train_iou=0.876-val_iou=0.753-train_f1=0.934-val_f1=0.859.ckpt'
    input_size = (320, 256)
    threshold = 0.3

    config_path = Path(ckpt_path).parent / 'config.yaml'
    with open(config_path, 'r') as f:
        config = EasyDict(yaml.load(f, Loader=yaml.FullLoader))

    model = smp.Unet(**config.model.model.init_args)
    state = torch.load(ckpt_path, map_location='cpu')['state_dict']
    items = list(state.items())
    for k, v in items:
        state[k.replace('model.', '')] = state.pop(k)
    model.load_state_dict(state)
    print('Model loaded !')
    model.eval().to('cuda')

    cnt = 0
    for img_fp in Path(data_dir).rglob('*.jpg'):
        orig_img = cv2.imread(str(img_fp))
        img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, input_size)
        img = torch.from_numpy(img).float() / 255.
        img = img.permute(2, 0, 1)
        img = img.to('cuda')
        print('input shape: ', img.shape)

        mask = model(img.unsqueeze(0))[0].squeeze(0)   # logits
        mask = torch.sigmoid(mask)
        mask = (mask > threshold).int().cpu().numpy()
        print('mask shape: ', mask.shape)

        # mask image with mask
        print(np.unique(mask))
        mask = cv2.resize(mask, orig_img.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
        mask = np.stack([mask] * 3, axis=-1)
        masked_img = orig_img * mask

        out_fp = os.path.join(img_fp.parent, img_fp.stem + '_masked.png').replace('classification_data', 'classification_data_masked_unet')
        os.makedirs(os.path.dirname(out_fp), exist_ok=True)

        cv2.imwrite(out_fp, masked_img)
        print(f'Done save mask at {out_fp}')

        cnt += 1
        if cnt == 1e9:
            break