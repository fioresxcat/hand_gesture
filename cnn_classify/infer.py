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

from model import CNNClassifier

if __name__ == '__main__':
    from sys import argv
    import yaml

    data_dir = '/data3/users/tungtx2/hand_gesture/classification_data'
    ckpt_path = 'ckpt/exp1_no_mask/epoch=45-train_loss=0.001-val_loss=2.263-train_acc=1.000-val_acc=0.729.ckpt'

    config_path = Path(ckpt_path).parent / 'config.yaml'
    with open(config_path, 'r') as f:
        config = EasyDict(yaml.load(f, Loader=yaml.FullLoader))
    input_size = config.data.data_cfg.input_size

    pl_module = CNNClassifier.load_from_checkpoint(
        ckpt_path, 
        n_classes=config.model.n_classes, 
        learning_rate=config.model.learning_rate,
        reset_optimizer=False,
        acc=None,
        criterion=None,
    )
    model = pl_module.model
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
        img = img.unsqueeze(0).to('cuda')
        print('input shape: ', img.shape)

        with torch.no_grad():
            feats = model.features(img)
            feats = model.avgpool(feats)
            feats = feats.squeeze().detach().cpu().numpy()
        # pdb.set_trace()
        
        save_fp = str(img_fp).replace('/classification_data/', '/classification_data_features/').replace('.jpg', '.npy')
        os.makedirs(os.path.dirname(save_fp), exist_ok=True)
        np.save(save_fp, feats)
        print(f'Saved to {save_fp}')

        cnt += 1
        if cnt == 1e9:
            break