import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

import torchvision
import segmentation_models_pytorch as smp
from unet_utils import *
import pytorch_lightning as pl
from copy import deepcopy
import torch
import torchmetrics
    


class EffUnet(nn.Module):
    def __init__(self, general_cfg, model_cfg):
        super(EffUnet, self).__init__(general_cfg, model_cfg)
        self.config = model_cfg
        self._init_layers()
    

    def _init_layers(self):
        encoder = torchvision.models.efficientnet_b0()

        self.conv_e1 = ConvBlock(self.config.in_c, 32, stride=2, act=nn.SiLU())    # /2, 32
        self.conv_e2 = encoder.features[1:3]     # /4, 24
        self.conv_e3 = encoder.features[3]       # /8, 40
        self.conv_e4 = encoder.features[4]       # /16, 80,    cua nos chac la [4:5] de co 112 channel
        self.conv_e5 = encoder.features[5:7]      # /32, 192
        self.conv_connect = nn.Sequential(
            ConvBlock(in_c=192, out_c=192, act=nn.SiLU()),
            ConvBlock(in_c=192, out_c=192, act=nn.SiLU()),
        )

        self.conv_d1 = nn.Sequential(
            ConvBlock(in_c=272, out_c=80, act=nn.SiLU()),
            ConvBlock(in_c=80, out_c=40, act=nn.SiLU())
        )
        self.conv_d2 = nn.Sequential(
            ConvBlock(in_c=80, out_c=80, act=nn.SiLU()),
            ConvBlock(in_c=80, out_c=24, act=nn.SiLU())
        )
        self.conv_d3 = nn.Sequential(
            ConvBlock(in_c=48, out_c=48, act=nn.SiLU()),
            ConvBlock(in_c=48, out_c=32, act=nn.SiLU())
        )
        self.conv_d4 = nn.Sequential(
            ConvBlock(in_c=64, out_c=64, act=nn.SiLU()),
            ConvBlock(in_c=64, out_c=32, act=nn.SiLU())
        )
        # self.head = nn.Sequential(
        #     nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1, stride=1),
        #     nn.SiLU(),
        #     nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, padding=1, stride=1),
        #     nn.Sigmoid()
        # )

        c=32
        self.head = nn.Sequential(
            # ConvBlock(in_c=c, out_c=c*4, act=nn.SiLU()),
            # ConvBlock(in_c=c*4, out_c=c*4, act=nn.SiLU()),
            # ConvBlock(in_c=c*4, out_c=c, act=nn.SiLU()),

            # nn.Conv2d(c, c*4, kernel_size=3, stride=1, padding=1),
            # nn.SiLU(),
            # nn.Conv2d(c*4, c*4, kernel_size=3, stride=1, padding=1),
            # nn.SiLU(),
            nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),

            nn.Conv2d(c, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )
        

    def forward(self, input):
        e1 = self.conv_e1(input)     # /2, 32
        e2 = self.conv_e2(e1)     # /4, 24
        e3 = self.conv_e3(e2)       # /8, 40
        e4 = self.conv_e4(e3)       # /16, 80
        e5 = self.conv_e5(e4)      # /32, 192
        e5 = self.conv_connect(e5)   # /32, 192

        d1 = torch.concat([e4, F.interpolate(e5, scale_factor=2, mode='bilinear')], dim=1)
        d1 = self.conv_d1(d1)

        d2 = torch.concat([e3, F.interpolate(d1, scale_factor=2, mode='bilinear')], dim=1)
        d2 = self.conv_d2(d2)

        d3 = torch.concat([e2, F.interpolate(d2, scale_factor=2, mode='bilinear')], dim=1)
        d3 = self.conv_d3(d3)

        d4 = torch.concat([e1, F.interpolate(d3, scale_factor=2, mode='bilinear')], dim=1)
        d4 = self.conv_d4(d4)

        d5 = F.interpolate(d4, scale_factor=2, mode='bilinear')
        out = self.head(d5)

        return out


class EffSmpUnet(nn.Module):
    def __init__(self, general_cfg, model_cfg):
        super(EffSmpUnet, self).__init__(general_cfg, model_cfg)
        self.config = model_cfg
        self._init_layers()
    

    def _init_layers(self):
        encoder = torchvision.models.efficientnet_b0()

        self.conv_e1 = ConvBlock(self.config.in_c, 32, stride=2, act=nn.SiLU())    # /2, 32
        self.conv_e2 = encoder.features[1:3]     # /4, 24
        self.conv_e3 = encoder.features[3]       # /8, 40
        self.conv_e4 = encoder.features[4:6]       # /16, 80,    cua nos chac la [4:5] de co 112 channel
        self.conv_e5 = encoder.features[6:8]      # /32, 192
        
        self.decoder = UnetPlusPlusDecoder(
            encoder_channels = (15, 32, 24, 40, 112, 320),
            decoder_channels=(256, 128, 64, 32, 16),
            n_blocks=5,
            use_batchnorm=True,
            center=False,
            attention_type='scse',
        )

        c = 16
        self.hm_out = nn.Sequential(
            # ConvBlock(in_c=c, out_c=c*4, act=nn.SiLU()),
            # ConvBlock(in_c=c*4, out_c=c*4, act=nn.SiLU()),
            # ConvBlock(in_c=c*4, out_c=c, act=nn.SiLU()),

            # nn.Conv2d(c, c*4, kernel_size=3, stride=1, padding=1),
            # nn.SiLU(),
            # nn.Conv2d(c*4, c*4, kernel_size=3, stride=1, padding=1),
            # nn.SiLU(),
            nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),

            nn.Conv2d(c, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )
        

    def forward(self, input):
        e0 = input
        e1 = self.conv_e1(input)     # /2, 32
        e2 = self.conv_e2(e1)     # /4, 24
        e3 = self.conv_e3(e2)       # /8, 40
        e4 = self.conv_e4(e3)       # /16, 80
        e5 = self.conv_e5(e4)      # /32, 192

        decoder_output = self.decoder(e0, e1, e2, e3, e4, e5)
        out = self.hm_out(decoder_output)

        return out
    

    def forward_features(self, input):
        e0 = input
        e1 = self.conv_e1(input)     # /2, 32
        e2 = self.conv_e2(e1)     # /4, 24
        e3 = self.conv_e3(e2)       # /8, 40
        e4 = self.conv_e4(e3)       # /16, 80
        e5 = self.conv_e5(e4)      # /32, 192

        decoder_output = self.decoder(e0, e1, e2, e3, e4, e5)
        return decoder_output


class SmpUnetModified(nn.Module):
    def __init__(self, general_cfg, model_cfg):
        super(SmpUnetModified, self).__init__(general_cfg, model_cfg)
        self.config = model_cfg
        self._init_layers()
    

    def _init_layers(self):
        base_model = smp.UnetPlusPlus(
            encoder_name=self.config.backbone, 
            encoder_depth=self.config.encoder_depth,
            encoder_weights="imagenet",
            in_channels=self.config.in_c,                  
            classes=1,
        )

        self.encoder = base_model.encoder
        self.decoder = base_model.decoder

        c = self.decoder.blocks.x_0_4.conv2[0].out_channels

        self.hm_out = nn.Sequential(
            ConvBlock(in_c=c, out_c=c*4),
            ConvBlock(in_c=c*4, out_c=c*4),
            ConvBlock(in_c=c*4, out_c=c),
            nn.Conv2d(c, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )
        

    def forward(self, input):
        features = self.encoder(input)
        decoder_output = self.decoder(*features)
        hm = self.hm_out(decoder_output)
        return hm

    
class SmpDeepLab(nn.Module):
    def __init__(self, general_cfg, model_cfg):
        super(SmpDeepLab, self).__init__(general_cfg, model_cfg)
        self.config = model_cfg
        self._init_layers()
    

    def _init_layers(self):
        self.model = smp.DeepLabV3Plus(
            encoder_name=self.config.backbone, 
            encoder_depth=self.config.encoder_depth,
            encoder_weights="imagenet",
            in_channels=self.config.in_c,                  
            classes=1,
        )
        

    def forward(self, input):
        out = self.model(input)
        out = torch.sigmoid(out)
        return out
    

class SmpUnet(nn.Module):
    def __init__(self, backbone, encoder_depth, in_c, classes):
        super().__init__()
        self.model = smp.Unet(
            encoder_name=self.config.backbone, 
            encoder_depth=self.config.encoder_depth,
            encoder_weights="imagenet",
            in_channels=self.config.in_c,                  
            classes=1,
        )

    def forward(self, input):
        out = self.model(input)
        return out
    


class UnetModule(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float,
        criterion: nn.Module,
        iou: torchmetrics.classification.JaccardIndex,
        f1: torchmetrics.classification.F1Score,
    ):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.criterion = criterion

        self.train_iou = iou
        self.val_iou = deepcopy(iou)
        self.test_iou = deepcopy(iou)

        self.train_f1 = f1
        self.val_f1 = deepcopy(f1)
        self.test_f1 = deepcopy(f1)
        

    
    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode=self.trainer.callbacks[0].mode,
            factor=0.2,
            patience=8,
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


    def common_step(self, batch, batch_idx, split):
        imgs, gt_masks = batch
        pred_masks = self.model(imgs)
        loss = self.criterion(pred_masks, gt_masks)

        iou = getattr(self, f'{split}_iou')
        iou(pred_masks, gt_masks)
        
        f1 = getattr(self, f'{split}_f1')
        f1(pred_masks, gt_masks)

        self.log_dict({
            f'{split}_loss': loss,
            f'{split}_iou': iou,
            f'{split}_f1': f1,
        }, prog_bar=True, on_step=True, on_epoch=True)

        return loss
    

    def training_step(self, batch, batch_idx):
        return self.common_step(batch, batch_idx, 'train')

    def validation_step(self, batch, batch_idx):
        return self.common_step(batch, batch_idx, 'val')

    def test_step(self, batch, batch_idx):
        return self.common_step(batch, batch_idx, 'test')


    def on_predict_start(self) -> None:
        self.bs = self.trainer.datamodule.training_cfg.bs








    def on_train_epoch_start(self) -> None:
        print('\n')




if __name__ == '__main__':
    import yaml
    from easydict import EasyDict
    from loss import *
    from metric import *

    with open('config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = EasyDict(config)
    
    model = smp.Unet(**config.model.model.init_args)
    criterion = DiceLoss(**config.model.criterion.init_args)
    iou = MyIOU(**config.model.iou.init_args)
    unet_module = UnetModule(model, config.model.learning_rate, criterion, iou)

    imgs = torch.randn(2, 3, 256, 320)
    out = model(imgs)
    print(out.shape)
    pdb.set_trace()