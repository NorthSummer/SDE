from __future__ import absolute_import, division, print_function

import numpy as np

import torch
import torch.nn as nn
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
from models.submodule import *

device = torch.device("cuda:{}".format(6))



class ResNetMultiImageInput(models.ResNet):
    """Constructs a resnet model with varying number of input images.
    Adapted from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    """
    def __init__(self, block, layers, num_classes=1000, num_input_images=1):
        super(ResNetMultiImageInput, self).__init__(block, layers)
        self.inplanes = 64
        self.conv1 = nn.Conv2d(
            num_input_images * 1, 64, kernel_size=7, stride=2, padding=3, bias=False)  # num_input_images * 3
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def resnet_multiimage_input(num_layers, pretrained=False, num_input_images=1):
    """Constructs a ResNet model.
    Args:
        num_layers (int): Number of resnet layers. Must be 18 or 50
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        num_input_images (int): Number of frames stacked as input
    """
    assert num_layers in [18, 50], "Can only run with 18 or 50 layer resnet"
    blocks = {18: [2, 2, 2, 2], 50: [3, 4, 6, 3]}[num_layers]
    block_type = {18: models.resnet.BasicBlock, 50: models.resnet.Bottleneck}[num_layers]
    model = ResNetMultiImageInput(block_type, blocks, num_input_images=num_input_images)
    
    if pretrained:
        loaded = model_zoo.load_url(models.resnet.model_urls['resnet{}'.format(num_layers)])
        loaded['conv1.weight'] = torch.cat(
            [loaded['conv1.weight']] * num_input_images, 1) / num_input_images
        #model.load_state_dict(loaded)#
    return model


class ResnetEncoder(nn.Module):
    """Pytorch module for a resnet encoder
    """
    def __init__(self, num_layers, pretrained, num_input_images=1):
        super(ResnetEncoder, self).__init__()

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])

        resnets = {18: models.resnet18,
                   34: models.resnet34,
                   50: models.resnet50,
                   101: models.resnet101,
                   152: models.resnet152}

        if num_layers not in resnets:
            raise ValueError("{} is not a valid number of resnet layers".format(num_layers))

        if num_input_images >= 1:
            self.encoder = resnet_multiimage_input(num_layers, pretrained, num_input_images)
        else:
            self.encoder = resnets[num_layers](pretrained)

        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

    def forward(self, input_image):
        self.features = []
        x = (input_image - 0.45) / 0.225
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        self.features.append(self.encoder.relu(x))
        self.features.append(self.encoder.layer1(self.encoder.maxpool(self.features[-1])))
        self.features.append(self.encoder.layer2(self.features[-1]))
        self.features.append(self.encoder.layer3(self.features[-1]))
        self.features.append(self.encoder.layer4(self.features[-1]))

        return self.features
        
        


import numpy as np
import torch
import torch.nn as nn

from collections import OrderedDict
from .layers import *
from .layers import Conv3x3

class DepthDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True):
        super(DepthDecoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        # decoder
        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        for s in self.scales:
            self.convs[("dispconv", s)] =  nn.Sequential(nn.ReflectionPad2d(1),
                    nn.Conv2d(self.num_ch_dec[s], self.num_output_channels, 3),  
                    nn.ELU(inplace=True))#Conv3x3(self.num_ch_dec[s], self.num_output_channels)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features):
        self.outputs = {}

        # decoder
        x = input_features[-1]
        for i in range(4, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            x = [upsample(x)]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            x = self.convs[("upconv", i, 1)](x)
            if i in self.scales:
                self.outputs[("disp", i)] = self.sigmoid(self.convs[("dispconv", i)](x))

        return self.outputs        
        
        
class MonocularWay(nn.Module):
    def __init__(self):
        super(MonocularWay, self).__init__()
        self.batch_size = 4
        self.encoder = ResnetEncoder(num_layers = 50, pretrained="scratch")
        
        self.decoder = DepthDecoder(self.encoder.num_ch_enc, scales = [0, 1, 2, 3])        
        
    def forward(self, x):
        enc_features = self.encoder(x)
        
        #all_features = [torch.split(f, self.batch_size) for f in enc_features]  #
     
        outputs = self.decoder(enc_features)
        
        return outputs
        
        
        
##################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            #nn.Dropout(0.2),
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)        
   
'''   
                
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512//2)#512
        factor = 2 if bilinear else 1
        #self.down4 = Down(512, 1024 // factor)
        #self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        #x5 = self.down4(x4)
        #x = self.up1(x5, x4)
        
        x = self.up2(x4, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits        


'''
class UNet_P(nn.Module):
    def __init__(self):
        pass
   
                
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)#512
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits#, y1, y2, y3, y4       

                
class UNet_KD(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet_KD, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)#512
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)

        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        
        x1 = self.inc(x)
        x1 = F.upsample(x1, [128,256], mode='bilinear', 
                                 align_corners=True)#[("disp"), 0]
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        y4 = self.up1(x5, x4)
        
        y3 = self.up2(y4, x3)
        y2 = self.up3(y3, x2)
        y1 = self.up4(y2, x1)
        logits = self.outc(y1)
        logits = F.upsample(logits, [256,512], mode='bilinear', align_corners=True).squeeze(1) 
        
        result = {}
        result["monocular"] = logits, x2, x3, x4, x5  
        
        return logits#result#logits     

class UNet_KD1(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet_KD1, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)#512
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
       
        self.middle= nn.Sequential(nn.Conv2d(512,1024,3,1,1),
        nn.BatchNorm2d(1024),
        nn.ReLU(inplace = True),
        nn.Conv2d(1024,1024,3,1,1),
        nn.BatchNorm2d(1024),
        nn.ReLU(inplace = True),
        nn.Conv2d(1024,512,3,1,1),
        nn.BatchNorm2d(512),
        nn.Dropout(0.1), 
        nn.ReLU(inplace = True),
        )
         
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        
        self.outc = OutConv(64, n_classes)
        
                  
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1) 
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        b = self.middle(x5)
        
        y4 = self.up1(b, x4)
        
        y3 = self.up2(y4, x3)
        y2 = self.up3(y3, x2)
        y1 = self.up4(y2, x1)
        
        logits = self.outc(y1)   
        
        logits = F.upsample(logits, [256,512], mode='bilinear', align_corners=True).squeeze(1) 
        
        return logits    



class UNet_KD2(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet_KD2, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)#512
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        
        self.middle= nn.Sequential(nn.Conv2d(512,1024,3,1,1),
        nn.BatchNorm2d(1024),
        nn.ReLU(inplace = True),
        nn.Conv2d(1024,1024,3,1,1),
        nn.BatchNorm2d(1024),
        nn.ReLU(inplace = True),
        nn.Conv2d(1024,512,3,1,1),
        nn.BatchNorm2d(512),
        nn.ReLU(inplace = True),
        )
        
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        
        self.outc = OutConv(64, n_classes)        
  
               
                  
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1) 
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        b = self.middle(x5)
        
        y4 = self.up1(b, x4)        
        y3 = self.up2(y4, x3)
        y2 = self.up3(y3, x2)
        y1 = self.up4(y2, x1)
        
        logits = self.outc(y1)           
        logits = F.upsample(logits, [256,512], mode='bilinear', align_corners=True).squeeze(1) 
    
        z4 = self.up1(b, x4)        
        z3 = self.up2(z4, x3)
        z2 = self.up3(z3, x2)
        z1 = self.up4(z2, x1)
        
        logits1 = self.outc(z1)          
        logits1 = F.upsample(logits1, [256,512], mode='bilinear', align_corners=True).squeeze(1) 
           
        return logits, logits1


class UNet_KDD(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet_KDD, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)#512
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
       
        self.middle= nn.Sequential(nn.Conv2d(512,1024,3,1,1),
        nn.BatchNorm2d(1024),
        nn.ReLU(inplace = True),
        nn.Conv2d(1024,1024,3,1,1),
        nn.BatchNorm2d(1024),
        nn.ReLU(inplace = True),
        nn.Conv2d(1024,512,3,1,1),
        nn.BatchNorm2d(512),
        nn.ReLU(inplace = True),
        )
         
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        
        self.outc = OutConv(64, n_classes)
         
     
                  
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1) 
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        b = self.middle(x5)
        
        y4 = self.up1(b, x4)
        
        y3 = self.up2(y4, x3)
        y2 = self.up3(y3, x2)
        y1 = self.up4(y2, x1)
        
        logits = self.outc(y1)   
        
        logits = F.upsample(logits, [256,512], mode='bilinear', align_corners=True).squeeze(1) 
        #print(logits.size())
        N, C, H, W = logits.size()
        ord_num = C // 2


        logits = logits.view(-1, 2, ord_num, H, W)
        if self.training:
            prob = F.log_softmax(logits, dim=1).view(N, C, H, W)
            
          
        ord_prob = F.softmax(logits, dim=1)[:, 0, :, :, :]
        ord_label = torch.sum((ord_prob > 0.5), dim=1)
        #print(666666666666)
        if self.training:
            return prob, ord_label  
        else:
            return ord_prob, ord_label   



class UNet_KD_unc(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet_KD_unc, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)#512
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)

        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        
        x1 = self.inc(x)
        x1 = F.upsample(x1, [128,256], mode='bilinear', 
                                 align_corners=True)#[("disp"), 0]
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        y4 = self.up1(x5, x4)
        
        y3 = self.up2(y4, x3)
        y2 = self.up3(y3, x2)
        y1 = self.up4(y2, x1)
        logits = self.outc(y1)
        logits = self.sigmoid(logits)
        print(logits.size())
        logits = F.upsample(logits, [256,512], mode='bilinear', align_corners=True)#.squeeze(1) 
        
        result = {}
        result["depth"] = logits[:,0,:,:].squeeze(1)
        result["uncertainty"] = logits[:,1,:,:].squeeze(1)
        #print(result["depth"].size())
        return result#result#logits     



class UNet_KD_drop(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet_KD_drop, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)#512
        factor = 2 if bilinear else 1
        
        self.down4 = Down(512, 1024 // factor)

        
        self.dropout = nn.Dropout(0.05)
        self.down4 = Down(512, 1024 // factor)
      
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        
        x1 = self.inc(x)
        x1 = F.upsample(x1, [128,256], mode='bilinear', 
                                 align_corners=True)#[("disp"), 0]
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x5 = self.dropout(x5)
        y4 = self.up1(x5, x4)
        
        
        y3 = self.up2(y4, x3)
        y2 = self.up3(y3, x2)
        y1 = self.up4(y2, x1)
        
        logits = self.outc(y1)
        logits = F.upsample(logits, [256,512], mode='bilinear', align_corners=True).squeeze(1) 

        
        return logits#result#logits     



class KD_bottle(nn.Module):
    def __init__(self):
        super(KD_bottle, self).__init__()

        self.kd5 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            Mish(),)
                    
          
        self.kd4 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            Mish(),)
            
        self.kd3 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            Mish(),)

        self.kd2 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(256),
            Mish(),)
            
        self.kd1 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(128),
            Mish(),)      
            
    def forward(self, y1, y2, y3, y4, b):
        
        z1 = self.kd1(y1)
        z2 = self.kd2(y2)
        z3 = self.kd3(y3)
        z4 = self.kd4(y4)
        zb = self.kd5(b)
        
        return z1, z2, z3, z4, zb
    

class KD_bottle_t(nn.Module):
    def __init__(self):
        super(KD_bottle, self).__init__()

          
        self.kd5 = nn.Sequential(
            Mish(),)        
          
        self.kd4 = nn.Sequential(
            Mish(),)
            
        self.kd3 = nn.Sequential(
            Mish(),)

        self.kd2 = nn.Sequential(
            Mish(),)
            
        self.kd1 = nn.Sequential(
            Mish(),)      
            
            
    def forward(self, y1, y2, y3, y4, b):
        
        z1 = self.kd1(y1)
        z2 = self.kd2(y2)
        z3 = self.kd3(y3)
        z4 = self.kd4(y4)
        zb = self.kd5(b)
        
        
        return z1, z2, z3, z4, zb
    

                
class UNet_Stu(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        #self.down3 = Down(256, 512)#512
        factor = 2 if bilinear else 1
        s#elf.down4 = Down(512, 1024 // factor)
        #self.up1 = Up(1024, 512 // factor, bilinear)
        #self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        #x4 = self.down3(x3)
        #x5 = self.down4(x4)
        #x = self.up1(x5, x4)
        
        #x = self.up2(x, x3)
        x = self.up3(x3, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits        





#######cvpr 2018

class Dorn(nn.Module):

    def __init__(self, ord_num=90, gamma=1.0, beta=80.0,
                 input_size=(385, 513), kernel_size=16, pyramid=[8, 12, 16],
                 batch_norm=False,
                 discretization="SID", pretrained=True):
        super().__init__()
        assert len(input_size) == 2
        assert isinstance(kernel_size, int)
        self.ord_num = ord_num
        self.gamma = gamma
        self.beta = beta
        self.discretization = discretization

        self.backbone = ResNetBackbone(pretrained=pretrained)
        self.SceneUnderstandingModule = SceneUnderstandingModule(ord_num, size=input_size,
                                                                 kernel_size=kernel_size,
                                                                 pyramid=pyramid,
                                                                 batch_norm=batch_norm)
        self.regression_layer = OrdinalRegressionLayer()
        self.criterion = OrdinalRegressionLoss(ord_num, beta, discretization)

    def forward(self, image, target=None):
        """
        :param image: RGB image, torch.Tensor, Nx3xHxW
        :param target: ground truth depth, torch.Tensor, NxHxW
        :return: output: if training, return loss, torch.Float,
                         else return {"target": depth, "prob": prob, "label": label},
                         depth: predicted depth, torch.Tensor, NxHxW
                         prob: probability of each label, torch.Tensor, NxCxHxW, C is number of label
                         label: predicted label, torch.Tensor, NxHxW
        """
        N, C, H, W = image.shape
        feat = self.backbone(image)
        feat = self.SceneUnderstandingModule(feat)
        # print("feat shape:", feat.shape)
        # feat = F.interpolate(feat, size=(H, W), mode="bilinear", align_corners=True)
        if self.training:
            prob = self.regression_layer(feat)
            loss = self.criterion(prob, target)
            return loss

        prob, label = self.regression_layer(feat)
        # print("prob shape:", prob.shape, " label shape:", label.shape)
        if self.discretization == "SID":
            t0 = torch.exp(np.log(self.beta) * label.float() / self.ord_num)
            t1 = torch.exp(np.log(self.beta) * (label.float() + 1) / self.ord_num)
        else:
            t0 = 1.0 + (self.beta - 1.0) * label.float() / self.ord_num
            t1 = 1.0 + (self.beta - 1.0) * (label.float() + 1) / self.ord_num
        depth = (t0 + t1) / 2 - self.gamma
        # print("depth min:", torch.min(depth), " max:", torch.max(depth),
        #       " label min:", torch.min(label), " max:", torch.max(label))
        return {"target": [depth], "prob": [prob], "label": [label]}    


#####nips- 2014- eigen

class coarseNet(nn.Module):
    def __init__(self,init_weights=True):
        super(coarseNet, self).__init__()
        self.conv1 = nn.Conv2d(32, 96, kernel_size = 11, stride = 4, padding = 0)
        self.conv2 = nn.Conv2d(96, 256, kernel_size = 5, padding = 2)
        self.conv3 = nn.Conv2d(256, 384, kernel_size = 3, padding = 1)
        self.conv4 = nn.Conv2d(384, 384, kernel_size = 3, padding = 1)
        self.conv5 = nn.Conv2d(384, 256, kernel_size = 3, stride = 2)
        self.fc1 = nn.Linear(12288, 4096)
        self.fc2 = nn.Linear(4096, 4070)
        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout2d()
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
                                                # [n, c,  H,   W ]
                                                # [8, 3, 228, 304]
        x = self.conv1(x)                       # [8, 96, 55, 74]
        x = F.relu(x)
        x = self.pool(x)                        # [8, 96, 27, 37] -- 
        x = self.conv2(x)                       # [8, 256, 23, 33]
        x = F.relu(x)
        x = self.pool(x)                        # [8, 256, 11, 16] 18X13
        x = self.conv3(x)                       # [8, 384, 9, 14]
        x = F.relu(x)
        x = self.conv4(x)                       # [8, 384, 7, 12]
        x = F.relu(x)
        x = self.conv5(x)                       # [8, 256, 5, 10] 8X5
        x = F.relu(x)
        x = x.view(x.size(0), -1)               # [8, 12800]
        x = F.relu(self.fc1(x))                 # [8, 4096]
        x = self.dropout(x)
        x = self.fc2(x)                         # [8, 4070]     => 55x74 = 4070
        x = x.view(-1, 1, 55, 74)
        return x
 
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
                
                
class fineNet(nn.Module):
    def __init__(self, init_weights=True):
        super(fineNet, self).__init__()
        self.conv1 = nn.Conv2d(32, 63, kernel_size = 9, stride = 2)
        self.conv2 = nn.Conv2d(64, 64, kernel_size = 5, padding = 2)
        self.conv3 = nn.Conv2d(64, 1, kernel_size = 5, padding = 2)
        self.pool = nn.MaxPool2d(2)
        if init_weights:
            self._initialize_weights()

    def forward(self, x, y):
                                                # [8, 3, 228, 304]
        x = F.relu(self.conv1(x))               # [8, 63, 110, 148]
        x = self.pool(x)                        # [8, 63, 55, 74]
        x = torch.cat((x,y),1)                  # x - [8, 63, 55, 74] y - [8, 1, 55, 74] => x = [8, 64, 55, 74]
        x = F.relu(self.conv2(x))               # [8, 64, 55, 74]
        x = self.conv3(x)                       # [8, 64, 55, 74]
        return x
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


### adaptive bins cvpr 2021
from .layers import PatchTransformerEncoder, PixelWiseDotProduct

class mViT(nn.Module):
    def __init__(self, in_channels, n_query_channels=128, patch_size=16, dim_out=256,
                 embedding_dim=128, num_heads=4, norm='linear'):
        super(mViT, self).__init__()
        self.norm = norm
        self.n_query_channels = n_query_channels
        self.patch_transformer = PatchTransformerEncoder(in_channels, patch_size, embedding_dim, num_heads)
        self.dot_product_layer = PixelWiseDotProduct()

        self.conv3x3 = nn.Conv2d(in_channels, embedding_dim, kernel_size=3, stride=1, padding=1)
        self.regressor = nn.Sequential(nn.Linear(embedding_dim, 256),
                                       nn.LeakyReLU(),
                                       nn.Linear(256, 256),
                                       nn.LeakyReLU(),
                                       nn.Linear(256, dim_out))

    def forward(self, x):
        # n, c, h, w = x.size()
        tgt = self.patch_transformer(x.clone())  # .shape = S, N, E

        x = self.conv3x3(x)

        regression_head, queries = tgt[0, ...], tgt[1:self.n_query_channels + 1, ...]

        # Change from S, N, E to N, S, E
        queries = queries.permute(1, 0, 2)
        range_attention_maps = self.dot_product_layer(x, queries)  # .shape = n, n_query_channels, h, w

        y = self.regressor(regression_head)  # .shape = N, dim_out
        if self.norm == 'linear':
            y = torch.relu(y)
            eps = 0.1
            y = y + eps
        elif self.norm == 'softmax':
            return torch.softmax(y, dim=1), range_attention_maps
        else:
            y = torch.sigmoid(y)
        y = y / y.sum(dim=1, keepdim=True)
        return y, range_attention_maps


class UnetAdaptiveBins(nn.Module):
    def __init__(self, backend, n_bins=100, min_val=0.1, max_val=10, norm='linear'):
        super(UnetAdaptiveBins, self).__init__()
        self.num_classes = n_bins
        self.min_val = min_val
        self.max_val = max_val
        self.encoder = Encoder(backend)
        self.adaptive_bins_layer = mViT(128, n_query_channels=128, patch_size=16,
                                        dim_out=n_bins,
                                        embedding_dim=128, norm=norm)

        self.decoder = DecoderBN(num_classes=128)
        self.conv_out = nn.Sequential(nn.Conv2d(128, n_bins, kernel_size=1, stride=1, padding=0),
                                      nn.Softmax(dim=1))

    def forward(self, x, **kwargs):
        unet_out = self.decoder(self.encoder(x), **kwargs)
        bin_widths_normed, range_attention_maps = self.adaptive_bins_layer(unet_out)
        out = self.conv_out(range_attention_maps)

        # Post process
        # n, c, h, w = out.shape
        # hist = torch.sum(out.view(n, c, h * w), dim=2) / (h * w)  # not used for training

        bin_widths = (self.max_val - self.min_val) * bin_widths_normed  # .shape = N, dim_out
        bin_widths = nn.functional.pad(bin_widths, (1, 0), mode='constant', value=self.min_val)
        bin_edges = torch.cumsum(bin_widths, dim=1)

        centers = 0.5 * (bin_edges[:, :-1] + bin_edges[:, 1:])
        n, dout = centers.size()
        centers = centers.view(n, dout, 1, 1)

        pred = torch.sum(out * centers, dim=1, keepdim=True)

        return bin_edges, pred

    def get_1x_lr_params(self):  # lr/10 learning rate
        return self.encoder.parameters()

    def get_10x_lr_params(self):  # lr learning rate
        modules = [self.decoder, self.adaptive_bins_layer, self.conv_out]
        for m in modules:
            yield from m.parameters()
    
















    
    