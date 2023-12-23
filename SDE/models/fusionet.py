from __future__ import print_function, absolute_import, division, print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
from models.submodule import *
import math
from .monocular_way import UNet, UNet_P, UNet_KD
from .fusion import fcn, ResNet_Fusion
from .stereonet import StereoNet
from .psmnet import PSMNet
from .gwcnet import GwcNet
from. gcnet import GCNet


class feature_extraction(nn.Module):
    def __init__(self, concat_feature=False, concat_feature_channel=12):
        super(feature_extraction, self).__init__()
        self.concat_feature = concat_feature

        self.inplanes = 32
        self.firstconv = nn.Sequential(convbn(1, 32, 3, 2, 1, 1),
                                       Mish(),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       Mish(),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       Mish())

        # self.layer1 = self._make_layer(BasicBlock, 32, 3, 1, 1, 1)
        self.layer2 = self._make_layer(BasicBlock, 64, 1, 1, 1, 1)
        self.layer3 = self._make_layer(BasicBlock, 128, 1, 2, 1, 1)
        self.layer4 = self._make_layer(BasicBlock, 192, 1, 2, 1, 1)
        self.layer5 = self._make_layer(BasicBlock, 256, 1, 2, 1, 1)
        self.layer6 = self._make_layer(BasicBlock, 512, 1, 2, 1, 1)
        self.pyramid_pooling = pyramidPooling(512, None, fusion_mode='sum', model_name='icnet')
        self.upconv6 = nn.Sequential(nn.Upsample(scale_factor=2),
                                     convbn(512, 256, 3, 1, 1, 1),
                                     Mish())
        self.iconv5 = nn.Sequential(convbn(512, 256, 3, 1, 1, 1),
                                    Mish())
        self.upconv5 = nn.Sequential(nn.Upsample(scale_factor=2),
                                     convbn(256, 192, 3, 1, 1, 1),
                                     Mish())
        self.iconv4 = nn.Sequential(convbn(384, 192, 3, 1, 1, 1),
                                    Mish())
        self.upconv4 = nn.Sequential(nn.Upsample(scale_factor=2),
                                     convbn(192, 128, 3, 1, 1, 1),
                                     Mish())
        self.iconv3 = nn.Sequential(convbn(256, 128, 3, 1, 1, 1),
                                    Mish())
        self.upconv3 = nn.Sequential(nn.Upsample(scale_factor=2),
                                     convbn(128, 64, 3, 1, 1, 1),
                                     Mish())
        self.iconv2 = nn.Sequential(convbn(128, 64, 3, 1, 1, 1),
                                    Mish())
        # self.upconv2 = nn.Sequential(nn.Upsample(scale_factor=2),
        #                              convbn(64, 32, 3, 1, 1, 1),
        #                              nn.ReLU(inplace=True))

        # self.gw1 = nn.Sequential(convbn(32, 40, 3, 1, 1, 1),
        #                          nn.ReLU(inplace=True),
        #                          nn.Conv2d(40, 40, kernel_size=1, padding=0, stride=1,
        #                                    bias=False))

        self.gw2 = nn.Sequential(convbn(64, 80, 3, 1, 1, 1),
                                          Mish(),
                                          nn.Conv2d(80, 80, kernel_size=1, padding=0, stride=1,
                                                    bias=False))

        self.gw3 = nn.Sequential(convbn(128, 160, 3, 1, 1, 1),
                                 Mish(),
                                 nn.Conv2d(160, 160, kernel_size=1, padding=0, stride=1,
                                           bias=False))

        self.gw4 = nn.Sequential(convbn(192, 160, 3, 1, 1, 1),
                                 Mish(),
                                 nn.Conv2d(160, 160, kernel_size=1, padding=0, stride=1,
                                           bias=False))

        self.gw5 = nn.Sequential(convbn(256, 320, 3, 1, 1, 1),
                                 Mish(),
                                 nn.Conv2d(320, 320, kernel_size=1, padding=0, stride=1,
                                           bias=False))

        self.gw6 = nn.Sequential(convbn(512, 320, 3, 1, 1, 1),
                                 Mish(),
                                 nn.Conv2d(320, 320, kernel_size=1, padding=0, stride=1,
                                           bias=False))

        if self.concat_feature:
            # self.concat1 = nn.Sequential(convbn(32, 16, 3, 1, 1, 1),
            #                              nn.ReLU(inplace=True),
            #                              nn.Conv2d(16, concat_feature_channel // 4, kernel_size=1, padding=0, stride=1,
            #                                        bias=False))

            self.concat2 = nn.Sequential(convbn(64, 32, 3, 1, 1, 1),
                                          Mish(),
                                          nn.Conv2d(32, concat_feature_channel // 2, kernel_size=1, padding=0, stride=1,
                                                    bias=False))
            self.concat3 = nn.Sequential(convbn(128, 128, 3, 1, 1, 1),
                                          Mish(),
                                          nn.Conv2d(128, concat_feature_channel, kernel_size=1, padding=0, stride=1,
                                                    bias=False))

            self.concat4 = nn.Sequential(convbn(192, 128, 3, 1, 1, 1),
                                          Mish(),
                                          nn.Conv2d(128, concat_feature_channel, kernel_size=1, padding=0, stride=1,
                                                    bias=False))

            self.concat5 = nn.Sequential(convbn(256, 128, 3, 1, 1, 1),
                                         Mish(),
                                         nn.Conv2d(128, concat_feature_channel, kernel_size=1, padding=0, stride=1,
                                                   bias=False))

            self.concat6 = nn.Sequential(convbn(512, 128, 3, 1, 1, 1),
                                         Mish(),
                                         nn.Conv2d(128, concat_feature_channel, kernel_size=1, padding=0, stride=1,
                                                   bias=False))
        



    def _make_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion), )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, pad, dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None, pad, dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.firstconv(x)
        #x = self.layer1(x)
        l2 = self.layer2(x)     #1/2
        l3 = self.layer3(l2)    #1/4
        l4 = self.layer4(l3)    #1/8
        l5 = self.layer5(l4)    #1/16
        l6 = self.layer6(l5)    #1/32
        l6 = self.pyramid_pooling(l6) 

        concat5 = torch.cat((l5, self.upconv6(l6)), dim=1)
        decov_5 = self.iconv5(concat5)
        concat4 = torch.cat((l4, self.upconv5(decov_5)), dim=1)
        # concat4 = torch.cat((l4, self.upconv5(l5)), dim=1)
        decov_4 = self.iconv4(concat4)
        concat3 = torch.cat((l3, self.upconv4(decov_4)), dim=1)
        decov_3 = self.iconv3(concat3)
        concat2 = torch.cat((l2, self.upconv3(decov_3)), dim=1)
        decov_2 = self.iconv2(concat2)
        feature_last = decov_2
        # decov_1 = self.upconv2(decov_2)


        # gw1 = self.gw1(decov_1)
        gw2 = self.gw2(decov_2)
        gw3 = self.gw3(decov_3)
        gw4 = self.gw4(decov_4)
        gw5 = self.gw5(decov_5)
        gw6 = self.gw6(l6)

        if not self.concat_feature:
            return {"gw2": gw2, "gw3": gw3, "gw4": gw4}
        else:
            # concat_feature1 = self.concat1(decov_1)
            concat_feature2 = self.concat2(decov_2)
            concat_feature3 = self.concat3(decov_3)
            concat_feature4 = self.concat4(decov_4)
            concat_feature5 = self.concat5(decov_5)
            concat_feature6 = self.concat6(l6)
            return {"gw2": gw2, "gw3": gw3, "gw4": gw4, "gw5": gw5, "gw6": gw6,
                    "concat_feature2": concat_feature2, "concat_feature3": concat_feature3, "concat_feature4": concat_feature4,
                    "concat_feature5": concat_feature5, "concat_feature6": concat_feature6, "feature_last": feature_last}



class fusionet(nn.Module):  # stereonet and unet 
    def __init__(self, maxdisp, batch_size):
        super(fusionet, self).__init__()
        self.maxdisp = maxdisp
        self.concat_channels = 12
        #self.batch_size = batch_size
        self.feature_extraction = feature_extraction(concat_feature=True, concat_feature_channel=12)
        
        
        self.norm = Generator()
        self.norm.load_state_dict(torch.load("/home/lijianing/snn/spikling-master/checkpoint/spikling-0100.pth"))
        
        self.mono_up1 = nn.Sequential(convbn(64,32,3,1,1,1),
        Mish(),
        convbn(32,32,3,1,1,1),
        Mish())
        self.mono_up2 = nn.Sequential(convbn(32,32,3,1,1,1),
        Mish())
        
        self.fe_norm = nn.Sequential(convbn(64,32,3,1,1,1),
        Mish(),
        convbn(32,32,3,1,1,1),
        Mish(),
        )
        
        self.mono_way = UNet_KD(n_channels = 32, n_classes = 1)
        
        self.stereo_way = StereoNet(batch_size = batch_size, cost_volume_method = "subtract", maxdisp = self.maxdisp)
        
    def forward(self, left, right):
        
        left = self.norm(left)
        right = self.norm(right)
        
        left_feature = self.feature_extraction(left)
        right_feature = self.feature_extraction(right)
        
        left_stereo_features = self.fe_norm(left_feature["feature_last"])
        right_stereo_features = self.fe_norm(right_feature["feature_last"]) 
        #print(left_stereo_features.size())
        #stereo_results = self.stereo_way.forward(left_feature["feature_last"], right_feature["feature_last"])
        stereo_results = self.stereo_way.forward(left_stereo_features, right_stereo_features)
        
        mono_depth = self.mono_up1(left_feature["feature_last"])  
        mono_depth = self.mono_up2(mono_depth)  
        mono_results = self.mono_way(mono_depth)
        
        result = {}
        result["monocular"] = mono_results
        result["stereo"] = stereo_results
        #result["fusion"] = 
        
        return result



class fusionet1(nn.Module):  # psmnet and unet
    def __init__(self, maxdisp):
        super(fusionet1, self).__init__()
        self.maxdisp = maxdisp
        self.concat_channels = 12
        #self.batch_size = batch_size
        self.feature_extraction = feature_extraction(concat_feature=True, concat_feature_channel=12)
        
        
        self.norm = Generator()
        self.norm.load_state_dict(torch.load("/home/lijianing/snn/spikling-master/checkpoint/spikling-0100.pth"))
        
        self.mono_up1 = nn.Sequential(convbn(64,32,3,1,1,1),
        Mish(),
        convbn(32,32,3,1,1,1),
        Mish())
        self.mono_up2 = nn.Sequential(convbn(32,32,3,1,1,1),
        Mish())
        
        self.fe_norm = nn.Sequential(convbn(64,32,3,1,1,1),
        Mish(),
        convbn(32,32,3,1,1,1),
        Mish(),
        )
         
        self.mono_way = UNet_KD(n_channels = 32, n_classes = 1)
        
        #self.stereo_way = StereoNet(batch_size = batch_size, cost_volume_method = "subtract", maxdisp = self.maxdisp)
        self.stereo_way = PSMNet(max_disp = self.maxdisp)
        
    def forward(self, left, right):
        
        left = self.norm(left)
        right = self.norm(right)
        
        left_feature = self.feature_extraction(left)
        right_feature = self.feature_extraction(right)
        
        left_stereo_features = self.fe_norm(left_feature["feature_last"])
        right_stereo_features = self.fe_norm(right_feature["feature_last"]) 
        #print(left_stereo_features.size())
        #stereo_results = self.stereo_way.forward(left_feature["feature_last"], right_feature["feature_last"])
        stereo_results = self.stereo_way.forward(left_stereo_features, right_stereo_features)
        
        mono_depth = self.mono_up1(left_feature["feature_last"])  
        mono_depth = self.mono_up2(mono_depth)  
        mono_results = self.mono_way(mono_depth)
        
        result = {}
        result["monocular"] = mono_results
        result["stereo"] = stereo_results
        #result["fusion"] = 
        
        return result

class fusionet2(nn.Module):  # ganet and unet
    def __init__(self, maxdisp):
        super(fusionet1, self).__init__()
        self.maxdisp = maxdisp
        self.concat_channels = 12
        #self.batch_size = batch_size
        self.feature_extraction = feature_extraction(concat_feature=True, concat_feature_channel=12)
        
        
        self.norm = Generator()
        self.norm.load_state_dict(torch.load("/home/lijianing/snn/spikling-master/checkpoint/spikling-0100.pth"))
        
        self.mono_up1 = nn.Sequential(convbn(64,32,3,1,1,1),
        Mish(),
        convbn(32,32,3,1,1,1),
        Mish())
        self.mono_up2 = nn.Sequential(convbn(32,32,3,1,1,1),
        Mish())
        
        self.fe_norm = nn.Sequential(convbn(64,32,3,1,1,1),
        Mish(),
        convbn(32,32,3,1,1,1),
        Mish(),
        )
        
        self.mono_way = UNet_KD(n_channels = 32, n_classes = 1)
        
        #self.stereo_way = StereoNet(batch_size = batch_size, cost_volume_method = "subtract", maxdisp = self.maxdisp)
        self.stereo_way = PSMNet(max_disp = self.maxdisp)
        
    def forward(self, left, right):
        
        left = self.norm(left)
        right = self.norm(right)
        
        left_feature = self.feature_extraction(left)
        right_feature = self.feature_extraction(right)
        
        left_stereo_features = self.fe_norm(left_feature["feature_last"])
        right_stereo_features = self.fe_norm(right_feature["feature_last"]) 
        #print(left_stereo_features.size())
        #stereo_results = self.stereo_way.forward(left_feature["feature_last"], right_feature["feature_last"])
        stereo_results = self.stereo_way.forward(left_stereo_features, right_stereo_features)
        
        mono_depth = self.mono_up1(left_feature["feature_last"])  
        mono_depth = self.mono_up2(mono_depth)  
        mono_results = self.mono_way(mono_depth)
        
        result = {}
        result["monocular"] = mono_results
        result["stereo"] = stereo_results
        #result["fusion"] = 
        
        return result


class fusionet3(nn.Module):  # gwcnet and unet
    def __init__(self, maxdisp):
        super(fusionet3, self).__init__()
        self.maxdisp = maxdisp
        self.concat_channels = 12
        #self.batch_size = batch_size
        self.feature_extraction = feature_extraction(concat_feature=True, concat_feature_channel=12)
        
        
        self.norm = Generator()
        self.norm.load_state_dict(torch.load("/home/lijianing/snn/spikling-master/checkpoint/spikling-0100.pth"))
        
        self.mono_up1 = nn.Sequential(convbn(64,32,3,1,1,1),
        Mish(),
        convbn(32,32,3,1,1,1),
        Mish())
        self.mono_up2 = nn.Sequential(convbn(32,32,3,1,1,1),
        Mish())
        
        self.fe_norm = nn.Sequential(convbn(64,32,3,1,1,1),
        Mish(),
        convbn(32,32,3,1,1,1),
        Mish(),
        )
         
        self.mono_way = UNet_KD(n_channels = 32, n_classes = 1)
        
        #self.stereo_way = StereoNet(batch_size = batch_size, cost_volume_method = "subtract", maxdisp = self.maxdisp)
        self.stereo_way = GwcNet(maxdisp = 32, use_concat_volume=True)
        
    def forward(self, left, right):
        
        left = self.norm(left)
        right = self.norm(right)
        
        left_feature = self.feature_extraction(left)
        right_feature = self.feature_extraction(right)
        
        left_stereo_features = self.fe_norm(left_feature["feature_last"])
        right_stereo_features = self.fe_norm(right_feature["feature_last"]) 
        #print(left_stereo_features.size())
        #stereo_results = self.stereo_way.forward(left_feature["feature_last"], right_feature["feature_last"])
        stereo_results = self.stereo_way.forward(left_feature, right_feature)
        #stereo_results = self.stereo_way.forward(left, right)
        
        mono_depth = self.mono_up1(left_feature["feature_last"])  
        mono_depth = self.mono_up2(mono_depth)  
        mono_results = self.mono_way(mono_depth)
        
        result = {}
        result["monocular"] = mono_results
        result["stereo"] = stereo_results
        #result["fusion"] = 
        
        return result


class fusionet4(nn.Module):  # gwcnet and unet
    def __init__(self, maxdisp, batch_size):
        super(fusionet4, self).__init__()
        self.maxdisp = maxdisp
        self.concat_channels = 12
        #self.batch_size = batch_size
        self.feature_extraction = feature_extraction(concat_feature=True, concat_feature_channel=12)
        
        
        self.norm = Generator()
        self.norm.load_state_dict(torch.load("/home/lijianing/snn/spikling-master/checkpoint/spikling-0100.pth"))
        
        self.mono_up1 = nn.Sequential(convbn(64,32,3,1,1,1),
        Mish(),
        convbn(32,32,3,1,1,1),
        Mish())
        self.mono_up2 = nn.Sequential(convbn(32,32,3,1,1,1),
        Mish())
        
        self.fe_norm = nn.Sequential(convbn(64,32,3,1,1,1),
        Mish(),
        convbn(32,32,3,1,1,1),
        Mish(),
        )
        
        self.mono_way = UNet_KD(n_channels = 32, n_classes = 1)
        
        #self.stereo_way = StereoNet(batch_size = batch_size, cost_volume_method = "subtract", maxdisp = self.maxdisp)
        self.stereo_way = GCNet(batch_size = batch_size, cost_volume_method = "subtract", maxdisp = self.maxdisp)
        
    def forward(self, left, right):
        
        left = self.norm(left)
        right = self.norm(right)
        
        left_feature = self.feature_extraction(left)
        right_feature = self.feature_extraction(right)
        
        left_stereo_features = self.fe_norm(left_feature["feature_last"])
        right_stereo_features = self.fe_norm(right_feature["feature_last"]) 
        #print(left_stereo_features.size())
        #stereo_results = self.stereo_way.forward(left_feature["feature_last"], right_feature["feature_last"])
        stereo_results = self.stereo_way.forward(left_stereo_features, right_stereo_features)
        
        mono_depth = self.mono_up1(left_feature["feature_last"])  
        mono_depth = self.mono_up2(mono_depth)  
        mono_results = self.mono_way(mono_depth)
        
        result = {}
        result["monocular"] = mono_results
        result["stereo"] = stereo_results
        #result["fusion"] = 
        
        return result


class BasicModel(nn.Module):
    '''
    Basic model class that can be saved and loaded
        with specified names.
    '''

    def __init__(self):
        super().__init__()
        self.device = torch.device('cpu')

    def save(self, path: str):
        torch.save(self.state_dict(), path)
        print('save model to \"{}\"'.format(path))

    def load(self, path: str):
        if os.path.exists(path):
            state = torch.load(path, map_location=self.device)
            self.load_state_dict(state)
            print('load pre-trained model \"{}\"'.format(path))
        else:
            print('init model')
        return self
    
    def to(self, device: torch.device):
        self.device = device
        super().to(device)
        return self




        

class Generator(BasicModel):
    '''
    Input a (`batch`, `window`, `height`, `width`) sample,
        outputs a (`batch`, `1`, `height`, `width`) result.
    '''

    def __init__(self):
        super().__init__()
        self.down1 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(64),
            nn.Conv2d(64, 64, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.bottom = nn.Sequential(
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.up1 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.up2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, 1, 1, bias=False),
            nn.Tanh(),
        )
        self.flat = nn.Conv2d(32, 1, 1, bias=False)

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.bottom(d2)
        d2 = self.up1(d2 + d3)
        d1 = self.up2(d1 + d2)
        x = self.flat(d1)
        return x


