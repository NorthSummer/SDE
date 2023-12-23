import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim

import torch.utils.data
from torch.autograd import Variable
import torchvision.utils as vutils
import torch.nn.functional as F




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
    
    

class UNet(BasicModel):
    '''
    Input a (`batch`, `window`, `height`, `width`) sample,
        outputs a (`batch`, `1`, `height`, `width`) result.
    '''

    def __init__(self):
        super().__init__()
        self.down1 = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1, bias=False),
            #self.cgru1()
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
        self.sigmoid = nn.Sigmoid()                          
        
        #self.cgru1 = ConvGRU(32, 64, 3)
        
        
    def forward(self, x):
        #x = self.cgru1(x)
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.bottom(d2)
        d2 = self.up1(d2 + d3)
        d1 = self.up2(d1 + d2)
        x = self.flat(d1)
        x = F.upsample(x, [250, 400], mode='bilinear').squeeze(dim=1) 
        #x = F.relu(x)
        
        result = {}
        result["monocular"] = self.sigmoid(x)
        
        return result