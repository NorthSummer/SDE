import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize
'''
class Eigen(nn.Module):
    def __init__(self):
        super(Eigen, self).__init__()

        self.coarse1 = nn.Conv2d(in_channels=32, out_channels=96, kernel_size=11, stride=4)
        self.pool1 = nn.MaxPool2d(2)
        self.coarse2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5)
        self.pool2 = nn.MaxPool2d(2)
        self.pool3 = nn.MaxPool2d(2)
        self.pool4 = nn.MaxPool2d(2)
        self.coarse3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1)
        self.coarse4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1)
        self.coarse5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3)
        self.coarse6 = nn.Linear(in_features= 8 * 6 * 256, out_features=1 * 4096)  # change to size
        self.coarse7 = nn.Linear(in_features=1 * 4096, out_features=1 * 74 * 55)
        self.refined1 = nn.Conv2d(in_channels=3, out_channels=63, kernel_size=9, stride=2)
        self.refined2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding=2)
        self.refined3 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=5, padding=2)
        self.drop1 = nn.Dropout(p=0.5)
        
        self.resize = Resize([240, 320])

    def forward(self, x):
        x = self.resize(x)
        out = F.relu(self.coarse1(x))
        out = F.relu(self.pool1(out))
        out = F.relu(self.coarse2(out))
        out = F.relu(self.pool2(out))
        out = F.relu(self.coarse3(out))
        out = F.relu(self.coarse4(out))
        out = F.relu(self.coarse5(out))
        out = F.relu(self.pool3(out))
        out = out.view(-1,8*6*256)
        out = F.relu(self.coarse6(out))

        out = self.coarse7(out)
        out = out.reshape(-1,1,74,55)
        out2 = F.relu(self.refined1(x))
        out2 = F.relu(self.pool4(out2))
        out = torch.cat((out, out2), 1)
        out = F.relu(self.refined2(out))
        out = self.refined3(out)
        print(out.size())

        result = {}
        result["monocular"] = out
        return result
'''    

import torch
import torch.nn as nn
from torchvision.models import vgg16
import torch.nn.functional as F

def conv_layer(chann_in, chann_out, k_size, p_size):
    layer = nn.Sequential(
        nn.Conv2d(chann_in, chann_out, kernel_size=k_size, padding=p_size),
        nn.BatchNorm2d(chann_out),
        nn.ReLU()
    )
    return layer

def vgg_conv_block(in_list, out_list, k_list, p_list, pooling_k, pooling_s):

    layers = [ conv_layer(in_list[i], out_list[i], k_list[i], p_list[i]) for i in range(len(in_list)) ]
    layers += [ nn.MaxPool2d(kernel_size = pooling_k, stride = pooling_s)]
    return nn.Sequential(*layers)

def vgg_fc_layer(size_in, size_out):
    layer = nn.Sequential(
        nn.Linear(size_in, size_out),
        nn.BatchNorm1d(size_out),
        nn.ReLU()
    )
    return layer

class VGG16(nn.Module):
    def __init__(self, n_classes=1000):
        super(VGG16, self).__init__()

        # Conv blocks (BatchNorm + ReLU activation added in each block)
        self.layer1 = vgg_conv_block([32,64], [64,64], [3,3], [1,1], 2, 2)
        self.layer2 = vgg_conv_block([64,128], [128,128], [3,3], [1,1], 2, 2)
        self.layer3 = vgg_conv_block([128,256,256], [256,256,256], [3,3,3], [1,1,1], 2, 2)
        self.layer4 = vgg_conv_block([256,512,512], [512,512,512], [3,3,3], [1,1,1], 2, 2)
        self.layer5 = vgg_conv_block([512,512,512], [512,512,512], [3,3,3], [1,1,1], 2, 2)

        # FC layers


    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        vgg16_features = self.layer5(out)
        out = vgg16_features.view(out.size(0), -1)


        return out

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
    
    def forward(self, batch):
        return batch.view([batch.shape[0], -1])


class Scale1_Linear(nn.Module):
    #input 512x7x10
    #output 64x15x20
    
    def __init__(self):
        super(Scale1_Linear, self).__init__()
        self.block = nn.Sequential(
            Flatten(),
            nn.Linear(512*7*10, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 64*15*20)
        )
    
    def forward(self, x):
        scale_1_op = torch.reshape(self.block(x), (x.shape[0], 64, 15, 20))
        return nn.functional.interpolate(scale_1_op, scale_factor=4, mode='bilinear', align_corners=True)


class Scale2(nn.Module):
    #input 64x60x80, 3x240x320
    #output 1x120x160
    
    def __init__(self):
        super(Scale2, self).__init__()
        self.input_img_proc = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=9, padding=4, stride=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
            
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=64+64, out_channels=64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=64),
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=5, padding=2)
        )
    
    def forward(self, x, input_img):
        proc_img = self.input_img_proc(input_img)
        concatenate_input = torch.cat((x,proc_img), dim=1)
        return nn.functional.interpolate(self.block(concatenate_input), scale_factor=2, mode='bilinear', align_corners=True)


class Scale3(nn.Module):
    #input 1x120x160, 3x240x320
    #output 1x120x160
    
    def __init__(self):
        super(Scale3, self).__init__()
        self.input_img_proc = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=9, padding=4, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
            
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=65, out_channels=64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=64),
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=5, padding=2)
        )
    
    def forward(self, x, input_img):
        proc_img = self.input_img_proc(input_img)
        concatenate_input = torch.cat((x,proc_img), dim=1)
        return self.block(concatenate_input)


class Eigen(nn.Module):
    def __init__(self):
        super(Eigen, self).__init__()
        self.VGG = VGG16()#nn.Sequential(*list(vgg16(pretrained=False).children())[0])
        self.Scale_1 = Scale1_Linear()
        self.Scale_2 = Scale2()
        self.Scale_3 = Scale3()
        self.resize = Resize([240, 320])
        
    def forward(self, x):
        x = self.resize(x)
        input_img = x.clone()                  # 3x240x320
        x = self.VGG(x)                        # 512x7x10
        x = self.Scale_1(x)                    # 64x60x80
        x = self.Scale_2(x, input_img.clone()) # 1x120x160
        x = self.Scale_3(x, input_img.clone()) # 1x120x160
        x = F.upsample(x, size=(256,512), mode='bilinear').squeeze(dim=1)
        result = {}
        result["monocular"] = x
        
        return result