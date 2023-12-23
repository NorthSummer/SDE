import torch
import torch.nn as nn
affine_par = True
import torch.nn.functional as F

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv7x7(in_planes, out_planes, stride=2):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=7, stride=stride,
                     padding=3, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, fist_dilation=1, multi_grid=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation * multi_grid, dilation=dilation * multi_grid, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=False)
        self.relu_inplace = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu_inplace(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers):
        self.inplanes = 128
        super(ResNet, self).__init__()
        self.conv1 = conv3x3(32, 64, stride=2)
        self.bn1 = nn.BatchNorm2d(64, momentum=0.95)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(64, 64)
        self.bn2 = nn.BatchNorm2d(64, momentum=0.95)
        self.relu2 = nn.ReLU(inplace=False)
        self.conv3 = conv3x3(64, 128)
        self.bn3 = nn.BatchNorm2d(128, momentum=0.95)
        self.relu3 = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

#         self.relu = nn.ReLU(inplace=False)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)  # change
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4, multi_grid=(1, 1, 1))

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, multi_grid=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, affine=affine_par))

        layers = []
        generate_multi_grid = lambda index, grids: grids[index % len(grids)] if isinstance(grids, tuple) else 1
        layers.append(block(self.inplanes, planes, stride, dilation=dilation, downsample=downsample,
                            multi_grid=generate_multi_grid(0, multi_grid)))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(self.inplanes, planes, dilation=dilation, multi_grid=generate_multi_grid(i, multi_grid)))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def freeze(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()


class ResNet50(nn.Module):

    def __init__(self, pretrained=False):
        super().__init__()
        self.backbone = ResNet(Bottleneck, [3, 4, 6, 3])

        if pretrained:
            saved_state_dict = torch.load('/datasets/KITTI/depth_prediction/pretrained/resnet101-imagenet.pth',
                                          map_location="cpu")
            new_params = self.backbone.state_dict().copy()
            for i in saved_state_dict:
                i_parts = i.split('.')
                if not i_parts[0] == 'fc':
                    new_params['.'.join(i_parts[0:])] = saved_state_dict[i]

            self.backbone.load_state_dict(new_params)

    def forward(self, input):
        return self.backbone(input)
   
   
        
class RegressionLayer(nn.Module):
    def __init__(self):
        super(RegressionLayer, self).__init__()
        
        self.outc = nn.Conv2d(2048, 256, 1, 1, 0)
        
    def forward(self, x):
        """
        :input x: shape = (N,C,H,W), C = 2*ord_num (2*K)
        :return: ord prob is the label probability of each label, N x OrdNum x H x W
        """
        '''
        N, C, H, W = x.size() # (N, 2K, H, W)
        ord_num = C // 2
        
        label_0 = x[:, 0::2, :, :].clone().view(N, 1, ord_num, H, W)  # (N, 1, K, H, W)
        label_1 = x[:, 1::2, :, :].clone().view(N, 1, ord_num, H, W)  # (N, 1, K, H, W)

        label = torch.cat((label_0, label_1), dim=1) # (N, 2, K, H, W)
        label = torch.clamp(label, min=1e-8, max=1e8)  # prevent nans
        '''
        x = F.interpolate(x, size=[256,512], mode="bilinear", align_corners=True)
        label = self.outc(x)
        label_ord = torch.nn.functional.softmax(label, dim=1)
        
        N,C,H,W = label_ord.size() 
        value = torch.linspace(0, 1, 256).view(1, 256, 1, 1).cuda()
        value = value.repeat(N, 1, H, W)
        value = value * label_ord
        d = torch.sum(value,dim=1)
        #prob = label_ord[:,1,:,:,:].clone() # label_ord is the output softmax probability of this model
        print(d.size())
        return d#prob        

class MonoRes(nn.Module):
    def __init__(self):
        super(MonoRes, self).__init__()
        self.backbone = ResNet50(pretrained = False)
        self.decoder = RegressionLayer()
        
    def forward(self, x):
        x = self.backbone(x)
        x = self.decoder(x)
        
        return x    
        
        
        
                 
        