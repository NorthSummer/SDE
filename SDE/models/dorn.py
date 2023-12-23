# nothing
import torch
import torch.nn as nn
import torch.nn.functional as F
affine_par = True
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize
import numpy as np
device = torch.device("cuda:{}".format(2))




class FullImageEncoder(nn.Module):
    def __init__(self, h, w, kernel_size):
        super(FullImageEncoder, self).__init__()
#         self.global_pooling = nn.AvgPool2d(kernel_size, stride=kernel_size, padding=kernel_size // 2)  # KITTI 16 16
        self.global_pooling = nn.AvgPool2d(kernel_size, stride=kernel_size)  # KITTI 16 16
        self.dropout = nn.Dropout2d(p=0.5)
        self.h = h // kernel_size 
        self.w = w // kernel_size
        self.global_fc = nn.Linear(2048 * self.h * self.w, 512)  # kitti 4x5
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(512, 512, 1)  # 1x1 conv
        self.relu1 = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.global_pooling(x)
        x = self.dropout(x)
        x = x.view(-1, 2048 * self.h * self.w)  # kitti 4x5
        x = self.relu(self.global_fc(x))
        x = x.view(-1, 512, 1, 1)
        x = self.conv1(x)
        x = self.relu1(x)
        return x


class SceneUnderstandingModule(nn.Module):
    def __init__(self, ord_num, size, kernel_size, pyramid=[6, 12, 18]):
        # pyramid kitti [6, 12, 18] nyu [4, 8, 12]
        super(SceneUnderstandingModule, self).__init__()
        assert len(size) == 2
        assert len(pyramid) == 3
        self.size = size
        h, w = self.size
        self.encoder = FullImageEncoder(h // 8, w // 8, kernel_size)
        self.aspp1 = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=1, padding=0, dilation=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=1, padding=0, dilation=1),
            nn.ReLU(inplace=True)
        )
        self.aspp2 = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=3, padding=pyramid[0], dilation=pyramid[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=1, padding=0, dilation=1),
            nn.ReLU(inplace=True)
        )
        self.aspp3 = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=3, padding=pyramid[1], dilation=pyramid[1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=1, padding=0, dilation=1),
            nn.ReLU(inplace=True)
        )
        self.aspp4 = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=3, padding=pyramid[2], dilation=pyramid[2]),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=1, padding=0, dilation=1),
            nn.ReLU(inplace=True)
        )
        self.concat_process = nn.Sequential(
            nn.Dropout2d(p=0.5),
            nn.Conv2d(512*5, 2048, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5),
            nn.Conv2d(2048, ord_num * 2, kernel_size=1)
        )

    def forward(self, x):
        N, C, H, W = x.shape
        x1 = self.encoder(x)
        x1 = F.interpolate(x1, size=(H, W), mode="bilinear", align_corners=True)
        x2 = self.aspp1(x)
        x3 = self.aspp2(x)
        x4 = self.aspp3(x)
        x5 = self.aspp4(x)

        x6 = torch.cat((x1, x2, x3, x4, x5), dim=1)
        out = self.concat_process(x6)
        out = F.interpolate(out, size=self.size, mode="bilinear", align_corners=True)
        return out




   
        
class OrdinalRegressionLayer(nn.Module):
    def __init__(self):
        super(OrdinalRegressionLayer, self).__init__()

    def forward(self, x):
        """
        :input x: shape = (N,C,H,W), C = 2*ord_num (2*K)
        :return: ord prob is the label probability of each label, N x OrdNum x H x W
        """
        N, C, H, W = x.size() # (N, 2K, H, W)
        ord_num = C // 2
        
        label_0 = x[:, 0::2, :, :].clone().view(N, 1, ord_num, H, W)  # (N, 1, K, H, W)
        label_1 = x[:, 1::2, :, :].clone().view(N, 1, ord_num, H, W)  # (N, 1, K, H, W)

        label = torch.cat((label_0, label_1), dim=1) # (N, 2, K, H, W)
        label = torch.clamp(label, min=1e-8, max=1e8)  # prevent nans

        label_ord = torch.nn.functional.softmax(label, dim=1)
        prob = label_ord[:,1,:,:,:].clone() # label_ord is the output softmax probability of this model
        return prob
        


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
        self.conv1 = conv3x3(32, 64, stride=2) ####3 # modify input channels  
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


class ResNet101(nn.Module):

    def __init__(self, pretrained=True):
        super().__init__()
        self.backbone = ResNet(Bottleneck, [3, 4, 6, 3]) #23

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


class OrdinalRegressionLoss(torch.nn.Module):

    def __init__(self, ord_num, beta, discretization="SID"):
        super(OrdinalRegressionLoss, self).__init__()
        self.ord_num = ord_num
        self.beta = beta
        self.discretization = discretization

    def _create_ord_label(self, depth):
        N, H, W = depth.shape
        depth = depth.unsqueeze(1)

        ord_c0 = torch.ones(N, self.ord_num, H, W).to(depth.device)
        if self.discretization == "SID":
            label = self.ord_num * torch.log(depth) / np.log(self.beta)
        else:
            label = self.ord_num * (depth - 1.0) / (self.beta - 1.0)
        label = label.long()
        mask = torch.linspace(0, self.ord_num - 1, self.ord_num, requires_grad=False) \
            .view(1, self.ord_num, 1, 1).to(depth.device)
        mask = mask.repeat(N, 1, H, W).contiguous().long()
        mask = (mask < label)
        ord_c0[mask] = 0
        ord_c1 = 1 - ord_c0
        return ord_c0.to(device), ord_c1.to(device)

    def forward(self, prob, depth):
        """
        :param prob: ordinal regression probability, N x 2*Ord Num x H x W, torch.Tensor
        :param gt: depth ground truth, NXHxW, torch.Tensor
        :return: loss: loss value, torch.float
        """
        N, C, H, W = prob.shape
        valid_mask = depth > 0.
        ord_c0, ord_c1 = self._create_ord_label(depth)
        logP = torch.log(torch.clamp(prob, min=1e-8)).to(device)
        log1_P = torch.log(torch.clamp(1 - prob, min=1e-8)).to(device)
        
        entropy = torch.sum(ord_c1*logP, dim=1) + torch.sum(ord_c0*log1_P, dim=1) # eq. (2)
        
        valid_mask = torch.squeeze(valid_mask, 1)
        loss = - entropy[valid_mask].mean()
        return loss


class DORN(torch.nn.Module):

    def __init__(self, ord_num=256, input_size=(385, 513), kernel_size=16, pyramid=[6, 12, 18], pretrained=False):
        super().__init__()
        assert len(input_size) == 2
        assert isinstance(kernel_size, int)
        self.ord_num = ord_num
        self.resnet101 = ResNet101(pretrained=pretrained)
        self.scene_understanding_module = SceneUnderstandingModule(ord_num, size=input_size,
                                                                 kernel_size=kernel_size,
                                                                 pyramid=pyramid)
        self.ord_regression_layer = OrdinalRegressionLayer()
        self.resize = Resize([385, 513]) 

    def forward(self, image, target=None):
        """
        :input: image: torch.Tensor, (N,3,H,W)
                target: target depth, torch.Tensor, (N,H,W)
                
        :return:prob: probability of each label, torch.Tensor, (N,K,H,W), K is the ordinal number 
                label: predicted label, torch.Tensor, (N,H,W)
        """
        image = self.resize(image)
        N, C, H, W = image.shape
        
        
        
        feat = self.resnet101(image)
        feat = self.scene_understanding_module(feat) # (N, 2K, H, W) > (N, 160, 385, 513)
        prob = self.ord_regression_layer(feat) # (N, K, H, W)
        
        # calculate label
        label = torch.sum((prob >= 0.5), dim=1).view(-1, 1, H, W) # (N, 1, H, W)
    
        return prob, label
     