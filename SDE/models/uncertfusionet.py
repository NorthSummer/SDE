import math
import torch
import torch.nn as nn
#from models.costnet import CostNet
#from models.stackedhourglass import StackedHourglass
import torch.nn.functional as F
from models.submodule import *

device = torch.device("cuda:{}".format(2))

class SpikeFusionet(nn.Module):

    def __init__(self, max_disp):
        super().__init__()
        
        self.encoder = Encoder()
        
        self.decoder = Decoder()
        
        self.cost_net = CostNet()
        
        
        
        self.stackedhourglass = StackedHourglass(max_disp)
        self.D = max_disp
        self.norm = Generator()
        self.norm.load_state_dict(torch.load("/home/lijianing/snn/spikling-master/checkpoint/spikling-0100.pth"))
                
        

        self.__init_params()

    def forward(self, left_img, right_img):
        original_size = [self.D, left_img.size(2), left_img.size(3)]

        left_coding = self.encoder(left_img)
        #print("left coding size : {}".format(left_coding[0].size()))
        right_coding = self.encoder(right_img)
        
        right_depth = self.decoder(right_coding)
        
        
        left_cost = self.cost_net(left_coding)  
        right_cost = self.cost_net(right_coding)  


        B, C, H, W = left_cost.size()

        cost_volume = torch.zeros(B, C * 2, self.D // 4, H, W).type_as(left_cost)  


        for i in range(self.D // 4):
            if i > 0:
                cost_volume[:, :C, i, :, i:] = left_cost[:, :, :, i:]
                cost_volume[:, C:, i, :, i:] = right_cost[:, :, :, :-i]
            else:
                cost_volume[:, :C, i, :, :] = left_cost
                cost_volume[:, C:, i, :, :] = right_cost

        disp1, disp2, disp3, unc3 = self.stackedhourglass(cost_volume, out_size=original_size)
        if self.training or self.eval:
            disp3 = disp3.unsqueeze(1)
            disp3 = F.upsample(disp3, [256,512], mode='bilinear', align_corners=True).squeeze(1)
            
            mono_uncert = right_depth["uncertainty"]
            ster_uncert = unc3[0]
 
            
            C, H, W = mono_uncert.size()
            mask_dual = torch.zeros((C,H,W), dtype=torch.float, requires_grad=True).cuda()
            
            mask_dual[  ster_uncert >= 0.5  ] = mask_dual[  ster_uncert >= 0.5  ] + 0
            mask_dual[  ster_uncert < 0.5  ]  = mask_dual[  ster_uncert < 0.5  ] + 1
            
            mask_dual[  mono_uncert <= 0.005  ]  = mask_dual[  mono_uncert <= 0.005  ] + 2
            mask_dual[  mono_uncert > 0.005  ]  = mask_dual[  mono_uncert > 0.005  ] + 0
            
            
            mask_dual = mask_dual / 2
            
            mask_dual[ mask_dual == 1.5] = 0.5 
            mask_dual[ mask_dual == 1] = 0 
            mask_dual[ mask_dual == 0.5] = 1 
            mask_dual[ mask_dual == 0] = 0.5 
                        

            fusion = mask_dual * right_depth["depth"] + (1-mask_dual) * (1 / disp3)
            fusion = (fusion / 128.0) - 1.0
                      
            result = {}
            result["monocular"] = right_depth
            result["stereo"] = [disp1, disp2, disp3] 
            result["stereo_uncertainty"]= unc3[0]
            result["fusion"] = fusion

        
        return result

    def __init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
                
        
class CostNet(nn.Module):

    def __init__(self):
        super().__init__()

        #self.cnn = Encoder() #CNN()
        self.spp = SPP()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.fusion = nn.Sequential(
                Conv2dBn(in_channels=320, out_channels=128, kernel_size=3, stride=1, padding=1, use_relu=True),
                nn.Conv2d(in_channels=128, out_channels=32, kernel_size=1, stride=1, padding=0, bias=False)
            )

    def forward(self, inputs):
        #conv2_out, conv4_out = self.cnn(inputs)           # [B, 64, 1/4H, 1/4W], [B, 128, 1/4H, 1/4W]
        conv2_out, conv4_out = inputs[0], inputs[1]
        #print(conv2_out.size(), conv4_out.size())
        conv4_out = self.upsample(conv4_out)
        spp_out = self.spp(conv4_out)       
                     # [B, 128, 1/4H, 1/4W]
        
        out = torch.cat([conv2_out, conv4_out, spp_out], dim=1)  # [B, 320, 1/4H, 1/4W]
        out = self.fusion(out)                            # [B, 32, 1/4H, 1/4W]

        return out


class SPP(nn.Module):

    def __init__(self):
        super().__init__()

        self.branch1 = self.__make_branch(kernel_size=16, stride=16)
        self.branch2 = self.__make_branch(kernel_size=8, stride=8)
        self.branch3 = self.__make_branch(kernel_size=4, stride=4)
        self.branch4 = self.__make_branch(kernel_size=2, stride=2)
        
    def forward(self, inputs):

        out_size = inputs.size(2), inputs.size(3)
        branch1_out = F.upsample(self.branch1(inputs), size=out_size, mode='bilinear')  # [B, 32, 1/4H, 1/4W]
        # print('branch1_out')
        # print(branch1_out[0, 0, :3, :3])
        branch2_out = F.upsample(self.branch2(inputs), size=out_size, mode='bilinear')  # [B, 32, 1/4H, 1/4W]
        branch3_out = F.upsample(self.branch3(inputs), size=out_size, mode='bilinear')  # [B, 32, 1/4H, 1/4W]
        branch4_out = F.upsample(self.branch4(inputs), size=out_size, mode='bilinear')  # [B, 32, 1/4H, 1/4W]
        
        out = torch.cat([branch4_out, branch3_out, branch2_out, branch1_out], dim=1)  # [B, 128, 1/4H, 1/4W]

        return out

    @staticmethod
    def __make_branch(kernel_size, stride):
        branch = nn.Sequential(
                nn.AvgPool2d(kernel_size, stride),
                Conv2dBn(in_channels=128, out_channels=32, kernel_size=3, stride=1, padding=1, use_relu=True)  # kernel size maybe 1
            )
        return branch



class Encoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv0 = nn.Sequential(
                Conv2dBn(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1, use_relu=True),  # downsample
                Conv2dBn(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, use_relu=True),
                Conv2dBn(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, use_relu=True)
            )


        self.conv1 = StackedBlocks(n_blocks=3, in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv2 = StackedBlocks(n_blocks=16, in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1, dilation=1)  # downsample
        self.conv3 = StackedBlocks(n_blocks=3, in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=2, dilation=2)  # dilated
        self.conv4 = StackedBlocks(n_blocks=3, in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=4, dilation=4)  # dilated
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
    def forward(self, inputs):
        conv0_out = self.conv0(inputs)
        conv1_out = self.conv1(conv0_out)  # [B, 32, 1/2H, 1/2W]
        conv2_out = self.conv2(conv1_out)  # [B, 64, 1/4H, 1/4W]
        conv3_out = self.conv3(conv2_out)  # [B, 128, 1/4H, 1/4W]
        conv4_out = self.conv4(conv3_out) 
        conv5_out = self.upsample(conv4_out)            # [B, 128, 1/4H, 1/4W] !!!!!!!!

        return [conv2_out, conv4_out]


class Decoder(nn.Module): # add uncertainty module

    def __init__(self):
        super().__init__()


        self.up1 =  nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True), # Mish?
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True), # Mish?
        )
        
        
        self.up2 =  nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),#Mish(), # Mish?
            nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
        )
        
                
        self.up3 =  nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),#Mish(),# Mish?
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),                        
        )
        
        self.out = nn.Sequential(nn.Conv2d(32, 2, kernel_size=3, padding=1, bias=False))
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        
    def forward(self, inputs):
        
        inputs = inputs[1]
        up_stage1 = self.up1(inputs)
        up_stage2 = self.up2(up_stage1)
        up_stage3 = self.up3(up_stage2)
        
        #output = self.out(up_stage3)
        output = self.out(up_stage3)
        result = {}
        #print(output.size())
        result["depth"] = output[:,0,:,:].squeeze(1)
        result["uncertainty"] = self.sigmoid(output[:,1,:,:]).squeeze(1)
        
        return result#.squeeze(1)



class StackedBlocks(nn.Module):

    def __init__(self, n_blocks, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1):
        super().__init__()

        if stride == 1 and in_channels == out_channels:
            downsample = False
        else:
            downsample = True
        net = [ResidualBlock(in_channels, out_channels, kernel_size, stride, padding, dilation, downsample)]

        for i in range(n_blocks - 1):
            net.append(ResidualBlock(out_channels, out_channels, kernel_size, 1, padding, dilation, downsample=False))
        self.net = nn.Sequential(*net)

    def forward(self, inputs):
        out = self.net(inputs)
        return out


class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, downsample=False):
        super().__init__()

        self.net = nn.Sequential(
                Conv2dBn(in_channels, out_channels, kernel_size, stride, padding, dilation, use_relu=True),
                Conv2dBn(out_channels, out_channels, kernel_size, 1, padding, dilation, use_relu=False)
            )

        self.downsample = None
        if downsample:
            self.downsample = Conv2dBn(in_channels, out_channels, 1, stride, use_relu=False)

    def forward(self, inputs):
        out = self.net(inputs)
        if self.downsample:
            inputs = self.downsample(inputs)
        out = out + inputs

        return out


class Conv2dBn(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, use_relu=True):
        super().__init__()

        net = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=False),
               nn.BatchNorm2d(out_channels)]
        if use_relu:
            net.append(Mish())#nn.ReLU(inplace=True)
        self.net = nn.Sequential(*net)

    def forward(self, inputs):
        out = self.net(inputs)
        return out
        
class StackedHourglass(nn.Module):
    '''
    inputs --- [B, 64, 1/4D, 1/4H, 1/4W]
    '''

    def __init__(self, max_disp):
        super().__init__()

        self.conv0 = nn.Sequential(
            Conv3dBn(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1, dilation=1, use_relu=True),
            Conv3dBn(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, dilation=1, use_relu=True)
        )
        self.conv1 = nn.Sequential(
            Conv3dBn(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, dilation=1, use_relu=True),
            Conv3dBn(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, dilation=1, use_relu=False)
        )
        self.hourglass1 = Hourglass()
        self.hourglass2 = Hourglass()
        self.hourglass3 = Hourglass()

        self.out1 = nn.Sequential(
            Conv3dBn(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, dilation=1, use_relu=True),
            nn.Conv3d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1, dilation=1, bias=False)
        )
        self.out2 = nn.Sequential(
            Conv3dBn(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, dilation=1, use_relu=True),
            nn.Conv3d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1, dilation=1, bias=False)
        )
        self.out3 = nn.Sequential(
            Conv3dBn(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, dilation=1, use_relu=True),
            nn.Conv3d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1, dilation=1, bias=False)
        )

        self.regression = DisparityRegression(max_disp)

    def forward(self, inputs, out_size):

        conv0_out = self.conv0(inputs)     # [B, 32, 1/4D, 1/4H, 1/4W]
        conv1_out = self.conv1(conv0_out)
        conv1_out = conv0_out + conv1_out  # [B, 32, 1/4D, 1/4H, 1/4W]

        hourglass1_out1, hourglass1_out3, hourglass1_out4 = self.hourglass1(conv1_out, scale1=None, scale2=None, scale3=conv1_out)
        hourglass2_out1, hourglass2_out3, hourglass2_out4 = self.hourglass2(hourglass1_out4, scale1=hourglass1_out3, scale2=hourglass1_out1, scale3=conv1_out)
        hourglass3_out1, hourglass3_out3, hourglass3_out4 = self.hourglass3(hourglass2_out4, scale1=hourglass2_out3, scale2=hourglass1_out1, scale3=conv1_out)

        out1 = self.out1(hourglass1_out4)  # [B, 1, 1/4D, 1/4H, 1/4W]
        out2 = self.out2(hourglass2_out4) + out1
        out3 = self.out3(hourglass3_out4) + out2

        cost1 = F.upsample(out1, size=out_size, mode='trilinear').squeeze(dim=1)  # [B, D, H, W]
        cost2 = F.upsample(out2, size=out_size, mode='trilinear').squeeze(dim=1)  # [B, D, H, W]
        cost3 = F.upsample(out3, size=out_size, mode='trilinear').squeeze(dim=1)  # [B, D, H, W]

        prob1 = F.softmax(-cost1, dim=1)  # [B, D, H, W]
        prob2 = F.softmax(-cost2, dim=1)
        prob3 = F.softmax(-cost3, dim=1)

        disp1, unc1 = self.regression(prob1)
        disp2, unc2 = self.regression(prob2)
        disp3, unc3 = self.regression(prob3)
        
        
        return disp1, disp2, disp3, unc3


class DisparityRegression(nn.Module):

    def __init__(self, max_disp):
        super().__init__()

        self.disp_score = torch.range(0, max_disp - 1)  # [D]
        self.disp_score = self.disp_score.unsqueeze(0).unsqueeze(2).unsqueeze(3)  # [1, D, 1, 1]

    def forward(self, prob):
        disp_score = self.disp_score.expand_as(prob).type_as(prob)  # [B, D, H, W]
        out = torch.sum(disp_score * prob, dim=1) / 256  # [B, H, W], 256 add
        
        out_uncert = torch.max(prob, dim=1) #torch.sum(prob * torch.log(prob), dim=1)#.mean() 
        
        return out, out_uncert


class Hourglass(nn.Module):

    def __init__(self):
        super().__init__()

        self.net1 = nn.Sequential(
            Conv3dBn(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1, dilation=1, use_relu=True),
            Conv3dBn(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, dilation=1, use_relu=False)
        )
        self.net2 = nn.Sequential(
            Conv3dBn(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, dilation=1, use_relu=True),
            Conv3dBn(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, dilation=1, use_relu=True)
        )
        self.net3 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm3d(num_features=64),
           
            # Mish(),nn.ReLU(inplace=True)
        )
        self.net4 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm3d(num_features=32)
        )

    def forward(self, inputs, scale1=None, scale2=None, scale3=None):
        net1_out = self.net1(inputs)  # [B, 64, 1/8D, 1/8H, 1/8W]

        if scale1 is not None:
            net1_out = F.relu(net1_out + scale1, inplace=True)
        else:
            net1_out = F.relu(net1_out, inplace=True)

        net2_out = self.net2(net1_out)  # [B, 64, 1/16D, 1/16H, 1/16W]
        net3_out = self.net3(net2_out)  # [B, 64, 1/8D, 1/8H, 1/8W]

        if scale2 is not None:
            net3_out = F.relu(net3_out + scale2, inplace=True)
        else:
            net3_out = F.relu(net3_out + net1_out, inplace=True)

        net4_out = self.net4(net3_out)

        if scale3 is not None:
            net4_out = net4_out + scale3
        

        return net1_out, net3_out, net4_out


class Conv3dBn(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, use_relu=True):
        super().__init__()

        net = [nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=False),
               nn.BatchNorm3d(out_channels)]
        if use_relu:
            net.append(Mish())#nn.ReLU(inplace=True)

        self.net = nn.Sequential(*net)

    def forward(self, inputs):
        out = self.net(inputs)
        return out
        
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