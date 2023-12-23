from __future__ import print_function, division
import argparse
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
import numpy as np
import time
from tensorboardX import SummaryWriter
from datasets import __datasets__
from models import __models__, model_loss
from utils import *
from torch.utils.data import DataLoader
import gc
from PIL import Image
from tqdm import tqdm
import cv2
from datasets.SpikeDataset import SpikeDS
from datasets.kitti_dataset import SpikeKitti
from datasets.spikeds import SpikeDrivingStereo

from PIL import Image

cudnn.benchmark = True

from models.ugde_ds import SpikeFusionet
from models.ugde_kitti import SpikeFusionet

device = torch.device("cuda:{}".format(0))

model = SpikeFusionet(max_disp=128, device = device)

def to_video(model):
    
    #state_dict = torch.load('/home/lijianing/depth/MMlogs/256/ours/checkpoint_max_4.6_stage_dual.ckpt')
    state_dict = torch.load('/home/lijianing/depth/CFNet-mod/logs_sup/checkpoint_max_kitti.ckpt')  #checkpoint_max_4.7_base)
    
    
    model.to(device)
    model.load_state_dict(state_dict['model'])
    model.eval()
    #dataset = SpikeDataset(pathr = "/home/Datadisk/spikedata5622/spiking-2022/train/firsthright/", pathl = "/home/Datadisk/spikedata5622/spiking-2022/train/firsthleft/", mode = "training")
    #dataset = SpikeDataset(pathr = "/home/Datadisk/spikedata5622/spiking-2022/test/nrpz/", pathl = "/home/Datadisk/spikedata5622/spiking-2022/test/nlpz/", mode = "test")
    #dataset = SpikeDS(pathr = "/home/Datadisk/spikedata5622/spiking-2022/SpikeDriveStereo/trainset/rightspike/Sequence_33/spike/", pathl = "/home/Datadisk/spikedata5622/spiking-2022/SpikeDriveStereo/trainset/leftspike/Sequence_33      /spike/", mode = "train")
    #dataset = JYSpikeDataset(path="/home/Datadisk/spikedata5622/spiking-2022/JYsplit/train/") 
    #dataset = SpikeDataset(pathr = "/home/Datadisk/spikedata5622/spiking-2022/test/nrpz/", pathl = "/home/Datadisk/spikedata5622/spiking-2022/test/nlpz/", mode = "test")
    dataset = SpikeKitti(spike_base_path="/home/Datadisk/spikedata5622/spiking-2022/spikesterkitti/", depth_base_path="/home/lijianing/kitti_depth/", split = "val")
    #dataset = SpikeDrivingStereo(spike_base_path="/home/Datadisk/spikedata5622/spiking-2022/spikeds/", depth_base_path="/home/Datadisk/spikedata5622/DrivingStereo/train-depth-map/",         split = "val")
    dataloader = DataLoader(dataset, 1, False)
    #print(dataset.filesr)
    
    fps=2
    fourcc=cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter('kitti-30.mp4', cv2.VideoWriter_fourcc(*"mp4v"), fps, (1242,375))
    video1 = cv2.VideoWriter('rgb.mp4', cv2.VideoWriter_fourcc(*"mp4v"), fps, (881,400))
    
    fourcc1 = cv2.VideoWriter_fourcc('M','J','P','G')
    videoWrite = cv2.VideoWriter( '2.avi', fourcc1, 5, [250,400] ) 
    
    
    
    for data in tqdm(dataloader):
        x_l, x_r, real_y, real_d = data["left"].to(device), data["right"].to(device), data["disparity"].to(device), data["depth"].to(device)#, data["rgb"]#.to(device), rgb_r
        #print(rgb_r)

        #fake_y = model(x_l, x_r)["monocular"]["depth"]#["uncertainty"]#["stereo"][-1]
        ests = model(x_l,x_r)
        est_disp = ests["stereo"][-1]
        est_mono = ests["monocular"]["depth"]
        
        #y_map = fake_y[-1].detach().cpu()
        y_map_1 = est_mono.detach().cpu()            
        y_map_2 = est_disp.detach().cpu()
        print(y_map_2)
               

        y_map_1 = np.array((1 / y_map_1), dtype = np.float32)
        y_map_2 = np.array((1 / y_map_2), dtype = np.float32)#1 / y_map_2
        
        
        y_map = y_map_2
        
              
        y_map[ y_map> 30.0] = 30.0
        y_map[ y_map< 0.3] = 0.3
        
        
        y_map = (196.0 / 30.0) * y_map
        
        y_map = y_map.squeeze(0)      
                
        
        fig = Image.fromarray(y_map.astype(np.uint8))#.convert('L')
     
        fig = cv2.cvtColor(np.asarray(fig),cv2.COLOR_GRAY2RGB) 
        fig = cv2.applyColorMap(fig, cv2.COLORMAP_TURBO)
        video.write(fig)

def vis_spike(npy):

    npy = 255*npy[16]
    img = Image.fromarray(255*npy).convert("L")
    
    return img
    
if "__main__" == __name__:
    device = torch.device("cuda:{}".format(0))
    model = SpikeFusionet(max_disp=128, device = device)
    to_video(model)