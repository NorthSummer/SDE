
import os
import re
import numpy as np
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from PIL import Image

import cv2
import OpenEXR
import Imath
import pytorch_ssim

from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize, transforms
from . import readpfm as rp

import numpy as np
import argparse
import math
import scipy
import struct
from scipy import integrate
import cv2  
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import os



class SpikeReal(Dataset):

    def __init__(self, root_path: str, split = "train"):   # train / val
    
        self.H = 250
        self.W = 400
        self.depth_scale = 1000
        
        
        self.spike_base_path = os.path.join(root_path, "spiking", split)  #/home/Datadisk/spikedata5622/spiking-2022/          
        self.depth_base_path = os.path.join(root_path, "depth_trans", split)      
        self.spike_sequences = os.listdir(self.spike_base_path) # e.g. 2018-10-16-11-43-02
        
        self.rec_base_path = os.path.join(root_path, "reconstruction", split)
        
        
        self.depth_list = []
        self.spike_list_left = []
        self.spike_list_right = []
        self.rec_list_left = []
        self.rec_list_right = []
        
        
        for s in self.spike_sequences:
            s_seq_path = os.path.join(self.spike_base_path, s)
           
            pics_left = os.listdir(os.path.join(s_seq_path, "left"))
            for pic in pics_left:
                if pic.endswith("npy"):
                #print(os.path.join(s_seq_path, "left", pic))
                    self.spike_list_left.append(os.path.join(s_seq_path, "left", pic))
                    depth_left = os.path.join(self.depth_base_path, s, pic.replace("dat", "pfm"))
                    self.depth_list.append(depth_left)
    
                    self.spike_list_right.append(os.path.join(s_seq_path, "right", pic))

        self.depth_list.sort()   
        self.spike_list_left.sort() 
        self.spike_list_right.sort()
        #self.rgb_list_left.sort()
        #self.rgb_list_right.sort()
        
        
        self.split = split
        
    def __len__(self):
        return len(self.spike_list_left)

    def __getitem__(self, idx):

        
        spike_path_left = self.spike_list_left[idx]
        spike_path_right = spike_path_left.replace("left","right")
        #print(spike_path_right)

        if self.split == "train":
            this_sequence = spike_path_left[75:78]  
            this_order = spike_path_left[85: -4]
        elif self.split == "val":
            this_sequence = spike_path_left[73:76]  
            this_order = spike_path_left[83: -4]            
        #print(this_sequence, this_order)
        
        
        #depth_gt_path = os.path.join(self.depth_base_path, this_sequence, this_order) + "."
        depth_gt_path = self.depth_list[idx]
        #depth_gt = np.expand_dims(self.get_gt_depth_maps(depth_gt_path), 0) 
        
        #print(depth_gt_path)
        depth_gt = np.expand_dims(self.get_gt_disparity(depth_gt_path.replace(".npy",".pfm")), 0) 
       
        depth_gt = torch.FloatTensor(depth_gt)
        
        
        
        
        '''
        rgb_left = np.array(Image.open(self.rgb_list_left[idx])).transpose(2,0,1)
        rgb_right = np.array(Image.open(self.rgb_list_right[idx])).transpose(2,0,1)
        '''
        '''       
        rgb_left = torch.FloatTensor(rgb_left)
        rgb_right = torch.FloatTensor(rgb_right)
        '''

        disp_gt =  depth_gt #* 10
        #disp_gt = disp_gt.int()
        disp_gt[disp_gt >200] = 200
        disp_gt[disp_gt <=0 ] = -1
        disp_gt = 1 / disp_gt
        #disp_gt = disp_gt.int()
        disp_gt[disp_gt <=0 ] = 0
        
        
        #depth_gt[depth_gt > 100] = 100.0
        
        
        #print(torch.min(depth_gt), torch.max(depth_gt))
        #depth_gt = depth_gt / 20 #1/depth_gt #/ 100.0
        depth_gt = depth_gt / 20.0
       
        npy_path = os.path.join(self.spike_base_path, this_sequence, "spike/r50/{}.npy".format(this_order)) #spike_ 
        
        spike_mat_left = self.load_np(spike_path_left)#self.analysedat(dat_path)
        #spike_mat_left = self.analysedat(spike_path_left)
        #spike_mat_left = 2*spike_mat_left - 1 #+ 1
        spike_mat_right = self.load_np(spike_path_right)#self.analysedat(dat_path)
        #spike_mat_right = self.analysedat(spike_path_right)
        #spike_mat_right = 2*spike_mat_right - 1#- 1

        
        spike_left = torch.FloatTensor(spike_mat_left)
        spike_right =  torch.FloatTensor(spike_mat_right)

        scale = transforms.Compose([
         transforms.Resize([128, 256]),
         ])  

        sample = {}
        sample["left"] = scale(spike_left)
        sample["right"] = scale(spike_right)
        sample["depth"] = depth_gt
        sample["disparity"] = disp_gt.squeeze(0)
        #sample["rgb_left"] = scale(rgb_left)
        #sample["rgb_right"] = scale(rgb_right)
  
        return sample#spike, depth
        
    def get_gt_depth_maps(self, depth_map_path):
        
        depth_map_gt = np.array(Image.open(depth_map_path), dtype=np.float32) / self.depth_scale
        
        return depth_map_gt

    def get_gt_disparity(self, disp_map_path):

 
        data = rp.readPFM(disp_map_path)[0]
        data = np.ascontiguousarray(data, dtype=np.float32)
        return data
        
        
    def load_np(self, np_path):
        npy = np.load(np_path).astype(np.uint8)
        return torch.FloatTensor(npy)

    def random_crop(self, spike, rgb, depth, height, width):  # load numpy format # height = 352 width = 1216
        '''
        assert spike.shape[1] >= height
        assert spike.shape[2] >= width
        assert spike.shape[1] == depth.shape[1]
        assert spike.shape[2] == depth.shape[2]
        '''
        #print(spike)
        
        x = random.randint(0, spike.shape[2] - width)
        y = random.randint(0, spike.shape[1] - height)
        spike = spike[:, y:y + height, x:x + width]
        depth = depth[:, y:y + height, x:x + width]
        #rgb = rgb[y:y + height, x:x + width, :]
        return spike, depth
        
    def refine(self, mat):
        mat[mat > 100] = 100
        return mat

        
    def analysedat(self, dat_file_path):
        c = self.W #1242 # 400
        r = self.H #375 # 250
        frame = 0
        frame_tot = 101
        stop_flag = 0
        pos = 0
        ff = []
        count = 0
  
        sum = np.zeros((400, r, c))
        binfile = open(dat_file_path, 'rb')
        while(1):
            if count > 8000:
                break
            a = binfile.read(1)
            if not a:
                break
            real_a = struct.unpack('b', a)[0]

            for i in range(8):
                pan = (real_a & (1 << (7 - i)))
                if pan !=0:
                    pan = 1
                ff.append(pan)
                
                pos += 1
            
            if pos >=  r * c: 
                pos = 0
                       
                sum[count,:,:] = self.list2img(ff, dat_file_path)
                count = count + 1
                ff = []
                
                break
        np.save(dat_file_path.replace(".dat",".npy"), sum)
        print(sum[350,:,:])
        return sum

    def list2img(self, list_, path):
        w = self.W
        h = self.H
        
        path = path.replace("dat", "npy")
        
        #name = os.path.join("/home/Datadisk/SpikeSet/SpikeData/", path[-8:-4]) + ".png"
        name = path.replace(".dat", ".npy")
        
        mat = np.array(list_).reshape(h, w)
        img = Image.fromarray(np.uint8(mat))
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
        img = np.array(img, dtype = np.uint8)
        name = name.replace(".png","")
        
        
        
        return img
        #img.save(name)   #/home/Datadisk/spikedata5622/spiking-2022/outdoor_real_spike/spiking/train/0040/left/      
                          #/home/Datadisk/spikenpya5622/spiking-2022/outdoor_real_spike/spiking/train/0000/left/0000.npy   

    
    def list2np(self, sum, list_, index): # right or left
        w = 1242
        h = 375
        if index % 24 == 0:
            sum = np.zeros((24, h, w))
       
        mat = np.array(list_).reshape(24, h, w)
        

        return mat                 









class SpikeRealII(Dataset):

    def __init__(self, root_path: str, split = "train"):   # train / val
    
        self.H = 250
        self.W = 400
        self.depth_scale = 1000
        
        
        self.spike_base_path = os.path.join(root_path, "spiking", split)  #/home/Datadisk/spikedata5622/spiking-2022/          
        self.depth_base_path = os.path.join(root_path, "depth", split)      
        self.spike_sequences = os.listdir(self.spike_base_path) # e.g. 2018-10-16-11-43-02
        
        self.rec_base_path = os.path.join(root_path, "reconstruction", split)


        root_path2 = "/home/Datadisk/spikedata5622/spiking-2022/indoor_real_spike/indoor_/"
        self.spike_base_path2 = os.path.join(root_path2, "spiking", split)  #/home/Datadisk/spikedata5622/spiking-2022/          
        self.depth_base_path2 = os.path.join(root_path2, "depth", split)      
        self.spike_sequences2 = os.listdir(self.spike_base_path2) # e.g. 2018-10-16-11-43-02        
        
        self.rec_base_path2 = os.path.join(root_path, "reconstruction", split)
        
        
        self.depth_list = []
        self.spike_list_left = []
        self.spike_list_right = []
        self.rec_list_left = []
        self.rec_list_right = []
        
        
        for s in self.spike_sequences:
            s_seq_path = os.path.join(self.spike_base_path, s)
           
            pics_left = os.listdir(os.path.join(s_seq_path, "left"))
            for pic in pics_left:
                if pic.endswith("npy"):
                #print(os.path.join(s_seq_path, "left", pic))
                    self.spike_list_left.append(os.path.join(s_seq_path, "left", pic))
                    depth_left = os.path.join(self.depth_base_path, s, pic.replace("dat", "pfm"))
                    self.depth_list.append(depth_left)
    
                    self.spike_list_right.append(os.path.join(s_seq_path, "right", pic))

        self.spike_base_path2 = os.path.join(root_path2, "spiking", split)
        for s in self.spike_sequences2:
            s_seq_path = os.path.join(self.spike_base_path2, s)
           
            pics_left = os.listdir(os.path.join(s_seq_path, "left"))
            for pic in pics_left:
                if pic.endswith("npy"):
                #print(os.path.join(s_seq_path, "left", pic))
                    self.spike_list_left.append(os.path.join(s_seq_path, "left", pic))
                    depth_left = os.path.join(self.depth_base_path2, s, pic.replace("dat", "pfm"))
                    self.depth_list.append(depth_left)
    
                    self.spike_list_right.append(os.path.join(s_seq_path, "right", pic))

        self.depth_list.sort()   
        self.spike_list_left.sort() 
        self.spike_list_right.sort()
        #self.rgb_list_left.sort()
        #self.rgb_list_right.sort()
        
        
        self.split = split
        
    def __len__(self):
        return len(self.spike_list_left)

    def __getitem__(self, idx):

        
        spike_path_left = self.spike_list_left[idx]
        spike_path_right = spike_path_left.replace("left","right")
        print(spike_path_right)

        if self.split == "train":
            this_sequence = spike_path_left[75:78]  
            this_order = spike_path_left[85: -4]
        elif self.split == "val":
            this_sequence = spike_path_left[73:76]  
            this_order = spike_path_left[83: -4]            
        #print(this_sequence, this_order)
        
        
        
        #depth_gt_path = os.path.join(self.depth_base_path, this_sequence, this_order) + "."
        depth_gt_path = self.depth_list[idx]
        #depth_gt = np.expand_dims(self.get_gt_depth_maps(depth_gt_path), 0) 
        
        #print(depth_gt_path)
        depth_gt = np.expand_dims(self.get_gt_disparity(depth_gt_path.replace(".npy",".pfm")), 0) 
       
        depth_gt = torch.FloatTensor(depth_gt)
        
        
        
        
        '''
        rgb_left = np.array(Image.open(self.rgb_list_left[idx])).transpose(2,0,1)
        rgb_right = np.array(Image.open(self.rgb_list_right[idx])).transpose(2,0,1)
        '''
        '''       
        rgb_left = torch.FloatTensor(rgb_left)
        rgb_right = torch.FloatTensor(rgb_right)
        '''

        disp_gt = 20 * depth_gt
        disp_gt = disp_gt.int()
        disp_gt[disp_gt >200] = 200
        disp_gt[disp_gt <=0 ] = -1
        disp_gt = 1 / disp_gt
        disp_gt[disp_gt <=0 ] = 0
        
        
        #depth_gt[depth_gt > 100] = 100.0
        
        
        #print(torch.min(depth_gt), torch.max(depth_gt))
        #depth_gt = depth_gt / 20 #1/depth_gt #/ 100.0
        depth_gt = depth_gt / 20.0
       
        npy_path = os.path.join(self.spike_base_path, this_sequence, "spike/r50/{}.npy".format(this_order)) #spike_ 
        
        spike_mat_left = self.load_np(spike_path_left)#self.analysedat(dat_path)
        #spike_mat_left = self.analysedat(spike_path_left)
        spike_mat_left = 2*spike_mat_left - 1
        spike_mat_right = self.load_np(spike_path_right)#self.analysedat(dat_path)
        #spike_mat_right = self.analysedat(spike_path_right)
        spike_mat_right = 2*spike_mat_right - 1

        
        spike_left = torch.FloatTensor(spike_mat_left)
        spike_right =  torch.FloatTensor(spike_mat_right)

        scale = transforms.Compose([
         transforms.Resize([256, 512]),
         ])  

        sample = {}
        sample["left"] = scale(spike_left)
        sample["right"] = scale(spike_right)
        sample["depth"] = depth_gt
        sample["disparity"] = disp_gt.squeeze(0)
        #sample["rgb_left"] = scale(rgb_left)
        #sample["rgb_right"] = scale(rgb_right)
  
        return sample#spike, depth
        
    def get_gt_depth_maps(self, depth_map_path):
        
        depth_map_gt = np.array(Image.open(depth_map_path), dtype=np.float32) / self.depth_scale
        
        return depth_map_gt

    def get_gt_disparity(self, disp_map_path):

 
        data = rp.readPFM(disp_map_path)[0]
        data = np.ascontiguousarray(data, dtype=np.float32)
        return data
        
        
    def load_np(self, np_path):
        npy = np.load(np_path).astype(np.uint8)
        return torch.FloatTensor(npy)

    def random_crop(self, spike, rgb, depth, height, width):  # load numpy format # height = 352 width = 1216
        '''
        assert spike.shape[1] >= height
        assert spike.shape[2] >= width
        assert spike.shape[1] == depth.shape[1]
        assert spike.shape[2] == depth.shape[2]
        '''
        #print(spike)
        
        x = random.randint(0, spike.shape[2] - width)
        y = random.randint(0, spike.shape[1] - height)
        spike = spike[:, y:y + height, x:x + width]
        depth = depth[:, y:y + height, x:x + width]
        #rgb = rgb[y:y + height, x:x + width, :]
        return spike, depth
        
    def refine(self, mat):
        mat[mat > 100] = 100
        return mat

        


    def list2img(self, list_, path):
        w = self.W
        h = self.H
        
        path = path.replace("dat", "npy")
        
        #name = os.path.join("/home/Datadisk/SpikeSet/SpikeData/", path[-8:-4]) + ".png"
        name = path.replace(".dat", ".npy")
        
        mat = np.array(list_).reshape(h, w)
        img = Image.fromarray(np.uint8(mat))
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
        img = np.array(img, dtype = np.uint8)
        name = name.replace(".png","")
        
        
        
        return img
        #img.save(name)   #/home/Datadisk/spikedata5622/spiking-2022/outdoor_real_spike/spiking/train/0040/left/      
                          #/home/Datadisk/spikenpya5622/spiking-2022/outdoor_real_spike/spiking/train/0000/left/0000.npy   

    
    def list2np(self, sum, list_, index): # right or left
        w = 1242
        h = 375
        if index % 24 == 0:
            sum = np.zeros((24, h, w))
       
        mat = np.array(list_).reshape(24, h, w)
        

        return mat                 





class IMGReal(Dataset):

    def __init__(self, root_path: str, split = "train"):   # train / val
    
        self.H = 250
        self.W = 400
        self.depth_scale = 1000
        
        
        self.spike_base_path = os.path.join(root_path, "spiking", split)  #/home/Datadisk/spikedata5622/spiking-2022/          
        self.depth_base_path = os.path.join(root_path, "depth_trans", split)      
        self.spike_sequences = os.listdir(self.spike_base_path) # e.g. 2018-10-16-11-43-02
        
        self.rec_base_path = os.path.join(root_path, "reconstruction", split)
        
        
        self.depth_list = []
        self.spike_list_left = []
        self.spike_list_right = []
        self.rec_list_left = []
        self.rec_list_right = []
        
        
        for s in self.spike_sequences:
            s_seq_path = os.path.join(self.spike_base_path, s)
           
            pics_left = os.listdir(os.path.join(s_seq_path, "left"))
            for pic in pics_left:
                if pic.endswith("npy"):
                #print(os.path.join(s_seq_path, "left", pic))
                    self.spike_list_left.append(os.path.join(s_seq_path, "left", pic))
                    depth_left = os.path.join(self.depth_base_path, s, pic.replace("npy", "pfm"))
                    
                    img_left = os.path.join(self.rec_base_path, s, "left", pic.replace("npy","png"))
                    #rec_list_left = os.path.join(self.rec_base_path, s, )
                    
                    self.depth_list.append(depth_left)
                    self.rec_list_left.append(img_left)
                    
    
                    self.spike_list_right.append(os.path.join(s_seq_path, "right", pic))
                    self.rec_list_right.append(img_left.replace("left","right"))
    
        self.depth_list.sort()   
        self.spike_list_left.sort() 
        self.spike_list_right.sort()
        self.rec_list_left.sort()
        self.rec_list_right.sort()
        
        
        self.split = split
        
    def __len__(self):
        return len(self.spike_list_left)

    def __getitem__(self, idx):

        
        spike_path_left = self.spike_list_left[idx]
        spike_path_right = spike_path_left.replace("left","right")
        #print(spike_path_right)

        if self.split == "train":
            this_sequence = spike_path_left[75:78]  
            this_order = spike_path_left[85: -4]
        elif self.split == "val":
            this_sequence = spike_path_left[73:76]  
            this_order = spike_path_left[83: -4]            
        #print(this_sequence, this_order)
        
        
        #depth_gt_path = os.path.join(self.depth_base_path, this_sequence, this_order) + "."
        depth_gt_path = self.depth_list[idx]
        #depth_gt = np.expand_dims(self.get_gt_depth_maps(depth_gt_path), 0) 
        
        #print(depth_gt_path)
        depth_gt = np.expand_dims(self.get_gt_disparity(depth_gt_path.replace(".npy",".pfm")), 0) 
       
        depth_gt = torch.FloatTensor(depth_gt)
        
               
        
        
        rgb_left = np.array(Image.open(self.rec_list_left[idx]))#.transpose(2,0,1)
        #print(rgb_left.shape)
        
        rgb_right = np.array(Image.open(self.rec_list_right[idx]))#.transpose(2,0,1)
        
        rgb_left = torch.FloatTensor(rgb_left).unsqueeze(0)
        rgb_right = torch.FloatTensor(rgb_right).unsqueeze(0)
        '''       
        rgb_left = torch.FloatTensor(rgb_left)
        rgb_right = torch.FloatTensor(rgb_right)
        '''

        disp_gt =  depth_gt 
        #disp_gt = disp_gt.int()
        disp_gt[disp_gt >200] = 200
        disp_gt[disp_gt <=0 ] = -1
        disp_gt = 1 / disp_gt
        #disp_gt = disp_gt.int()
        disp_gt[disp_gt <=0 ] = 0
        
        
        depth_gt = depth_gt / 20.0
       
        npy_path = os.path.join(self.spike_base_path, this_sequence, "spike/r50/{}.npy".format(this_order)) #spike_ 
        
        spike_mat_left = self.load_np(spike_path_left)#self.analysedat(dat_path)
        #spike_mat_left = self.analysedat(spike_path_left)
        spike_mat_left = 2*spike_mat_left - 1 #+ 1
        spike_mat_right = self.load_np(spike_path_right)#self.analysedat(dat_path)
        #spike_mat_right = self.analysedat(spike_path_right)
        spike_mat_right = 2*spike_mat_right - 1#- 1

        
        spike_left = torch.FloatTensor(spike_mat_left)
        spike_right =  torch.FloatTensor(spike_mat_right)

        scale = transforms.Compose([
         transforms.Resize([128, 256]),
         ])  

        sample = {}
        sample["left"] = scale(spike_left)
        sample["right"] = scale(spike_right)
        sample["depth"] = depth_gt
        sample["disparity"] = disp_gt.squeeze(0)
        sample["left_img"] = scale(rgb_left)#.unsqueeze(0)
        sample["right_img"] = scale(rgb_right)#.unsqueeze(0)
        
        return sample#spike, depth
        
    def get_gt_depth_maps(self, depth_map_path):
        
        depth_map_gt = np.array(Image.open(depth_map_path), dtype=np.float32) / self.depth_scale
        
        return depth_map_gt

    def get_gt_disparity(self, disp_map_path):

 
        data = rp.readPFM(disp_map_path)[0]
        data = np.ascontiguousarray(data, dtype=np.float32)
        return data
        
        
    def load_np(self, np_path):
        npy = np.load(np_path).astype(np.uint8)
        return torch.FloatTensor(npy)

    def random_crop(self, spike, rgb, depth, height, width):  # load numpy format # height = 352 width = 1216
        
        x = random.randint(0, spike.shape[2] - width)
        y = random.randint(0, spike.shape[1] - height)
        spike = spike[:, y:y + height, x:x + width]
        depth = depth[:, y:y + height, x:x + width]
        #rgb = rgb[y:y + height, x:x + width, :]
        return spike, depth
        
    def refine(self, mat):
        mat[mat > 100] = 100
        return mat

        
    def list2img(self, list_, path):
        w = self.W
        h = self.H
        
        path = path.replace("dat", "npy")
        
        #name = os.path.join("/home/Datadisk/SpikeSet/SpikeData/", path[-8:-4]) + ".png"
        name = path.replace(".dat", ".npy")
        
        mat = np.array(list_).reshape(h, w)
        img = Image.fromarray(np.uint8(mat))
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
        img = np.array(img, dtype = np.uint8)
        name = name.replace(".png","")
        
        
        
        return img
        #img.save(name)   #/home/Datadisk/spikedata5622/spiking-2022/outdoor_real_spike/spiking/train/0040/left/      


    
    def list2np(self, sum, list_, index): # right or left
        w = 1242
        h = 375
        if index % 24 == 0:
            sum = np.zeros((24, h, w))
       
        mat = np.array(list_).reshape(24, h, w)
        

        return mat                 


