import os
import re
import numpy as np
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import struct
from torchvision.transforms import transforms

import cv2
import OpenEXR
import Imath
import pytorch_ssim

from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize

class SpikeDrivingStereo(Dataset):

    def __init__(self, spike_base_path: str, depth_base_path: str, split = "train"):   # train / val
    
        self.H = 400
        self.W = 881
        self.depth_scale = 256
        
        
        self.spike_base_path = os.path.join(spike_base_path, split)  #/home/Datadisk/spikedata5622/spiking-2022/
          
        self.depth_base_path = depth_base_path #/home/lijianing/drivingstereo_train/train/
        
        self.spike_sequences = os.listdir(self.spike_base_path) # e.g. 2018-10-16-11-43-02
        
        self.depth_list = []
        self.spike_list_left = []
        self.spike_list_right = []
        self.rgb_list_left = []
        self.rgb_list_right = []
        self.rgb_base_path = "/home/Datadisk/spikedata5622/spiking-2022/XVFI-main/DS/"
        
        for s in self.spike_sequences:
            s_seq_path = os.path.join(self.spike_base_path, s)
           
            pics_left = os.listdir(os.path.join(s_seq_path, "Camera_0/spike/r50"))
            for pic in pics_left:
                self.depth_list.append(os.path.join(self.depth_base_path, s, pic))
                self.spike_list_left.append(os.path.join(s_seq_path, "Camera_0/spike/r50", pic))
                self.spike_list_right.append(os.path.join(s_seq_path, "Camera_1/spike/r50", pic))
                
                if split == "train":
                    left_dir = s + "-left"
                    right_dir = s + "-right"
                elif split == "val":
                    left_dir = s + "-left-val"
                    right_dir = s + "-right-val"                    
                    
                self.rgb_list_left.append(os.path.join(self.rgb_base_path, left_dir, pic.replace("npy","jpg")))
                self.rgb_list_right.append(os.path.join(self.rgb_base_path, right_dir, pic.replace("npy","jpg")))                                                                                                
        
        self.depth_list.sort()   
        self.spike_list_left.sort() 
        self.spike_list_right.sort()
        self.rgb_list_left.sort()
        self.rgb_list_right.sort()
        
        
        self.split = split
        
    def __len__(self):
        return len(self.spike_list_left)

    def __getitem__(self, idx):
        
        #depth_gt_path = self.depth_list[idx] #/home/lijianing/drivingstereo_depth/train/2018-10-16-11-43-02/
        #depth_gt = np.expand_dims(self.get_gt_depth_maps(depth_gt_path), 0)         #transpose(2, 1, 0)
        #
        
        spike_path_left = self.spike_list_left[idx]
        spike_path_right = spike_path_left.replace("Camera_0","Camera_1")
        #npy_path = os.path.join(self.spike_base_path, )
        
        #depth_gt_path = self.depth_list[idx] 
        #depth_gt = np.expand_dims(self.get_gt_depth_maps(depth_gt_path), 0)         #transpose(2, 1, 0)
        if self.split == "train":
            this_sequence = spike_path_left[56:75]   ##这个地方要改，第一个切片必须是正向的，第二个切片一个是正向一个是负项的，可以把spike——path print出来放到txt里用鼠标看位置lol
            this_order = spike_path_left[95: -4]
        elif self.split == "val":
            this_sequence = spike_path_left[54:73]   ##这个地方要改，第一个切片必须是正向的，第二个切片一个是正向一个是负项的，可以把spike——path print出来放到txt里用鼠标看位置lol
            this_order = spike_path_left[93: -4]            
        #print(this_sequence, this_order)
        
        depth_gt_path = os.path.join(self.depth_base_path, this_sequence, this_order) + ".png"
        depth_gt = np.expand_dims(self.get_gt_depth_maps(depth_gt_path), 0) 
        
        
        
        depth_gt = torch.FloatTensor(depth_gt)
        
        rgb_left = np.array(Image.open(self.rgb_list_left[idx]).convert('L'))#.transpose(2,0,1).convert('L')
        rgb_right = np.array(Image.open(self.rgb_list_right[idx]).convert('L'))#.transpose(2,0,1).convert('L')
        #print(depth_gt_path, self.rgb_list_left[idx], self.rgb_list_right[idx])

                
        rgb_left = torch.FloatTensor(rgb_left).unsqueeze(0)
        rgb_right = torch.FloatTensor(rgb_right).unsqueeze(0)
        #print(rgb_right.size())
        
        #print(torch.min(depth_gt), torch.max(depth_gt), torch.sum(depth_gt==0), torch.sum(depth_gt<100))        
        scale = transforms.Compose([
         transforms.Resize([256, 512]),
         ])  
         
        #depth_gt = scale(depth_gt)
         
        disp_gt = depth_gt
        disp_gt[disp_gt >100] = 100
        disp_gt[disp_gt <=0 ] = -1
        disp_gt = 1 / disp_gt
        disp_gt[disp_gt <=0 ] = 0
        
        
        depth_gt[depth_gt > 100] = 100.0
        
        depth_gt = 1/depth_gt #/ 100.0
        #dat_path = os.path.join(self.spike_base_path, this_sequence, "spike/r32/spike_{}.npy".format(this_order))
        npy_path = os.path.join(self.spike_base_path, this_sequence, "spike/r50/{}.npy".format(this_order)) #spike_ 
        
        #rgb_path = os.path.join(self.rgb_base_path, this_sequence, this_sequence[0:9], this_sequence, "image_03", "data/{}.png".format(this_order))
        
        
        spike_mat_left = self.load_np(spike_path_left)#self.analysedat(dat_path)
        spike_mat_left = 254*spike_mat_left + 1
        spike_mat_right = self.load_np(spike_path_right)#self.analysedat(dat_path)
        spike_mat_right = 254*spike_mat_right + 1
                
        #spike, depth = self.random_crop(spike_mat, spike_mat, depth_gt, 384, 864)
        
        
        spike_left = torch.FloatTensor(spike_mat_left)
        spike_right =  torch.FloatTensor(spike_mat_right)
        
        #depth = torch.FloatTensor(depth_gt)
        #disp = torch.FloatTensor(disp_gt)
        
        
        #print(spike_left.size())
          
        #print(depth_gt)
        sample = {}
        sample["left"] = scale(spike_left)
        sample["right"] = scale(spike_right)
        sample["depth"] = depth_gt.squeeze(0)
        sample["disparity"] = disp_gt.squeeze(0)
        sample["rgb_left"] = scale(rgb_left)
        sample["rgb_right"] = scale(rgb_right)
  
        return sample#spike, depth
        
    def get_gt_depth_maps(self, depth_map_path):
        
        depth_map_gt = np.array(Image.open(depth_map_path), dtype=np.float32) / self.depth_scale

        
        #print(depth_map_gt.shape)
        
        return depth_map_gt

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
        c = 1242 # 400
        r = 375 # 250
        frame = 0
        frame_tot = 101
        stop_flag = 0
        pos = 0
        ff = []
        count = 0
  
        sum = np.zeros((32, 375, 1242))
        binfile = open(dat_file_path, 'rb')
        #print(dat_file_path)
        while(1):
            
            a = binfile.read(1)
            #if not a:
            #    break
            real_a = struct.unpack('b', a)[0]

            for i in range(8):
                pan = (real_a & (1 << (7 - i)))
                if pan !=0:
                    pan = 1
                ff.append(pan)
                
                pos += 1
            
            if pos >= 24 * r * c: 
                pos = 0
                        #print(len(ff))
                sum = self.list2np(sum, ff, count)
                print(len(ff))
                break

            

    
    def list2np(self, sum, list_, index): # right or left
        w = 1242
        h = 375
        if index % 24 == 0:
            sum = np.zeros((24, h, w))
       
        mat = np.array(list_).reshape(24, h, w)
        
        '''
        idx = index - 32*(index//32)
        sum[idx, :, :] = mat
        
        if idx == 31:
            result = sum.astype(np.uint8)
            #name = os.path.join(os.path.join("/home/AnalogIC1/stu62/", dirc), "np") 
            #name = os.path.join(name, str(index//32)) + ".npy"
            
            #np.save(name, result)
            
        else:
            result = None
        ''' 
        #print(mat)
        return mat                 



class AuxDS(Dataset):

    def __init__(self, spike_base_path: str, rgb_base_path: str, depth_base_path: str, split = "train"):   # train / val
    
        self.H = 384
        self.W = 864
        self.depth_scale = 256
        
        self.rgb_base_path = rgb_base_path
        self.spike_base_path = os.path.join(spike_base_path, split)  #/home/Datadisk/spikedata5622/spiking-2022/
        self.spike_base_path2 = os.path.join(spike_base_path, "val")
        self.spike_sequences2 = os.listdir(self.spike_base_path2)
          
        self.depth_base_path = depth_base_path #/home/lijianing/drivingstereo_train/train/
        
        self.spike_sequences = os.listdir(self.spike_base_path) # e.g. 2018-10-16-11-43-02
        
        self.rgb_sequences = os.listdir(rgb_base_path)
        
        self.depth_list = []
        self.spike_list_left = []
        self.spike_list_right = []
        self.rgb_list_left = []
        self.rgb_list_right = []
        
        for s in self.spike_sequences:
            s_seq_path = os.path.join(self.spike_base_path, s)
           
            pics_left = os.listdir(os.path.join(s_seq_path, "Camera_0/spike/r50"))
            for pic in pics_left:
                self.depth_list.append(os.path.join(self.depth_base_path, s, pic))
                self.spike_list_left.append(os.path.join(s_seq_path, "Camera_0/spike/r50", pic))
                self.spike_list_right.append(os.path.join(s_seq_path, "Camera_1/spike/r50", pic))
                
                left_dir = s + "-left"
                right_dir = s + "-right"
                
                self.rgb_list_left.append(os.path.join(self.rgb_base_path, left_dir, pic.replace("npy","jpg")))
                self.rgb_list_right.append(os.path.join(self.rgb_base_path, right_dir, pic.replace("npy","jpg")))
                                                                                                
        for s in self.spike_sequences2:
            s_seq_path = os.path.join(self.spike_base_path2, s)
           
            pics_left = os.listdir(os.path.join(s_seq_path, "Camera_0/spike/r50"))
            for pic in pics_left:
                
                self.spike_list_left.append(os.path.join(s_seq_path, "Camera_0/spike/r50", pic))
                self.spike_list_right.append(os.path.join(s_seq_path, "Camera_1/spike/r50", pic))
                
                left_dir = s + "-left-val"
                right_dir = s + "-right-val"
                
                self.rgb_list_left.append(os.path.join(self.rgb_base_path, left_dir, pic.replace("npy","jpg")))
                self.rgb_list_right.append(os.path.join(self.rgb_base_path, right_dir, pic.replace("npy","jpg")))  
        

        self.split = split
        
    def __len__(self):
        return len(self.spike_list_left)

    def __getitem__(self, idx):
        
        spike_path_left = self.spike_list_left[idx]
        spike_path_right = spike_path_left.replace("Camera_0","Camera_1")
       
        rgb_path_left = self.rgb_list_left[idx]
        rgb_path_right = self.rgb_list_right[idx]
        
        rgb_left = np.array(Image.open(rgb_path_left).convert('L'))
        rgb_right = np.array(Image.open(rgb_path_right).convert('L'))
        
        rgb_left = torch.FloatTensor(rgb_left)
        rgb_right = torch.FloatTensor(rgb_right)
        
        #print(rgb_left.size()) 
            
        #print(rgb_path_left, spike_path_left)
        if self.split == "train":
            this_sequence = spike_path_left[56:75]  
            this_order = spike_path_left[95: -4]
        elif self.split == "val":
            this_sequence = spike_path_left[54:73]   
            this_order = spike_path_left[93: -4]            
        #print(this_sequence, this_order)
        
        depth_gt_path = os.path.join(self.depth_base_path, this_sequence, this_order) + ".png"
        #depth_gt = np.expand_dims(self.get_gt_depth_maps(depth_gt_path), 0) 
        
        
        #print(torch.min(depth_gt), torch.max(depth_gt), torch.sum(depth_gt==0), torch.sum(depth_gt<100))        
        scale = transforms.Compose([
         transforms.Resize([256, 512]),
         ])  
         

        npy_path = os.path.join(self.spike_base_path, this_sequence, "spike/r50/{}.npy".format(this_order)) #spike_ 
        
        #rgb_path = os.path.join(self.rgb_base_path, this_sequence, this_sequence[0:9], this_sequence, "image_03", "data/{}.png".format(this_order))
        
        
        spike_mat_left = self.load_np(spike_path_left)#self.analysedat(dat_path)
        spike_mat_left = 2*spike_mat_left - 1
        spike_mat_right = self.load_np(spike_path_right)#self.analysedat(dat_path)
        spike_mat_right = 2*spike_mat_right - 1
                
        #spike, depth = self.random_crop(spike_mat, spike_mat, depth_gt, 384, 864)
        
        
        spike_left = torch.FloatTensor(spike_mat_left)
        spike_right =  torch.FloatTensor(spike_mat_right)

        sample = {}
        sample["left"] = scale(spike_left)
        sample["right"] = scale(spike_right)
        sample["left_rgb"] = scale(rgb_left.unsqueeze(0))
        sample["right_rgb"] = scale(rgb_right.unsqueeze(0))
  
        return sample#spike, depth
        
    def get_gt_depth_maps(self, depth_map_path):
        
        depth_map_gt = np.array(Image.open(depth_map_path), dtype=np.float32) / self.depth_scale

        
        #print(depth_map_gt.shape)
        
        return depth_map_gt

    def load_np(self, np_path):
        npy = np.load(np_path).astype(np.uint8)
        return torch.FloatTensor(npy)

    def refine(self, mat):
        mat[mat > 100] = 100
        return mat
        
               


if __name__ == "__main__":
    SpikeKITTI(spike_base_path = "/home/Datadisk/spikedata5622/spiking-2022/SpikeKitti/", rgb_base_path ='' , depth_base_path = "/home/lijianing/kitti_depth/train/" , mode = "train")
    Loader = DataLoader(SpikeKITTI, batch_size = 1)