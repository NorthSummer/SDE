#
# Dataset and model related tools
#
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

from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize

class SpikeDataset(Dataset):

    def __init__(self, pathr: str, pathl: str, mode:str):
        self.pathr = pathr
        self.pathl = pathl
       
        filesr = os.listdir(self.pathr)
        filesl = os.listdir(self.pathl)
        
        self.filesr = [f for f in filesr if f.endswith('.npz')]
        self.filesl = [f for f in filesl if f.endswith('.npz')]
        
        self.filesr.sort(key = lambda x: int(x[:-4]))
        self.filesl.sort(key = lambda x: int(x[:-4]))
        self.depth_list = os.listdir("/home/Datadisk/spikedata5622/spiking-2022/train/depth/vidar_right")
        
        self.depth_list.sort(key = lambda x: int(x[5:-5]))
       
        self.mode = mode
        
        self.rgb = "/home/Datadisk/SpikeRGB"
        self.rgb_files = os.listdir(self.rgb)
        self.rgb_files.sort()
        
        
    def __len__(self):
        return len(self.filesr)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        pathr = os.path.join(self.pathr, self.filesr[idx])
        pathl = os.path.join(self.pathl, self.filesl[idx])
        seq_r, tag = load_spike_numpy(pathr)
        seq_l, _ = load_spike_numpy(pathl)
        
        if self.mode == "training":
            d_tag = self.depth_list[16+64*idx]
            d_tag = os.path.join("/home/Datadisk/spikedata5622/spiking-2022/train/depth/vidar_right", d_tag)
            #rgb = self.rgb_files[1333*idx]
            #print(d_tag)
        elif self.mode == "test":
            d_tag = self.depth_list[16+320*idx]
            d_tag = os.path.join("/home/Datadisk/spikedata5622/spiking-2022/train/depth/vidar_right", d_tag)            

        file_ = OpenEXR.InputFile(d_tag)
        pt = Imath.PixelType(Imath.PixelType.FLOAT)
        dw = file_.header()['dataWindow']
        size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
        image1 = [Image.frombytes("F", size, file_.channel(c, pt)) for c in "G"]      
        d_image = np.array(image1[0].convert('L'), dtype = np.float32)
        d_image[d_image>=255.0] = 255.0
        
        disp_image = np.array(image1[0].convert('L'), dtype = np.float32)
        disp_image[disp_image>=255.0] = 255.0
        
        #d_image = Image.open(d_tag).convert('L')#cv2.imread(d_tag, cv2.IMREAD_UNCHANGED)
        tag = np.array(d_image, dtype = np.float32)
        tag = tag.transpose(0,1)
        
        tag_disp = np.array(disp_image, dtype = np.float32)
        tag_disp = tag.transpose(0,1)        
        # Random rotate
        #degree = random.randint(0, 3)
        #seq = np.rot90(seq, degree, (1, 2))
        #tag = np.rot90(tag, degree, (0, 1))

        # Random fliplr
        #if random.random() > 0.5:
        #    seq = np.flip(seq, 2)
        #    tag = np.flip(tag, 1)

        seq_r = seq_r.astype(np.float32) #* 2 - 1
        seq_l = seq_l.astype(np.float32)
        
        new_r = np.zeros((32, 648, 1152), dtype = np.float32)  # np.zeros((n, 648, 1152) n为通道数
        new_l = np.zeros((32, 648, 1152), dtype = np.float32)
        
        for i in range(0, 32): ### modified
            new_r[i,:,:] = seq_r[i,:,:]    #seq_r[m,:,:]   m = n / 2
            new_l[i,:,:] = seq_l[i,:,:]
        
        depth_dorn = tag.astype(np.float32)
        depth = (tag.astype(np.float32) / 128.0) - 1.0
        ####depth = tag.astype(np.float32) / 256.0
        #disp = 1.0 / tag  ###_disp ###for stereonet
        disp = 1.0 / tag_disp
        
        #disp = disp * 2.0 - 1.0

        torch_resize1 = Resize([256, 512])       
        torch_resize2 = Resize([385,513]) 
        
        depth = torch.from_numpy(depth)
        depth = depth.unsqueeze(0)
        
        depth = torch_resize1(depth)
        depth_dorn = torch_resize2(torch.from_numpy(depth_dorn).unsqueeze(0))
        
        depth_dorn = depth_dorn.squeeze(0)
        depth = depth.squeeze(0)
        
        
        
        disp = torch.from_numpy(disp)
        disp = disp.unsqueeze(0)
        disp = torch_resize1(disp)
        disp = disp.squeeze(0)
        
        seq_r = torch.FloatTensor(new_r)
        seq_l = torch.FloatTensor(new_l)
        
        seq_ldorn = torch_resize2(seq_l)
        
        seq_r = torch_resize1(seq_r)
        seq_l = torch_resize1(seq_l)
        #if crop
        
        width = 512
        height = 256
        
        space_w = 1152 - 512
        space_h = 648 - 256
        
        start_x = random.randint(0, space_w)
        start_y = random.randint(0, space_h)
        
        consist = seq_r - seq_l
        SID = get_depth_sid(depth, alpha=1.0, beta=80.0)        
        sample = {}
        sample["left"] = seq_l
        sample["right"] = seq_r
        sample["depth"] = depth
        sample["disparity"] = disp
        #sample["consist"] = consist[16].squeeze(1)
        
        sample["seq_ldorn"] = seq_ldorn
        sample["depth_dorn"] = depth_dorn#/256
        sample["cas"] = torch.cat((seq_l.unsqueeze(1), seq_r.unsqueeze(1)),1)
        dd = torch_resize1(torch.from_numpy(tag).unsqueeze(0)).squeeze(0)
        #N,H,W = dd.size()
        

        sample["SID"] = dd#gt_onehot

        return sample

    def resize_transform(self, x):
        trans = Compose([
        Resize((256, 512)),
        ToTensor(),
            ])
        return trans(x)


class SpikeDS(Dataset):      

    def __init__(self, pathr: str, pathl: str, mode:str):
        self.pathr = pathr
        self.pathl = pathl
       
        filesr = os.listdir(self.pathr)
        filesl = os.listdir(self.pathl)
        
        self.filesr = [f for f in filesr if f.endswith('.npz')]
        self.filesl = [f for f in filesl if f.endswith('.npz')]
        
        #self.filesr.sort(key = lambda x: int(x[:-4]))
        #self.filesl.sort(key = lambda x: int(x[:-4]))
        self.depth_list = os.listdir("/home/Datadisk/spikedata5622/spiking-2022/train/depth/vidar_right")
        self.depth_list.sort(key = lambda x: int(x[-9:-5]))
        
        self.mode = mode
        
        
    def __len__(self):
        return len(self.filesr)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        pathr = os.path.join(self.pathr, self.filesr[idx])
        pathl = pathr.replace("rightspike", "leftspike")
        pathd = pathr.replace("rightspike", "depth").replace(".npz", ".png").replace("\n", "").replace("spike/","")
        #pathl = os.path.join(self.pathl, self.filesl[idx])
        seq_r, _ = load_spike_numpy(pathr)
        
        seq_l, _ = load_spike_numpy(pathl)
        
        d_tag = Image.open(pathd)
        depth_gt = np.asarray(d_tag, dtype=np.float32)
        depth_gt = depth_gt / 256.0
        
        
        depth_gt[depth_gt >= 80.0] = 80.0
               
        #d_image = Image.open(d_tag).convert('L')#cv2.imread(d_tag, cv2.IMREAD_UNCHANGED)
        tag = np.array(depth_gt, dtype = np.float32)
        tag = tag.transpose(0,1)
        '''
        tag_disp = np.array(disp_image, dtype = np.float32)
        tag_disp = tag.transpose(0,1)        
        '''

        seq_r = seq_r.astype(np.float32) #* 2 - 1
        seq_l = seq_l.astype(np.float32)
        
        new_r = np.zeros((32, 400, 879), dtype = np.float32)  # np.zeros((n, 648, 1152) n为通道数
        new_l = np.zeros((32, 400, 879), dtype = np.float32)
        
        for i in range(0, 32): ### modified
            new_r[i,:,:] = seq_r[i,:,:]    #seq_r[m,:,:]   m = n / 2
            new_l[i,:,:] = seq_l[i,:,:]
        

        depth = (tag.astype(np.float32) / 40.0) - 1.0
        #disp = 1.0 / tag  ###_disp ###for stereonet
        disp = depth   ################# need modify

        torch_resize1 = Resize([256, 512])       
        torch_resize2 = Resize([385,513]) 
        
        depth = torch.from_numpy(depth)
        depth = depth.unsqueeze(0)
        
        depth = torch_resize1(depth)

        depth = depth.squeeze(0)
        
        
        
        disp = torch.from_numpy(disp)
        disp = disp.unsqueeze(0)
        disp = torch_resize1(disp)
        disp = disp.squeeze(0)
        
        seq_r = torch.FloatTensor(new_r)
        seq_l = torch.FloatTensor(new_l)
        
        seq_ldorn = torch_resize2(seq_l)
        
        seq_r = torch_resize1(seq_r)
        seq_l = torch_resize1(seq_l)
        #if crop
        
        width = 512
        height = 256
        
        space_w = 1152 - 512
        space_h = 648 - 256
        
        start_x = random.randint(0, space_w)
        start_y = random.randint(0, space_h)
        
        consist = seq_r - seq_l
        SID = get_depth_sid(depth, alpha=1.0, beta=80.0)        
        sample = {}
        sample["left"] = seq_l
        sample["right"] = seq_r
        sample["depth"] = depth
        sample["disparity"] = disp
        sample["SID"] = depth
        sample["depth_dorn"] = depth


        return sample

    def resize_transform(self, x):
        trans = Compose([
        Resize((256, 512)),
        ToTensor(),
            ])
        return trans(x)      
       
class SpikeRGBDataset(Dataset):

    def __init__(self, path_rgb: str, path_spike_left: str, path_spike_right:str, mode:str):
        self.path_rgb = path_rgb
        self.path_spike_left = path_spike_left
        self.path_spike_right = path_spike_right
        
        files_rgb = os.listdir(self.path_rgb)
        files_spike_left = os.listdir(self.path_spike_left)
        files_spike_right = os.listdir(self.path_spike_right)
        
        self.files_rgb = [f for f in files_rgb if f.endswith('.png')]
        self.files_spike_left = [f for f in files_spike_left if f.endswith('.npz')]
        self.files_spike_right = [f for f in files_spike_right if f.endswith('.npz')]
        
        self.files_rgb.sort()
        self.files_spike_left.sort()
        self.files_spike_right.sort()
        
        self.depth_list = os.listdir("/home/Datadisk/spikedata5622/spiking-2022/train/depth/vidar_right")
        self.depth_list.sort(key = lambda x: int(x[-9:-5]))
        
        self.mode = mode
        
        
    def __len__(self):
        return int(len(self.files_rgb)/3000)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        path_rgb = os.path.join(self.path_rgb, self.files_rgb[idx])
        path_spike_left = os.path.join(self.path_spike_left, self.files_spike_left[idx])
        #seq_rgb, _ = load_spike_numpy(path_rgb) # change
        seq_spike_left, tag = load_spike_numpy(path_spike_left)
        seq_spike_right, tag = load_spike_numpy(path_spike_left)
        
        ##load rgb
       
        
        if self.mode == "training":
            d_tag = self.depth_list[1333*idx]
            d_tag = os.path.join("/home/Datadisk/spikedata5622/spiking-2022/train/depth/vidar_right", d_tag)
            rgb_tag = os.path.join(self.path_rgb, self.files_rgb[idx*1333]) 
            
            
        elif self.mode == "test":
            d_tag = self.depth_list[30000+1333*idx]
            d_tag = os.path.join("/home/Datadisk/spikedata5622/spiking-2022/train/depth/vidar_right", d_tag)   
            #rgb_tag = self.files_rgb[30000 + idx*1333]         
            rgb_tag = os.path.join(self.path_rgb, self.files_rgb[30000 + idx*1333]) 


        rgb = np.array(Image.open(rgb_tag).convert('RGB'), dtype = np.float32)
        rgb = rgb.transpose(2,0,1)
        rgb = torch.from_numpy(rgb)
        
        file_ = OpenEXR.InputFile(d_tag)
        pt = Imath.PixelType(Imath.PixelType.FLOAT)
        dw = file_.header()['dataWindow']
        size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
        image1 = [Image.frombytes("F", size, file_.channel(c, pt)) for c in "G"]      
        d_image = np.array(image1[0].convert('L'), dtype = np.float32)
        d_image[d_image>=255.0] = 255.0
        
        disp_image = np.array(image1[0].convert('L'), dtype = np.float32)
        disp_image[disp_image>=255.0] = 255.0
        
        #d_image = Image.open(d_tag).convert('L')#cv2.imread(d_tag, cv2.IMREAD_UNCHANGED)
        tag = np.array(d_image, dtype = np.float32)
        tag = tag.transpose(0,1)


        #seq_rgb = seq_r.astype(np.float32) #* 2 - 1
        seq_spike = seq_spike_left.astype(np.float32)

        new_spike = np.zeros((32, 648, 1152), dtype = np.float32)
        
        for i in range(0, 31):
            #new_r[i,:,:] = seq_r[16,:,:]
            new_spike[i,:,:] = seq_spike[16,:,:]
        
        #depth = (tag.astype(np.float32) / 128.0) - 1.0
        depth = tag.astype(np.float32) / 255.0
        disp = 1.0 / tag ###for stereonet

        

        torch_resize1 = Resize([256, 512])        
        
        rgb = torch_resize1(rgb)
        
        depth = torch.from_numpy(depth)
        depth = depth.unsqueeze(0)
        depth = torch_resize1(depth)
        depth = depth.squeeze(0)
        
        disp = torch.from_numpy(disp)
        disp = disp.unsqueeze(0)
        disp = torch_resize1(disp)
        disp = disp.squeeze(0)
        
        
        seq_spike = torch.FloatTensor(new_spike)
        seq_spike = torch_resize1(seq_spike)
        #if crop
        
        width = 512
        height = 256
        
        space_w = 1152 - 512
        space_h = 648 - 256
        
        start_x = random.randint(0, space_w)
        start_y = random.randint(0, space_h)

        SID = get_depth_sid(depth, alpha=1.0, beta=255.0)
        #print(SID)
        sample = {}
        sample["spike"] = seq_spike
        sample["depth"] = depth
        sample["disparity"] = disp
        sample["rgb"] = rgb
        sample["left"] = rgb
        sample["right"] = rgb
        

        

        return sample


    def resize_transform(self, x):
        trans = Compose([
        Resize((256, 512)),
        ToTensor(),
            ])
        return trans(x)



class SpikeTN(Dataset):

    def __init__(self, pathr: str, pathl: str, mode:str):
        self.pathr = pathr
        self.pathl = pathl
       
        filesr = os.listdir(self.pathr)
        filesl = os.listdir(self.pathl)
        
        self.filesr = [f for f in filesr if f.endswith('.npy')]
        self.filesl = [f for f in filesl if f.endswith('.npy')]
        
        self.rgbr = [f for f in filesr if f.endswith('.png')]
        #print(self.rgbr)
        self.rgbr.sort(key = lambda x: int(x[0:-4]))
        
        #print(self.filesr)
        self.filesr.sort(key = lambda x: int(x[5:-4]))
        self.filesl.sort(key = lambda x: int(x[5:-4]))
        #print(self.filesr)
        self.depth_list = [f for f in filesr if f.endswith('.exr')]#os.listdir("/home/Datadisk/spikedata5622/spiking-2022/train/depth/vidar_right")
        #print(self.depth_list)
        self.depth_list.sort(key = lambda x: int(x[5:-4]))
        
        self.mode = mode
        
        
    def __len__(self):
        if self.mode == "train":
            return int(1.5*len(self.filesr)/(2*64)-1)
        elif self.mode == "test":
            return int(0.5*len(self.filesr)/(2*64)-1)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        
        #seq_r, tag = load_spike_numpy(pathr)
        #seq_l, _ = load_spike_numpy(pathl)
        
 #400*250
        
        new_seq_r = np.zeros((32, 250, 400), dtype = np.float32)
        new_seq_l = np.zeros((32, 250, 400), dtype = np.float32)

                
        if self.mode == "train":
        
           
            base = 0
            pathr = os.path.join(self.pathr, self.filesr[base + 64*idx])
            pathl = os.path.join(self.pathl, self.filesl[base + 64*idx])
            seq_r = np.load(pathr)
            seq_l = np.load(pathl) 
            new_seq_r[0,:,:] = seq_r
            new_seq_l[0,:,:] = seq_l  
            d_tag = self.depth_list[base + 64*idx]

            d_tag = os.path.join("/home/Datadisk/spikedata5622/spiking-2022/SpikeTunel/road/road/right", d_tag)
            #print(d_tag)
            #print(pathr)
            
            for i in range (1, 32):
                pathr_n = os.path.join(self.pathr, self.filesr[base + 64*idx + i])
                pathl_n = os.path.join(self.pathl, self.filesl[base + 64*idx + i])
                seqr_n = np.load(pathr_n)
                seql_n = np.load(pathl_n)     
                new_seq_r[i,:,:] = seqr_n
                new_seq_l[i,:,:] = seql_n      
        
        elif self.mode == "test":
            base = 75000
            pathr = os.path.join(self.pathr, self.filesr[base + 64*idx])
            pathl = os.path.join(self.pathl, self.filesl[base + 64*idx])
            seq_r = np.load(pathr)
            seq_l = np.load(pathl) 
            new_seq_r[0,:,:] = seq_r
            new_seq_l[0,:,:] = seq_l  
            d_tag = self.depth_list[base + 64*idx]
            #print(d_tag)
            d_tag = os.path.join("/home/Datadisk/spikedata5622/spiking-2022/SpikeTunel/road/road/right", d_tag)  
            rgb = os.path.join(self.pathr, self.rgbr[base + 64*idx])
            print(pathr)
            #rgb = cv2.imread(rgb)#Image.open(rgb)
            for i in range (1, 32):
                pathr_n = os.path.join(self.pathr, self.filesr[base + 64*idx + i])
                pathl_n = os.path.join(self.pathl, self.filesl[base + 64*idx + i])
                
                seqr_n = np.load(pathr_n)
                seql_n = np.load(pathl_n)          
                new_seq_r[i,:,:] = seqr_n
                new_seq_l[i,:,:] = seql_n    
                                       
        '''  
        else:
            d_tag = self.depth_list[64*idx]
            d_tag = os.path.join("/home/Datadisk/spikedata5622/spiking-2022/SpikeTunel/road/right", d_tag)
        '''
        
        
        
        file_ = OpenEXR.InputFile(d_tag)
        pt = Imath.PixelType(Imath.PixelType.FLOAT)
        dw = file_.header()['dataWindow']
        size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
        image1 = [Image.frombytes("F", size, file_.channel(c, pt)) for c in "G"]      
        d_image = np.array(image1[0], dtype = np.float32)
        d_image[d_image>=80.0] = 80.0
        #print(d_image)
        
        disp_image = np.array(image1[0].convert('L'), dtype = np.float32)
        disp_image[disp_image>=80.0] = 80.0
        
        #d_image = Image.open(d_tag).convert('L')#cv2.imread(d_tag, cv2.IMREAD_UNCHANGED)
        tag = np.array(d_image, dtype = np.float32)
        tag = tag.transpose(0,1)
        
        tag_disp = np.array(disp_image, dtype = np.float32)
        tag_disp = tag.transpose(0,1)        


        seq_r = new_seq_r.astype(np.float32) #* 2 - 1
        seq_l = new_seq_l.astype(np.float32)
        new_r = seq_r
        new_l = seq_l

        depth = 1.0 / tag.astype(np.float32) #depth#(tag.astype(np.float32) / 128.0) - 1.0
        #disp = 1.0 / tag  ###_disp ###for stereonet
        disp = 1.0/ tag_disp

        torch_resize1 = Resize([256, 512])    #256   
        torch_resize2 = Resize([385,513]) 
        
        depth = torch.from_numpy(depth)
        depth = depth.unsqueeze(0)
        
        depth = torch_resize1(depth)
       
        
        depth = depth.squeeze(0)
        
        
        
        disp = torch.from_numpy(disp)
        disp = disp.unsqueeze(0)
        disp = torch_resize1(disp)
        disp = disp.squeeze(0)
        
        seq_r = torch.FloatTensor(new_r)#.unsqueeze(0)
        seq_l = torch.FloatTensor(new_l)#.unsqueeze(0)
        

        
        seq_r = torch_resize1(seq_r)
        seq_l = torch_resize1(seq_l)
        #if crop
        
        width = 512
        height = 256
        
        space_w = 1152 - 512
        space_h = 648 - 256
        
        start_x = random.randint(0, space_w)
        start_y = random.randint(0, space_h)
     
        sample = {}
        sample["left"] = seq_l
        sample["right"] = seq_r
        sample["depth"] = depth
        sample["disparity"] = disp
        
        sample["rgb"] = rgb
        #sample["consist"] = consist[16].squeeze(1)
        

        return sample

    def resize_transform(self, x):
        trans = Compose([
        Resize((256, 512)),
        ToTensor(),
            ])
        return trans(x)



class SpikeDataset2(Dataset):

    def __init__(self, pathr: str, pathl: str, mode:str):
        self.pathr = pathr
        self.pathl = pathl
       
        filesr = os.listdir(self.pathr)
        filesl = os.listdir(self.pathl)
        
        self.filesr = [f for f in filesr if f.endswith('.npy')]
        self.filesl = [f for f in filesl if f.endswith('.npy')]
        
        #self.rgbr = [f for f in filesr if f.endswith('.png')]
        #print(self.rgbr)
        #self.rgbr.sort(key = lambda x: int(x[0:-4]))
        
        #print(self.filesr)
        self.filesr.sort(key = lambda x: int(x[0:-4]))
        self.filesl.sort(key = lambda x: int(x[0:-4]))
        #print(self.filesr)
        #print(self.filesr)
        depth = os.listdir("/home/Datadisk/spikedata5622/spiking-2022/train/depth/vidar_right")
        self.depth_list = [f for f in depth if f.endswith('.exr')]#os.listdir("/home/Datadisk/spikedata5622/spiking-2022/train/depth/vidar_right")
        print(len(self.depth_list))
        self.depth_list.sort(key = lambda x: int(x[5:-4]))
        
        self.mode = mode
        
        
    def __len__(self):
        if self.mode == "train":
            return int(0.6*len(self.filesr)-100)
        elif self.mode == "test":
            return int(0.5*len(self.filesr)-100)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        '''
        pathr = os.path.join(self.pathr, self.filesr[idx])
        pathl = os.path.join(self.pathl, self.filesl[idx])
        '''
        #seq_r, tag = load_spike_numpy(pathr)
        #seq_l, _ = load_spike_numpy(pathl)
        
        #seq_r = np.load(pathr)
        #seq_l = np.load(pathl)   #400*250
        
        new_seq_r = np.zeros((32, 648, 1152), dtype = np.float32)
        new_seq_l = np.zeros((32, 648, 1152), dtype = np.float32)
        #new_seq_r[0,:,:] = seq_r
        #new_seq_l[0,:,:] = seq_l
                
        if self.mode == "train":
            base = 0
            d_tag = self.depth_list[base + 32*idx]

            d_tag = os.path.join("/home/Datadisk/spikedata5622/spiking-2022/train/depth/vidar_right", d_tag)
            #print(d_tag)
            #print(pathr)
            pathr_n = os.path.join(self.pathr, self.filesr[base + idx])
            pathl_n = os.path.join(self.pathl, self.filesr[base + idx])
            seqr_n = np.load(pathr_n)
            seql_n = np.load(pathl_n)            
            for i in range (0, 32):
                #pathr_n = os.path.join(self.pathr, self.filesr[base + 64*idx + i])
                #pathl_n = os.path.join(self.pathl, self.filesl[base + 64*idx + i])
               # seqr_n = np.load(pathr_n)
               # seql_n = np.load(pathl_n)     
                new_seq_r[i,:,:] = seqr_n[i,:,:]
                new_seq_l[i,:,:] = seql_n[i,:,:]      
        
        elif self.mode == "test":
            base_d = 30000
            base_s = 900
            d_tag = self.depth_list[base_d + 32*idx]
            #print(d_tag)
            d_tag = os.path.join("/home/Datadisk/spikedata5622/spiking-2022/train/depth/vidar_right", d_tag)  
            #rgb = os.path.join(self.pathr, self.rgbr[base + 64*idx])
            #print(rgb)
            #rgb = cv2.imread(rgb)#Image.open(rgb)
            pathr_n = os.path.join(self.pathr, self.filesr[base_s + idx])
            pathl_n = os.path.join(self.pathl, self.filesl[base_s + idx])  
            
            seqr_n = np.load(pathr_n)
            seql_n = np.load(pathl_n)                       
            for i in range (1, 32):

                seqr_n = np.load(pathr_n)
                seql_n = np.load(pathl_n)  
                        
                new_seq_r[i,:,:] = seqr_n[i,:,:]
                new_seq_l[i,:,:] = seql_n[i,:,:]  
                                       
        '''  
        else:
            d_tag = self.depth_list[64*idx]
            d_tag = os.path.join("/home/Datadisk/spikedata5622/spiking-2022/SpikeTunel/road/right", d_tag)
        '''
        
        
        
        file_ = OpenEXR.InputFile(d_tag)
        pt = Imath.PixelType(Imath.PixelType.FLOAT)
        dw = file_.header()['dataWindow']
        size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
        image1 = [Image.frombytes("F", size, file_.channel(c, pt)) for c in "G"]      
        d_image = np.array(image1[0], dtype = np.float32)
        d_image[d_image>=255.0] = 255.0
        #print(d_image)
        
        disp_image = np.array(image1[0].convert('L'), dtype = np.float32)
        disp_image[disp_image>=255.0] = 255.0
        
        #d_image = Image.open(d_tag).convert('L')#cv2.imread(d_tag, cv2.IMREAD_UNCHANGED)
        tag = np.array(d_image, dtype = np.float32)
        tag = tag.transpose(0,1)
        
        tag_disp = np.array(disp_image, dtype = np.float32)
        tag_disp = tag.transpose(0,1)        


        seq_r = new_seq_r.astype(np.float32) #* 2 - 1
        seq_l = new_seq_l.astype(np.float32)
        new_r = seq_r
        new_l = seq_l

        depth = (tag.astype(np.float32) / 128.0) - 1.0
        #disp = 1.0 / tag  ###_disp ###for stereonet
        disp = 1.0/ tag_disp

        torch_resize1 = Resize([256, 512])    #256   
        torch_resize2 = Resize([385,513]) 
        
        depth = torch.from_numpy(depth)
        depth = depth.unsqueeze(0)
        
        depth = torch_resize1(depth)
       
        
        depth = depth.squeeze(0)
        
        
        
        disp = torch.from_numpy(disp)
        disp = disp.unsqueeze(0)
        disp = torch_resize1(disp)
        disp = disp.squeeze(0)
        
        seq_r = torch.FloatTensor(new_r)#.unsqueeze(0)
        seq_l = torch.FloatTensor(new_l)#.unsqueeze(0)
        

        
        seq_r = torch_resize1(seq_r)
        seq_l = torch_resize1(seq_l)
        #if crop
        
        width = 512
        height = 256
        
        space_w = 1152 - 512
        space_h = 648 - 256
        
        start_x = random.randint(0, space_w)
        start_y = random.randint(0, space_h)
     
        sample = {}
        sample["left"] = seq_l
        sample["right"] = seq_r
        sample["depth"] = depth
        sample["disparity"] = disp
        
        #sample["rgb"] = rgb
        #sample["consist"] = consist[16].squeeze(1)
        

        return sample

    def resize_transform(self, x):
        trans = Compose([
        Resize((256, 512)),
        ToTensor(),
            ])
        return trans(x)







def get_depth_log(depth, alpha, beta):
    alpha_ = torch.FloatTensor([alpha])
    beta_ = torch.FloatTensor([beta])
    K_ = torch.FloatTensor([K])
    t = torch.from_numpy(np.array(t.detach().cpu(), dtype = np.float32))
    t = K_ * torch.log(depth / alpha_) / torch.log(beta_ / alpha_)
    # t = t.int()
    
    return t
        

        
def get_depth_sid(depth_labels, alpha, beta):
    depth_labels = depth_labels.data.cpu()
    alpha_ = torch.FloatTensor([alpha])
    beta_ = torch.FloatTensor([beta])
    #K_ = torch.FloatTensor([K])
    #t = torch.from_numpy(np.array(t.detach().cpu(), dtype = np.float32))
    t = torch.exp(torch.log(alpha_) + torch.log(beta_ / alpha_) * depth_labels )#/ K_) * depth_labels / K_)
    
    return t         
        

def load_spike_numpy(path: str) -> (np.ndarray, np.ndarray):
    '''
    Load a spike sequence with it's tag from prepacked `.npz` file.\n
    The sequence is of shape (`length`, `height`, `width`) and tag of
        shape (`height`, `width`).
    '''
    data = np.load(path,allow_pickle=True)
    seq, tag = data['seq'], data['tag']
    #seq = np.array([(seq[i // 8] >> (i & 7)) & 1 for i in range(length)])
    return seq, tag


def dump_spike_numpy(path: str, seq: np.ndarray, tag: np.ndarray):
    '''
    Store a spike sequence with it's tag to `.npz` file.
    '''
    length = seq.shape[0]
    seq = seq.astype(np.bool)
    seq = np.array([seq[i] << (i & 7) for i in range(length)])
    seq = np.array([np.sum(seq[i: min(i+8, length)], axis=0)
                    for i in range(0, length, 8)]).astype(np.uint8)
    np.savez(path, seq=seq, tag=tag, length=np.array(length))


def load_spike_raw(path: str, width=256, height=256) -> np.ndarray:
    '''
    Load bit-compact raw spike data into an ndarray of shape
        (`frame number`, `height`, `width`).
    '''
    with open(path, 'rb') as f:
        fbytes = f.read()
    fnum = (len(fbytes) * 8) // (width * height)  # number of frames
    frames = np.frombuffer(fbytes, dtype=np.uint8)
    frames = np.array([frames & (1 << i) for i in range(8)])
    frames = frames.astype(np.bool).astype(np.uint8)
    frames = frames.transpose(1, 0).reshape(32, height, width)
    frames = np.flip(frames, 1)
    return frames


def get_latest_version(root: str, model: str) -> int:
    pattern = re.compile('^spikling\\-{}-(\\d{{4}}).pth$'.format(model))
    same = [0]
    for f in os.listdir(root):
        match = pattern.match(f)
        if match:
            same.append(int(match.group(1)))
    return max(same)


def online_generate(model: nn.Module, seq: np.ndarray,
                    device: torch.device, path: str):
    height, width = seq.shape[1:]
    seq = seq[np.newaxis, :, :, :] * 2.0 - 1
    seq = torch.FloatTensor(seq).to(device)
    with torch.no_grad():
        img = model(seq)
    img = np.array(img.to(torch.device('cpu')))
    img = img.reshape(height, width) * 128 + 127
    img = (img * (img >= 0)).astype(np.uint8)
    Image.fromarray(img).save(path)


def online_eval(model: nn.Module, device: torch.device, epoch: int):
    simu_set = ['data/ac0009.npz', 'data/ac0081.npz',
                'eval/ac0071.npz', 'eval/mm0071.npz']
    real_set = ['100kmcar.dat', 'disk-pku_short.dat',
                'number-rotation_short.dat', 'operacut.dat']
    for i in real_set:
        break
        seq = load_spike_raw(os.path.join('eval', i))
        seq = seq[seq.shape[0]//2-16:seq.shape[0]//2+16]
        online_generate(model, seq, device,
                        '{:04d}-{}.png'.format(epoch, i))
    for i, j in enumerate(simu_set):
        seq, _ = load_spike_numpy(j)
        online_generate(model, seq, device,
                        '{:04d}-{}.png'.format(epoch, i))
