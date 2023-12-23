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
from torchvision.transforms import Resize
import random



class JYSpikeDataset(Dataset):
    '''
    (`spike_sequence`, `ground_truth`) pairs.\n
    Each value has been normalized to (`-1`, `1`), and each
        tensor is of shape (`channel`, `height`, `width`).
    '''

    def __init__(self, path: str):
        self.path = path

        
        self.files = []
        self.depth_files = []
        #load spike frames
        self.path = self.path
        
        self.sequences = os.listdir(self.path)
        print(self.sequences)
        self.sequences.sort(key = lambda x: int(x[-2:-1]))
        for f_ in self.sequences:
            final_file_list = []
            final_depth_list = []
            
            seq_path = os.path.join(self.path, f_)
            seq_path += "/spike/r128" 
            
            files = os.listdir(seq_path)
            
            file_list = [f for f in files if f.endswith('.npz')]    
            for one_file in file_list:
                one_file = os.path.join(seq_path, one_file)
                final_file_list.append(one_file)
           
            final_file_list.sort(key = lambda x: int(x[-9:-4])) #sort spiking sequences
            
            depth_path = seq_path.replace("spike/","depth/")
            depth_path = depth_path.replace("r128","frames")
            self.depth_list = os.listdir(depth_path)#.sort(key = lambda x: int(x[-9:-4]))
            
            for one_depth in self.depth_list:
                one_depth = os.path.join(depth_path, one_depth)
                final_depth_list.append(one_depth)
            
            final_depth_list.sort(key = lambda x: int(x[-9:-4]))
            
            self.files += final_file_list
            self.depth_files += final_depth_list
        
        
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        seq_path = self.files[idx]#os.path.join(self.path, self.files[idx])    
        seq = load_spike_numpy(seq_path) 
        
        d_tag = self.depth_files[idx]

        #d_tag = os.path.join("/home/Datadisk/spikedata5622/left-100/", d_tag)
        #d_tag = "/home/Datadisk/spikedata5622/left-100/"+ d_tag
        raw = Image.open(d_tag)
        '''
        file_ = OpenEXR.InputFile(d_tag)
        pt = Imath.PixelType(Imath.PixelType.FLOAT)
        dw = file_.header()['dataWindow']
        size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
        image1 = [Image.frombytes("F", size, file_.channel(c, pt)) for c in "G"]
        '''
        #d_image = np.array(image1[0].convert('L'), dtype = np.float32)
        d_image = np.array(image1[0], dtype = np.float32) #.convert('L')
        
        #d_image = Image.open(d_tag).convert('L')#cv2.imread(d_tag, cv2.IMREAD_UNCHANGED)
        tag = np.array(d_image, dtype = np.float32)
        tag = tag.transpose(0,1)

        torch_resize1 = Resize([256, 512])    
        
        seq = seq.astype(np.float32) #* 2 - 1
        depth = (tag.astype(np.float32) / 128.0) - 1.0 
        seq = torch.FloatTensor(seq)
        seq = torch_resize1(seq)

        depth = torch.from_numpy(depth)
        depth = depth.unsqueeze(0)
        depth = torch_resize1(depth)
        depth = depth.squeeze(0)
        
        
        

        tag = torch.FloatTensor(tag[np.newaxis, :, :])
        sample = {}
        sample["left"] = seq
        sample["right"] = seq
        sample["depth"] = depth
        sample["disparity"] = 1 / depth
        return sample



class SpikeTN(Dataset):

    def __init__(self, pathr: str, pathl: str, mode:str):
        self.pathr = pathr
        self.pathl = pathl
       
        filesr = os.listdir(self.pathr)
        filesl = os.listdir(self.pathl)
        
        self.filesr = [f for f in filesr if f.endswith('.npy')]
        self.filesl = [f for f in filesl if f.endswith('.npy')]
        
        self.filesr.sort(key = lambda x: int(x[5:-4]))
        self.filesl.sort(key = lambda x: int(x[5:-4]))
        #print(self.filesr)
        self.depth_list = [f for f in filesr if f.endswith('.exr')]#os.listdir("/home/Datadisk/spikedata5622/spiking-2022/train/depth/vidar_right")
        #print(self.depth_list)
        self.depth_list.sort(key = lambda x: int(x[5:-4]))
        
        self.mode = mode
        
        
    def __len__(self):
        return int(len(self.filesr)/(2*64)-1)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        pathr = os.path.join(self.pathr, self.filesr[64*idx])
        pathl = os.path.join(self.pathl, self.filesl[64*idx])

        
        seq_r = np.load(pathr)
        seq_l = np.load(pathl)   #400*250
        
        new_seq_r = np.zeros((32, 250, 400), dtype = np.float32)
        new_seq_l = np.zeros((32, 250, 400), dtype = np.float32)
        new_seq_r[0,:,:] = seq_r
        new_seq_l[0,:,:] = seq_l
                
        if self.mode == "train":
            base = 0
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
            base = 50000
            d_tag = self.depth_list[base + 64*idx]
            d_tag = os.path.join("/home/Datadisk/spikedata5622/spiking-2022/SpikeTunel/road/road/right", d_tag)  

            for i in range (1, 32):
                pathr_n = os.path.join(self.pathr, self.filesr[base + 64*idx + i])
                pathl_n = os.path.join(self.pathl, self.filesl[base + 64*idx + i])
                seqr_n = np.load(pathr_n)
                seql_n = np.load(pathl_n)          
                new_seq_r[i,:,:] = seqr_n
                new_seq_l[i,:,:] = seql_n                           

        
        
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


        return sample

    def resize_transform(self, x):
        trans = Compose([
        Resize((256, 512)),
        ToTensor(),
            ])
        return trans(x)


def load_spike_numpy(path: str) -> (np.ndarray, np.ndarray):

    data = np.load(path)
    seq = data['seq']#, data['tag'], int(data['length'])
    #seq = np.array([(seq[i // 8] >> (i & 7)) & 1 for i in range(length)])
    return seq


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

if __name__ == "__main__":
    set = SpikeDataset(path="/home/Datadisk/spikedata5622/spiking-2022/JYsplit/train/")
    a, b = set[50]

    #print(a,b)
    print(len(set.files), len(set.depth_files))
