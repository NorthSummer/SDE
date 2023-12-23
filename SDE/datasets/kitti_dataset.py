import os
import random
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from datasets.data_io import get_transform, read_all_lines
from . import flow_transforms
import torchvision
import cv2
import copy
import torch
from torchvision.transforms import transforms

class KITTIDataset(Dataset):
    def __init__(self, datapath, list_filename, training):
        self.datapath = datapath
        self.left_filenames, self.right_filenames, self.disp_filenames = self.load_path(list_filename)
        self.training = training
        if self.training:
            assert self.disp_filenames is not None

    def load_path(self, list_filename):
        lines = read_all_lines(list_filename)
        splits = [line.split() for line in lines]
        left_images = [x[0] for x in splits]
        right_images = [x[1] for x in splits]
        if len(splits[0]) == 2:  # ground truth not available
            return left_images, right_images, None
        else:
            disp_images = [x[2] for x in splits]
            return left_images, right_images, disp_images

    def load_image(self, filename):
        return Image.open(filename).convert('RGB')

    def load_disp(self, filename):
        data = Image.open(filename)
        data = np.array(data, dtype=np.float32) / 256.
        return data

    # def RGB2GRAY(self, img):
    #     imgG = copy.deepcopy(img)
    #     img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    #     imgG[:, :, 0] = img
    #     imgG[:, :, 1] = img
    #     imgG[:, :, 2] = img
    #     return imgG

    def __len__(self):
        return len(self.left_filenames)

    def __getitem__(self, index):
        left_img = self.load_image(os.path.join(self.datapath, self.left_filenames[index]))
        right_img = self.load_image(os.path.join(self.datapath, self.right_filenames[index]))
        # left_img = self.RGB2GRAY(left_img)
        # right_img = self.RGB2GRAY(right_img)



        if self.disp_filenames:  # has disparity ground truth
            disparity = self.load_disp(os.path.join(self.datapath, self.disp_filenames[index]))
        else:
            disparity = None

        if self.training:
            th, tw = 256, 512
            #th, tw = 320, 704
            
            random_brightness = np.random.uniform(0.5, 2.0, 2)
            random_gamma = np.random.uniform(0.8, 1.2, 2)
            random_contrast = np.random.uniform(0.8, 1.2, 2)
            left_img = torchvision.transforms.functional.adjust_brightness(left_img, random_brightness[0])
            left_img = torchvision.transforms.functional.adjust_gamma(left_img, random_gamma[0])
            left_img = torchvision.transforms.functional.adjust_contrast(left_img, random_contrast[0])
            right_img = torchvision.transforms.functional.adjust_brightness(right_img, random_brightness[1])
            right_img = torchvision.transforms.functional.adjust_gamma(right_img, random_gamma[1])
            right_img = torchvision.transforms.functional.adjust_contrast(right_img, random_contrast[1])
            right_img = np.asarray(right_img)
            left_img = np.asarray(left_img)

            # w, h  = left_img.size
            # th, tw = 256, 512
            #
            # x1 = random.randint(0, w - tw)
            # y1 = random.randint(0, h - th)
            #
            # left_img = left_img.crop((x1, y1, x1 + tw, y1 + th))
            # right_img = right_img.crop((x1, y1, x1 + tw, y1 + th))
            # dataL = dataL[y1:y1 + th, x1:x1 + tw]
            # right_img = np.asarray(right_img)
            # left_img = np.asarray(left_img)

            # geometric unsymmetric-augmentation
            angle = 0;
            px = 0
            if np.random.binomial(1, 0.5):
                # angle = 0.1;
                # px = 2
                angle = 0.05
                px = 1
            co_transform = flow_transforms.Compose([
                # flow_transforms.RandomVdisp(angle, px),
                # flow_transforms.Scale(np.random.uniform(self.rand_scale[0], self.rand_scale[1]), order=self.order),
                flow_transforms.RandomCrop((th, tw)),
            ])
            augmented, disparity = co_transform([left_img, right_img], disparity)
            left_img = augmented[0]
            right_img = augmented[1]

            right_img.flags.writeable = True
            if np.random.binomial(1,0.2):
              sx = int(np.random.uniform(35,100))
              sy = int(np.random.uniform(25,75))
              cx = int(np.random.uniform(sx,right_img.shape[0]-sx))
              cy = int(np.random.uniform(sy,right_img.shape[1]-sy))
              right_img[cx-sx:cx+sx,cy-sy:cy+sy] = np.mean(np.mean(right_img,0),0)[np.newaxis,np.newaxis]

            # to tensor, normalize
            disparity = np.ascontiguousarray(disparity, dtype=np.float32)
            processed = get_transform()
            left_img = processed(left_img)
            right_img = processed(right_img)
            depth = 1.0/((disparity+0.00390)*256.0)
            #print(disparity)
            return {"left": left_img,
                    "right": right_img,
                    "disparity": disparity,
                    "depth":depth}
        else:
            w, h = left_img.size

            # normalize
            processed = get_transform()
            left_img = processed(left_img).numpy()
            right_img = processed(right_img).numpy()

            # pad to size 1248x384
            top_pad = 384 - h
            right_pad = 1248 - w
            assert top_pad > 0 and right_pad > 0
            # pad images
            left_img = np.lib.pad(left_img, ((0, 0), (top_pad, 0), (0, right_pad)), mode='constant', constant_values=0)
            right_img = np.lib.pad(right_img, ((0, 0), (top_pad, 0), (0, right_pad)), mode='constant',
                                   constant_values=0)
            # pad disparity gt
            if disparity is not None:
                assert len(disparity.shape) == 2
                disparity = np.lib.pad(disparity, ((top_pad, 0), (0, right_pad)), mode='constant', constant_values=0)
                depth = 1.0/((disparity+0.00390)*256.0)
                     
            if disparity is not None:
                return {"left": left_img,
                        "right": right_img,
                        "disparity": disparity,
                        "depth" : depth,
                        "top_pad": top_pad,  
                        "right_pad": right_pad,
                        "left_filename": self.left_filenames[index],
                        "right_filename": self.right_filenames[index]
                        }
            '''            
            else:
                return {"left": left_img,
                        "right": right_img,
                        "top_pad": top_pad,
                        
                        "right_pad": right_pad,
                        "left_filename": self.left_filenames[index],
                        "right_filename": self.right_filenames[index]}
            '''
            
class SpikeKitti(Dataset):

    def __init__(self, spike_base_path: str, depth_base_path: str, split = "train"):   # train / val
    
        self.H = 384
        self.W = 864
        self.depth_scale = 256
        self.split = split
        
        self.spike_base_path = os.path.join(spike_base_path, split) 
          
        self.depth_base_path = depth_base_path 
        
        self.spike_sequences = os.listdir(self.spike_base_path) 
        
        self.depth_list = []
        self.spike_list_left = []
        self.spike_list_right = []
        self.rgb_list_left = []
        self.rgb_list_right = []
        self.rgb_base_path = "/home/Datadisk/spikedata5622/spiking-2022/XVFI-main/sterkitti/"
        
        for s in self.spike_sequences:
            s_seq_path = os.path.join(self.spike_base_path, s,)
           
            pics_left = os.listdir(os.path.join(s_seq_path, "Camera_0/spike/r32"))
            lenth = len(pics_left)
            for pic in pics_left:
                this_ord = int(pic[-13: -4])                
                if this_ord >=5 and this_ord <= int(lenth-6):
                    
                    self.depth_list.append(os.path.join(self.depth_base_path, "train", s, "proj_depth", "groundtruth", "image_03", pic.replace("npy","png")))
                    
                self.spike_list_left.append(os.path.join(s_seq_path, "Camera_0/spike/r32", pic))
                self.spike_list_right.append(os.path.join(s_seq_path, "Camera_1/spike/r32", pic))
                
                if split == "train":
                    left_dir = s + "-left"
                    right_dir = s + "-right"
                elif split == "val":
                    left_dir = s + "-left-val"
                    right_dir = s + "-right-val"                    
                    
                self.rgb_list_left.append(os.path.join(self.rgb_base_path, left_dir, pic.replace("npy","png")))
                self.rgb_list_right.append(os.path.join(self.rgb_base_path, right_dir, pic.replace("npy","png")))                                                                                                
        
        self.depth_list.sort()   
        self.spike_list_left.sort() 
        self.spike_list_right.sort()
        self.rgb_list_left.sort()
        self.rgb_list_right.sort()
        
        
        self.split = split
        
    def __len__(self):
        return len(self.depth_list)

    def __getitem__(self, idx):
        
        '''
        spike_path_left = self.spike_list_left[idx]
        #print(spike_path_left)
        spike_path_right = spike_path_left.replace("Camera_0","Camera_1")
        
        if self.split == "train":
            this_sequence = spike_path_left[56:75]   
            this_order = spike_path_left[95: -4]
        elif self.split == "val":
            this_sequence = spike_path_left[54:73] 
            this_order = spike_path_left[93: -4]            
        #print(this_sequence, this_order)
        
        depth_gt_path = os.path.join(self.depth_base_path, "train",this_sequence, "proj_depth", "groundtruth", "image_03", this_order) + ".png"
        '''
        #/home/lijianing/kitti_depth/train/2011_09_26_drive_0015_sync/proj_depth/groundtruth/image_03/0000000001.png
        depth_gt_path = self.depth_list[idx]
        depth_gt = np.expand_dims(self.get_gt_depth_maps(depth_gt_path), 0) 
        
        this_sequence = depth_gt_path[34:60]
        this_order = depth_gt_path[93:103]
        

        
        depth_gt = torch.FloatTensor(depth_gt)
        
        #print(this_order)
        spike_path_left = os.path.join(self.spike_base_path, this_sequence, "Camera_0", "spike", "r32", this_order) + '.npy'
        spike_path_right = os.path.join(self.spike_base_path, this_sequence, "Camera_1", "spike", "r32", this_order) + '.npy'
        if self.split == "train":
            rgb_left_path = os.path.join(self.rgb_base_path, this_sequence +"-left", this_order) + '.png'
            rgb_right_path = os.path.join(self.rgb_base_path, this_sequence +"-right", this_order) + '.png'
        elif self.split == "val":
            rgb_left_path = os.path.join(self.rgb_base_path, this_sequence +"-left-val", this_order) + '.png'
            rgb_right_path = os.path.join(self.rgb_base_path, this_sequence +"-right-val", this_order) + '.png'
        '''    
        rgb_left = np.array(Image.open(rgb_left_path)).transpose(2,0,1)
        rgb_right = np.array(Image.open(rgb_right_path)).transpose(2,0,1)
        #print(depth_gt_path, self.rgb_list_left[idx], self.rgb_list_right[idx])

                
        rgb_left = torch.FloatTensor(rgb_left)
        rgb_right = torch.FloatTensor(rgb_right)
        #print(rgb_right.size())
        '''
        
        scale = transforms.Compose([
         transforms.Resize([256, 512]),
         ])  
         
        #F.upsample(self.softplus(output[:,0,:,:]).unsqueeze(1), [375,1242], mode='bilinear', align_corners=True).squeeze(1)depth_gt = scale(depth_gt)
         
        disp_gt = depth_gt
        disp_gt[disp_gt >80.0] = 80.0
        disp_gt[disp_gt <=0 ] = -1
        disp_gt = 1 / disp_gt
        disp_gt[disp_gt <=0 ] = 0
        
        
        depth_gt[depth_gt > 80.0] = 80.0
        depth_gt[depth_gt < 0.1] = 0.1
        
        depth_gt = 1/depth_gt #/ 100.0
        #dat_path = os.path.join(self.spike_base_path, this_sequence, "spike/r32/spike_{}.npy".format(this_order))
        npy_path = os.path.join(self.spike_base_path, this_sequence, "spike/r32/{}.npy".format(this_order)) #spike_ 
        

        spike_mat_left = self.load_np(spike_path_left)#self.analysedat(dat_path)
        spike_mat_left = 2*spike_mat_left - 1 #spike_mat_left*254 + 1 #
        spike_mat_right = self.load_np(spike_path_right)#self.analysedat(dat_path)
        spike_mat_right = 2*spike_mat_right - 1 #spike_mat_right*254 + 1#
                
        
        
        spike_left = torch.FloatTensor(spike_mat_left)
        spike_right =  torch.FloatTensor(spike_mat_right)
        

        sample = {}
        sample["left"] = scale(spike_left)
        sample["right"] = scale(spike_right)
        sample["depth"] = depth_gt.squeeze(0)
        sample["disparity"] = disp_gt.squeeze(0)
        sample["rgb_left"] = scale(spike_left)#None#scale(rgb_left)
        sample["rgb_right"] = scale(spike_right)#None#scale(rgb_right)
  
        return sample#spike, depth
        
    def get_gt_depth_maps(self, depth_map_path):
        
        depth_map_gt = np.array(Image.open(depth_map_path), dtype=np.float32) / self.depth_scale

        
        return depth_map_gt

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
        

        return mat                 


