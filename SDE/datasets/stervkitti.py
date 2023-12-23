import json
import os
from collections import namedtuple

import torch
import torch.utils.data as data
from PIL import Image
import numpy as np
import cv2
import torchvision.transforms as transforms

class SVKitti(data.Dataset):
    """Cityscapes <http://www.cityscapes-dataset.com/> Dataset.
    
    **Parameters:**
        - **root** (string): Root directory of dataset where directory 'leftImg8bit' and 'gtFine' or 'gtCoarse' are located.
        - **split** (string, optional): The image split to use, 'train', 'test' or 'val' if mode="gtFine" otherwise 'train', 'train_extra' or 'val'
        - **mode** (string, optional): The quality mode to use, 'gtFine' or 'gtCoarse' or 'color'. Can also be a list to output a tuple with all specified target types.
        - **transform** (callable, optional): A function/transform that takes in a PIL image and returns a transformed version. E.g, ``transforms.RandomCrop``
        - **target_transform** (callable, optional): A function/transform that takes in the target and transforms it.
    """

    # Based on https://github.com/mcordts/cityscapesScripts
    VKittiClass = namedtuple('VKittiClass', ['name', 'id', 'train_id', 'category', 'category_id',
                                                     'has_instances', 'ignore_in_eval', 'color'])


    classes = [
        VKittiClass('terrain',              0, 0, 'void', 0, False, True, (210, 0, 200)),
        VKittiClass('sky',                  1, 1, 'void', 0, False, True, (90, 200, 255)),
        VKittiClass('tree',                 2, 2, 'void', 0, False, True, (0, 199, 0)),
        VKittiClass('vegetation',           3, 3, 'void', 0, False, True, (90, 240, 0)),
        VKittiClass('building',             4, 4, 'void', 0, False, True, (140, 140, 140)),
        VKittiClass('road',                 5, 5, 'void', 0, False, True, (100, 60, 100)),
        VKittiClass('guardrail',            6, 6, 'void', 0, False, True, (250, 100, 255)),
        VKittiClass('trafficsign',          7, 7, 'flat', 1, False, False, (255, 255, 0)),
        VKittiClass('trafficlight',         8, 8, 'flat', 1, False, False, (255, 255, 0)),
        VKittiClass('pole',                 9, 9, 'flat', 1, False, True, (255, 130, 0)),
        VKittiClass('misc',                 10, 13, 'flat', 1, False, True, (80, 80, 80)),
        VKittiClass('truck',                11, 10, 'construction', 2, False, False, (160, 60, 60)),
        VKittiClass('car',                  12, 11, 'construction', 2, False, False, (255, 127, 80)),
        VKittiClass('van',                  13, 12, 'construction', 2, False, False, (0, 139, 139)),
        VKittiClass('undefined',            14, 13, 'construction', 2, False, True, (0, 0, 0)),
    ]





    train_id_to_color = [c.color for c in classes if (c.train_id != -1 and c.train_id != 255)]
    train_id_to_color.append([0, 0, 0])
    train_id_to_color = np.array(train_id_to_color)
    id_to_train_id = np.array([c.train_id for c in classes])
    
    #train_id_to_color = [(0, 0, 0), (128, 64, 128), (70, 70, 70), (153, 153, 153), (107, 142, 35),
    #                      (70, 130, 180), (220, 20, 60), (0, 0, 142)]
    #train_id_to_color = np.array(train_id_to_color)
    #id_to_train_id = np.array([c.category_id for c in classes], dtype='uint8') - 1

    def __init__(self, root, split='train', mode='fine', target_type='semantic', modality = 'clone', transform=None):
        self.root = os.path.expanduser(root)
        self.mode = 'color'
        self.mode_depth = 'depth'
        
        self.target_type = target_type
        self.target_depth_type = 'depth'
        
        self.images_dir = os.path.join(self.root, 'rgb', split)

        self.targets_dir = os.path.join(self.root, 'classSegmentation', split)
        self.depth_targets_dir = os.path.join(self.root, 'depth', split)
        self.spike_dir = os.path.join(self.root, 'spike', split)
        self.transform = transform
        self.modality = modality
        self.num_classes = 14
        


        self.split = split
        self.images_left = []
        self.images_right = []
        self.spike_left = []
        self.spike_right = []
        
        self.targets = []
        self.targets_depth = []

        if split not in ['train', 'test', 'val']:
            raise ValueError('Invalid split for mode! Please use split="train", split="test"'
                             ' or split="val"')

        if not os.path.isdir(self.images_dir) or not os.path.isdir(self.targets_dir):
            raise RuntimeError('Dataset not found or incomplete. Please make sure all required folders for the'
                               ' specified "split" and "mode" are inside the "root" directory')
        
        for city in os.listdir(self.images_dir):
            img_left_dir = os.path.join(self.images_dir, city, self.modality, 'frames', 'rgb', 'Camera_0')
            img_right_dir = os.path.join(self.images_dir, city, self.modality, 'frames', 'rgb', 'Camera_1')
            spike_left_dir = os.path.join(self.spike_dir, city, 'Camera_0', 'spike', 'r32')
            spike_right_dir = os.path.join(self.spike_dir, city, 'Camera_1', 'spike', 'r32')
            target_dir = os.path.join(self.targets_dir, city, self.modality, 'frames', 'classSegmentation', 'Camera_1')
            depth_target_dir = os.path.join(self.depth_targets_dir, city, self.modality, 'frames', 'depth', 'Camera_1')

            for file_name in os.listdir(img_left_dir):
                self.images_left.append(os.path.join(img_left_dir, file_name))                               #aachen_000000_000019_leftImg8bit.png        #rgb_00000.jpg
                self.images_right.append(os.path.join(img_right_dir, file_name))    
                 
                target_name = '{}_{}'.format(self._get_target_suffix(self.mode, self.target_type),
                                             file_name.split('_')[1].replace("jpg","png"))  #aachen_000000_000019_gtFine_color.png        #classgt_00000.png
                target_depth_name = '{}_{}'.format(self._get_target_suffix(self.mode_depth, self.target_depth_type),
                                             file_name.split('_')[1].replace("jpg","png"))                                              
                self.targets.append(os.path.join(target_dir, target_name))
                self.targets_depth.append(os.path.join(depth_target_dir, target_depth_name))        # sortttttttt them !!!!!!!!

            for spike_name in os.listdir(spike_left_dir):                
                self.spike_left.append(os.path.join(spike_left_dir, spike_name))
                self.spike_right.append(os.path.join(spike_right_dir, spike_name))
                
        self.images_left.sort()
        self.images_right.sort()
        self.spike_left.sort()
        self.spike_left.sort()
        self.targets.sort()
        self.targets_depth.sort()
        
        #print(self.spike_left)
        #print(self.targets_depth)
        
        
    @classmethod
    def encode_target(cls, target):        
        
        return cls.id_to_train_id[np.array(target)]

    @classmethod
    def decode_target(cls, target):
        target[target == 255] = 19
        
        return cls.train_id_to_color[target]
    @classmethod
    def encode_color_target(cls, target):
        
        
        pass

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is a tuple of all target types if target_type is a list with more
            than one item. Otherwise target is a json object if target_type="polygon", else the image segmentation.
        """        
        spike_left = self.load_np(self.spike_left[index])
        spike_right = self.load_np(self.spike_right[index])
        
        spike_left = spike_left*2 -1
        spike_right = spike_right*2 -1 
        
        image_right = Image.open(self.images_right[index]).convert('RGB')
        image_right = cv2.imread(self.images_right[index], cv2.IMREAD_COLOR)
        image_right = np.array(image_right).transpose(2,0,1)
        
        image_left = Image.open(self.images_left[index]).convert('RGB')
        image_left = cv2.imread(self.images_left[index], cv2.IMREAD_COLOR)
        image_left = np.array(image_left).transpose(2,0,1)
        #print(self.images_right[index])
        #target = Image.open(self.targets[index])#.convert('RGB')
        target = cv2.imread(self.targets[index], cv2.IMREAD_COLOR)
        target = np.array(target)
        
        target = self.rgb2label_mapping(target)
        
        '''        
        print(tuple(target[200,500]))
        f_target = np.array(target).transpose(2,0,1)
        print(f_target.shape)
        
        t_image = cv2.imread(self.targets[index], cv2.IMREAD_COLOR)
        t_image = np.array(t_image)#.transpose(2,0,1)
        print(t_image[200,500])
        '''
        target_depth = Image.open(self.targets_depth[index])
        
        target_depth = np.array(target_depth) / 100.0
        
        target_depth[target_depth > 255.0 ] = 255.0
        target_depth[target_depth < 0.1 ] = 0.1
        
        target_disp = 1 / target_depth
        target_depth = target_depth #/ 256.0
        
        #print(np.min(target_depth), np.max(target_depth))
        '''
        if self.transform:
            image, target = self.transform(image, target, target_depth)
        '''
        target = self.encode_target(target)
        #target = self.id2onehot(target)
        
        image_left = torch.FloatTensor(np.array(image_left))
        image_right = torch.FloatTensor(np.array(image_right))
        target = torch.FloatTensor(target).long()
        target_depth = torch.FloatTensor(target_depth)
        target_disp = torch.FloatTensor(target_disp)
        
        trans = transforms.Compose([
         transforms.Resize([256, 512]),
         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
         #transforms.ToTensor(),
         ])

        scale = transforms.Compose([
         transforms.Resize([256, 512]),
         ])        
        samples = {}
        #print(target_depth.shape)
        samples["depth"] = target_depth.unsqueeze(0).squeeze(0)#.squeeze(0)) #self.transform(target_depth.unsqueeze(0))#.squeeze(0)
        samples["disparity"] = target_disp.unsqueeze(0).squeeze(0)
        samples["seman"] = target #self.transform(target.unsqueeze(0)).squeeze(0)
        samples["rgb"] = trans(image_right)
        #samples["spike"] = self.transform(image)#None
        samples["left_img"] = trans(image_left)
        samples["right_img"] = trans(image_right)
        samples["left"] = scale(spike_left)
        samples["right"] = scale(spike_right)
        
       
        return samples#image, target

    def __len__(self):
        return int(len(self.images_right) - 10)

    def _load_json(self, path):
        with open(path, 'r') as file:
            data = json.load(file)
        return data
        
    def load_np(self, np_path):
        npy = np.load(np_path).astype(np.uint8)
        return torch.FloatTensor(npy)
        
    def _get_target_suffix(self, mode, target_type):
        if target_type == 'instance':
            return '{}_instanceIds.png'.format(mode)
        elif target_type == 'semantic':
            return '{}_labelIds.png'.format(mode)
        elif target_type == 'color':
            return 'classgt' #'{}_color.png'.format(mode)
        elif target_type == 'polygon':
            return '{}_polygons.json'.format(mode)
        elif target_type == 'depth':
            return 'depth' #'{}_disparity.png'.format(mode)
    
    def rgb2label_mapping(self, target):
        
        map_ = [(210, 0, 200),
              (90, 200, 255),
              (0, 199, 0),
              (90, 240, 0),
              (140, 140, 140),
              (100, 60, 100),
              (250, 100, 255),
              (255, 255, 0),
              (255, 255, 0),
              (255, 130, 0),
              (80, 80, 80),
              (160, 60, 60),
              (255, 127, 80),
              (0, 139, 139),
              (0, 0, 0),
        ]
        
        gray_img = np.zeros(shape=(target.shape[0], target.shape[1]), dtype=np.uint8)
        for map_idx, rgb in enumerate(map_):
                      
            idx = np.where((target[...,0] == rgb[2]) & (target[...,1] == rgb[1]) & (target[...,2] == rgb[0]))

            gray_img[idx] = map_idx
        
       
        return gray_img            
            
            
    def id2onehot(self, target):
        new_tar = np.zeros(shape=(self.num_classes, target.shape[0], target.shape[1]), dtype = np.uint8)
        for i in range(0, target.shape[0]):
            for j in range(0, target.shape[1]):
                cls = target[i,j]
                new_tar[cls,i,j] = 1            
        return new_tar
            
                    
            
            