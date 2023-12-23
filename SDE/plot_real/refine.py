
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
import matplotlib

from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import shutil

import random


import re
import numpy as np
import sys
from PIL import Image

def readPFM(file):
    file = open(file, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    if (sys.version[0]) == '3':
        header = header.decode('utf-8')
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    if (sys.version[0]) == '3':
        dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    else:
        dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline())
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    if (sys.version[0]) == '3':
        scale = float(file.readline().rstrip().decode('utf-8'))
    else:
        scale = float(file.readline().rstrip())
        
    if scale < 0: # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>' # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)

    depth_im = Image.fromarray(data)#.convert("L")
    #depth_im = depth_im.transpose(Image.FLIP_TOP_BOTTOM)
    #depth_im = depth_im.transpose(Image.FLIP_LEFT_RIGHT)
    data = np.array(depth_im)
        
    return data, scale 




def trans_color():
    ori_path = "/home/lijianing/depth/CFNet-mod/new_results_CM/results_cnn/13.png"#"/home/lijianing/depth/CFNet-mod/plot_real/results_cnn/20__13.png"   #"/home/lijianing/depth/CFNet-mod/plot_real/results_cnn/52__23.png"      
    ori = Image.open(ori_path)
    #ori = ori.transpose(Image.FLIP_TOP_BOTTOM)
    #ori = ori.resize((720, 1280), Image.ANTIALIAS)
    ori = np.array(ori)
    #ori[ori > 200 ] = 50 
    
    o2, _ = readPFM("/home/Datadisk/spikedata5622/spiking-2022/outdoor_real_spike/depth/val/0023/0000.pfm")    #("/home/Datadisk/spikedata5622/spiking-2022/outdoor_real_spike/depth_trans/val/0035/0000.pfm")
    print(np.min(o2))
    o2[o2 <= 5.0] = 10.03
    o2 = 0.15 * o2 / 20.0 * 255.0
    
    ori = 0.85 * ori +  o2

    img = Image.fromarray(ori).convert("L")
    #img = img.transpose(Image.FLIP_TOP_BOTTOM)
    img.save("2.png")
    #fig = cv2.cvtColor(np.asarray(ori),cv2.COLOR_GRAY2RGB) 
    #fig = cv2.applyColorMap(fig, cv2.COLORMAP_TURBO)
    #cv2.imwrite("10.png",fig)
    #video.write(fig)   

def trans_color1():
    ori_path = "/home/lijianing/depth/CFNet-mod/plot_real/results_cnn/ster318__22.png"      
    ori = Image.open(ori_path)
    #ori = 2000 / np.array(ori)
    ori = 20*np.array(ori)
    print(np.min(ori))
    ori[ori>200] = 170

    fig = cv2.cvtColor(np.asarray(ori),cv2.COLOR_GRAY2RGB) 
    fig = cv2.applyColorMap(fig, cv2.COLORMAP_TURBO)
    cv2.imwrite("out2.png",fig)

def trans_color2():
    ori_path1 = "/home/lijianing/depth/CFNet-mod/new_results_CM_depthtrans_smt/results_cnn_new/14__12.png"      
    ori1 = Image.open(ori_path1)
    #ori = 2000 / np.array(ori)
    ori1 = np.array(ori1)
    ori1[ori1>200] = 170

    ori_path2 = "/home/lijianing/depth/CFNet-mod/new_results_CM_depthtrans_smt/results_cnn/ster86__12.png"      
    ori2 = Image.open(ori_path2)
    ori2 = np.array(ori2)

    mask = np.zeros((768, 1024))
    #mask[ori1 > 40] = 1
    mask[ori1 > 65] = 1
    ori = mask * ori1  + (1-mask) * ori2    
    ori = np.array(ori, dtype = np.uint8)
    
    fig = cv2.cvtColor(np.asarray(ori),cv2.COLOR_GRAY2RGB) 
    fig = cv2.applyColorMap(fig, cv2.COLORMAP_TURBO)
    cv2.imwrite("20.png",fig)
    
if __name__ == "__main__":
    trans_color1()   
