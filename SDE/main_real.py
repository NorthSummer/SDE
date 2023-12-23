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
from models import __models__, model_loss#, get_smooth_loss
from utils import *
from torch.utils.data import DataLoader
import gc
from PIL import Image
from datasets.SpikeDataset import SpikeDataset, SpikeDataset2
from datasets.SpikeDataset import SpikeDS, SpikeTN
from datasets.Spikeset import JYSpikeDataset
from datasets.stervkitti import SVKitti
from datasets.spikeds import SpikeDrivingStereo
from datasets.RealSpike import SpikeReal#, SpikeRealII, IMGReal
from torchstat import stat

import torchvision.transforms as transforms
from evaluate_real import validate_spike, plot_spike
import matplotlib.pyplot as plt
from models.loss import scale_invariant_loss, SILogLoss, MultiScaleGradient, get_smooth_loss

cudnn.benchmark = True
#os.environ["CUDA_VISIBLE_DEVICES"]='1,2,3,4,5,6,7'

parser = argparse.ArgumentParser(description='Cascade and Fused Cost Volume for Robust Stereo Matching(CFNet)')
parser.add_argument('--model', default='cfnet', help='select a model structure', choices=__models__.keys())
parser.add_argument('--maxdisp', type=int, default=160, help='maximum disparity')

parser.add_argument('--dataset', default='kitti', required=False, help='dataset name', choices=__datasets__.keys())
parser.add_argument('--datapath', default='/home/lijianing/kitti/', required=False, help='data path')
parser.add_argument('--trainlist', default='/home/lijianing/depth/CFNet-mod/filenames/kitti15_train.txt', required=False, help='training list')
parser.add_argument('--testlist', default='/home/lijianing/depth/CFNet-mod/filenames/kitti15_val.txt', required=False, help='testing list')

parser.add_argument('--lr', type=float, default=0.001, help='base learning rate')
parser.add_argument('--batch_size', type=int, default=3, help='training batch size')
parser.add_argument('--test_batch_size', type=int, default=1, help='testing batch size')
parser.add_argument('--epochs', type=int, default=600, required=False, help='number of epochs to train')
parser.add_argument('--lrepochs', type=str, default='20:3',required=False, help='the epochs to decay lr: the downscale rate')

parser.add_argument('--logdir', default='/home/lijianing/depth/CFNet-mod/logs_aaai_cnn/', required=False, help='the directory to save logs and checkpoints')
parser.add_argument('--loadckpt', help='load the weights from a specific checkpoint')
parser.add_argument('--resume', action='store_true', help='continue training the model')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')

parser.add_argument('--summary_freq', type=int, default=20, help='the frequency of saving summary')
parser.add_argument('--save_freq', type=int, default=1, help='the frequency of saving checkpoint')

# parse arguments, set seeds
args = parser.parse_args()
torch.manual_seed(args.seed) 
torch.cuda.manual_seed(args.seed)
os.makedirs(args.logdir, exist_ok=True)

# create summary logger
print("creating new summary file")
logger = SummaryWriter(args.logdir)
 

StereoDataset = __datasets__[args.dataset]
#train_dataset =  SpikeDataset(pathr = "/home/Datadisk/spikedata5622/spiking-2022/train/firsthright/", pathl = "/home/Datadisk/spikedata5622/spiking-2022/train/firsthleft/", mode = "training")   
#train_dataset =  SpikeDataset2(pathr = "/home/Datadisk/spikedata5622/spiking-2022/train/leftspike/", pathl = "/home/Datadisk/spikedata5622/spiking-2022/train/leftspike/", mode = "train")   

#train_dataset =  JYSpikeDataset(path="/home/Datadisk/spikedata5622/spiking-2022/JYsplit/train/")
#train_dataset = SpikeDS(pathr = "/home/Datadisk/spikedata5622/spiking-2022/SpikeDriveStereo/trainset/rightspike/Sequence_33/spike/", pathl = "/home/Datadisk/spikedata5622/spiking-2022/SpikeDriveStereo/trainset/leftspike/Sequence_33/spike/", mode = "train")
#train_dataset = SpikeTN(pathr = "/home/Datadisk/spikedata5622/spiking-2022/SpikeTunel/road/road/right", pathl = "/home/Datadisk/spikedata5622/spiking-2022/SpikeTunel/road/road/right", mode = "train")

#test_dataset = SpikeDataset(pathr = "/home/Datadisk/spikedata5622/spiking-2022/test/nrpz/", pathl = "/home/Datadisk/spikedata5622/spiking-2022/test/nlpz/", mode = "test")
#test_dataset = SpikeDataset2(pathr = "/home/Datadisk/spikedata5622/spiking-2022/train/leftspike/", pathl = "/home/Datadisk/spikedata5622/spiking-2022/train/leftspike/", mode = "test")
#test_dataset = SpikeTN(pathr = "/home/Datadisk/spikedata5622/spiking-2022/SpikeTunel/road/road/right", pathl = "/home/Datadisk/spikedata5622/spiking-2022/SpikeTunel/road/road/left", mode = "test")



trans = transforms.Compose([
     transforms.Resize([256, 512]),])
'''
train_dataset = SVKitti(root = "/home/Datadisk/VirtualKitti/", split='train', mode='fine', target_type='color', modality = 'clone', transform=trans)
test_dataset = SVKitti(root = "/home/Datadisk/VirtualKitti/", split='val', mode='fine', target_type='color', modality = 'clone', transform=trans)
'''

#train_dataset = SpikeDrivingStereo(spike_base_path="/home/Datadisk/spikedata5622/spiking-2022/spikeds/", depth_base_path="/home/Datadisk/spikedata5622/DrivingStereo/train-depth-map/", split = "train")
#test_dataset = SpikeDrivingStereo(spike_base_path="/home/Datadisk/spikedata5622/spiking-2022/spikeds/", depth_base_path="/home/Datadisk/spikedata5622/DrivingStereo/train-depth-map/", split = "val")


train_dataset = SpikeReal(root_path = "/home/Datadisk/spikedata5622/spiking-2022/outdoor_real_spike/", split="train")
test_dataset = SpikeReal(root_path = "/home/Datadisk/spikedata5622/spiking-2022/outdoor_real_spike/", split="val")


TrainImgLoader = DataLoader(train_dataset, args.batch_size, shuffle=False, num_workers=0, drop_last=True)
TestImgLoader = DataLoader(test_dataset, args.test_batch_size, shuffle=False, num_workers=0, drop_last=False)

#from models.ugde_ds import SpikeFusionet
from models.loss import gcnet_loss, gwc_loss, psm_loss, SL1Loss, unc_loss, stereo_unc_loss
from models.ugde_real import SpikeFusionet
#from models.ugde_former_real import SpikeFusionet




device = torch.device("cuda:{}".format(7))  # set main gpu

model = SpikeFusionet(max_disp=128, device = device)
model = model.to(device)
'''
from thop import profile



input = torch.randn(1, 1, 256, 512).to(device)
macs, params = profile(model, inputs=(input, input))
print(macs, params)
'''
'''
with torch.cuda.device([5,6,7]):
    model = nn.DataParallel(model, [5,6,7]) 
'''
#model.cuda()


            
optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
#state_dict = torch.load('/home/lijianing/depth/MMlogs/256/ours/18---checkpoint_max_4.3_stage_unc.ckpt')
#model.load_state_dict(state_dict['model'], False)

# load parameters
start_epoch = 0
if args.resume:
    # find all checkpoints file and sort according to epoch id
    all_saved_ckpts = [fn for fn in os.listdir(args.logdir) if fn.endswith(".ckpt")]
    all_saved_ckpts = sorted(all_saved_ckpts, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    # use the latest checkpoint file
    loadckpt = os.path.join(args.logdir, all_saved_ckpts[-1])
    print("loading the lastest model in logdir: {}".format(loadckpt))
    state_dict = torch.load(loadckpt)
    model.load_state_dict(state_dict['model'])
    optimizer.load_state_dict(state_dict['optimizer'])
    start_epoch = state_dict['epoch'] + 1
elif args.loadckpt:
    # load the checkpoint file specified by args.loadckpt
    print("loading model {}".format(args.loadckpt)) 
    state_dict = torch.load(args.loadckpt)
    model.load_state_dict(state_dict['model'])
print("start at epoch {}".format(start_epoch))


def freeze_model(model, pos):
    if pos == "decoder":    
        for param in model.decoder.parameters():
            param.requires_grad = False
    if pos == "encoder":
        for param in model.encoder.parameters():
            param.requires_grad = False    
    return model    


def train():
    train_losses = []
    bestepoch = 0
    error = 100
    for epoch_idx in range(start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch_idx, args.lr, args.lrepochs)
       
        train_loss = 0
        # training
        
        for batch_idx, sample in enumerate(TrainImgLoader):
            global_step = len(TrainImgLoader) * epoch_idx + batch_idx
            start_time = time.time()
            do_summary = global_step % args.summary_freq == 0
            loss, scalar_outputs = train_sample(epoch_idx, sample, compute_metrics=do_summary) #, image_outputs 
            train_loss += loss
            if do_summary:
                save_scalars(logger, 'train', scalar_outputs, global_step)
                #save_images(logger, 'train', image_outputs, global_step)
            del scalar_outputs#, image_outputs
            print('Epoch {}/{}, Iter {}/{}, train loss = {:.3f}, time = {:.3f}'.format(epoch_idx, args.epochs,
                                                                                     batch_idx,
                                                                                       len(TrainImgLoader), loss,
                                                                                    time.time() - start_time))
                                                                                 
        train_losses.append(train_loss / len(TrainImgLoader))
        
        
        # saving checkpoints
        if (epoch_idx + 1) % args.save_freq == 0:
            checkpoint_data = {'epoch': epoch_idx, 'model': model.state_dict(), 'optimizer': optimizer.state_dict()} ##########3
            torch.save(checkpoint_data, "{}/checkpoint_max_real_0817_rec.ckpt".format(args.logdir))
        gc.collect()
        
        nowerror = validate_spike(model, dataloader = TestImgLoader, epoch = epoch_idx)
        
        # testing
    
        if  nowerror <= error :
            bestepoch = epoch_idx
            error = nowerror
            plot_spike(model, dataloader = TestImgLoader, epoch = epoch_idx)
            torch.save(checkpoint_data, "{}/checkpoint_max_real_ster_best.ckpt".format(args.logdir))
            
        print('MAX epoch %d total test error = %.5f' % (bestepoch, error))
        gc.collect()
        
    print('MAX epoch %d total test error = %.5f' % (bestepoch, error))
    plot_losses(train_losses)
       
    #print('MAX epoch %d total test error = %.5f' % (bestepoch, error))


# train one sample
def train_sample(epoch, sample, compute_metrics=False):
    model.train()
    
    imgL, imgR, disp_gt, depth_gt = sample['left'], sample['right'], sample['disparity'], sample['depth']#, sample['left_img'], sample['right_img'] #, sample['SID'], sample['depth_dorn']
    imgL = imgL.to(device)
    imgR = imgR.to(device)
    #recL = recL.to(device)
    #recR = recR.to(device)
    
    disp_gt = disp_gt.to(device)
    depth_gt = depth_gt.to(device)
    
    #sid_gt = sid_gt.to(device)
    
    stereo_gt = disp_gt[0].detach().cpu().squeeze(0)
        
    stereo_gt_np = np.array(stereo_gt, dtype = np.float32)
    stereo_gt_img = Image.fromarray(stereo_gt_np, 'L')
    stereo_gt_img.save('/home/lijianing/depth/CFNet-mod/gt{}.png'.format('1'))  
    

    optimizer.zero_grad()

    ests = model(imgL, imgR)
    disp_ests = ests['stereo']

    depth_ests = ests['monocular']
    
    
    fusion_ests = ests['fusion']
    
     
    
    stereo_uncertainty = ests["stereo_uncertainty"]
    
    stereo_depth = 1/disp_ests[-1]
    
    mask_disp = (disp_gt < args.maxdisp) & (disp_gt > 0)
    mask_depth = (depth_gt <= 20.0) & (depth_gt > 0) #& (depth_ests["depth"]*100.0 > 0)
    
    #depth_gt[depth_gt==0] = 20
    
    #depth_gt = (1 / 5.7) * torch.log(depth_gt/20.0) + 1.0
    
    
    loss1 = psm_loss(disp_ests, disp_gt, mask_disp)
    
    
    
    #F.smooth_l1_loss(depth_ests["depth"], depth_gt)#F.smooth_l1_loss(depth_ests["depth"], depth_gt)
    #loss2 = scale_invariant_loss(depth_ests["depth"], depth_gt)

    #
    #loss4 = F.smooth_l1_loss(fusion_ests[mask_depth], depth_gt[mask_depth])
    #loss2 = F.smooth_l1_loss(depth_ests["depth"][mask_depth], depth_gt[mask_depth])
    #loss2 = F.mse_loss(depth_ests["depth"][mask_depth], depth_gt[mask_depth])
    loss0 = F.mse_loss(depth_ests["depth"][mask_depth], depth_gt[mask_depth])
    
    sig = SILogLoss()#scale_invariant_loss(depth_ests["depth"], depth_gt)
    #smt = MultiScaleGradient(start_scale = 1, num_scales = 1, mask = mask_depth)
    
    loss2 = sig(depth_ests["depth"][mask_depth], depth_gt[mask_depth])
    
    loss3 = unc_loss(depth_ests, depth_gt, mask_depth)
    
    #loss4 = smt(depth_ests["depth"], depth_gt, preview=False)#get_smooth_loss(depth_ests["depth"], depth_gt, mask_depth)#smt(depth_ests["depth"], depth_gt, preview=False)
    #loss6 = stereo_unc_loss(disp_ests[-1], disp_gt, stereo_uncertainty)
    
    '''
    if epoch <= 10:
        loss = 5.0*loss2 + loss3
    elif 10<epoch <= 20:
        loss = loss1 + loss2 + loss3 #+ loss6
        
    elif 20<epoch<=40:
        loss = loss1 + loss3#loss3 + loss4 + loss6#loss1 + loss2 + loss3 + loss4 
    else:
        loss = loss3 + loss4 + loss6#0.5*loss1 + 0.5*loss2 + loss3 + 5.0*loss4
    '''
    #loss = loss1 + loss2 #+ 0.1* loss3 #+ 0.001* loss4 #+ 0.1* loss4 #loss2 + 0.5*loss9 #+ loss3 #loss2 
    #print(loss1, loss2)
    
    if epoch <= 10:
        loss = loss0
        
    elif 10<epoch<=20:
        loss = 2*loss1 + loss2 #+loss3#+ loss3 + loss6 + 10.0 *loss4
    
    else:
        loss = loss1 + loss2 + 0.1*loss3
    
    scalar_outputs = {"loss": loss}

 
    loss.backward()
    optimizer.step()

    return tensor2float(loss), tensor2float(scalar_outputs)#, image_outputs


def get_smooth_loss(disp, img, mask):
    """Computes the smoothness loss for a disparity image
    The color image is used for edge-aware smoothness
    """
    #print(disp.size())
    grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])[mask[:, :, :, 1:]]
    grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])[mask[:, :, 1:, :]]

    grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)[mask[:, :, :, 1:]]
    grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)[mask[:, :, 1:, :]]

    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)
    
    isnan1 = torch.isnan(grad_disp_x)
    isnan2 = torch.isnan(grad_disp_y)

    return grad_disp_x[~isnan1].mean() + grad_disp_y[~isnan2].mean()
    
    
    
def plot_losses(train_losses):
    plt.plot(train_losses, label='train losses') 
    #plt.plot(valid_losses, label='valid losses')
    
    plt.xlabel("Iterations")
    plt.ylabel("Losses")
    
    plt.legend()
    plt.title("Losses")
    plt.grid(True)
    plt.savefig("./losses.jpg")

# test one sample
@make_nograd_func
def test_sample(sample, compute_metrics=True): 
    model.eval()

    imgL, imgR, disp_gt, depth_gt = sample['left'], sample['right'], sample['disparity'], sample['depth']
    imgL = imgL.cuda()
    imgR = imgR.cuda()
    disp_gt = disp_gt.cuda()
    depth_gt = depth_gt.cuda()


    pred_depth,_,_,_,_ = model(imgL, imgR)["monocular"]

    
    mask_depth = (depth_gt < 256.0) & (depth_gt > 0)
    #loss = F.smooth_l1_loss(pred_depth[-1], depth_gt)#model_loss(disp_ests, disp_gt, mask)
    loss = F.smooth_l1_loss(pred_depth[mask_depth], depth_gt[mask_depth])
    scalar_outputs = {"loss": loss}
    

   
    scalar_outputs["D1"] = [D1_metric(100.0*(pred_depth + 1.0) , 100.0*(depth_gt + 1.0), mask)] 

    return tensor2float(loss), tensor2float(scalar_outputs)


if __name__ == '__main__':
    train()
