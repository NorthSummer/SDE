import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SILogLoss(nn.Module):  # Main loss function used in AdaBins paper£¬ better than L2(Mse loss)
    def __init__(self):
        super(SILogLoss, self).__init__()
        self.name = 'SILog'
        self.eps = 0.1 # in case of gradient explode

    def forward(self, input, target, mask=None, interpolate=True):
        '''
        if interpolate:
            input = nn.functional.interpolate(input, target.shape[-2:], mode='bilinear', align_corners=True)
        #print(input.size(), target.size())
        '''
        #print(input.size(), target.size())
        if mask is not None:
            input = input[mask]
            target = target[mask]
            
        
        g = torch.log(input + self.eps) - torch.log(target + self.eps)
        # n, c, h, w = g.shape
        # norm = 1/(h*w)
        # Dg = norm * torch.sum(g**2) - (0.85/(norm**2)) * (torch.sum(g))**2

        Dg = torch.var(g) + 0.15 * torch.pow(torch.mean(g), 2)
        return 10*torch.sqrt(Dg)#10 * torch.sqrt(torch.abs(Dg))
        
    def gradient_compute(self, log_pridiction, mask, log_gt):
        pass

def scale_invariant_loss(pred, truth):
    n_pixels = truth.shape[1] * truth.shape[2]
    


    pred[pred <= 0] = 0.01
    pred[pred >= 255] = 255
    
    #truth[truth == 0] = 0.3
    
    
    truth.unsqueeze_(dim=1)
    d = torch.log(pred+0.1) - torch.log(truth+0.1)
    term_1 = torch.pow(d.view(-1, n_pixels), 2).mean(dim=1).sum()  # pixel wise mean, then batch sum
    term_2 = (torch.pow(d.view(-1, n_pixels).sum(dim=1), 2) / (2 * (n_pixels ** 2))).sum()
    return term_1 - term_2


def unc_loss(outputs, outputs_gt, mask):
   
    losses ={}
    abs_diff = torch.abs(outputs["depth"][mask] - outputs_gt[mask])
    uncerted_l1_loss = ( abs_diff / outputs["uncertainty"][mask] + torch.log(outputs["uncertainty"][mask])).mean()
    return uncerted_l1_loss

def stereo_unc_loss(outputs, outputs_gt, uncertainty_map):
    losses ={}
    abs_diff = torch.abs(outputs - outputs_gt)
    uncerted_l1_loss = ( abs_diff / uncertainty_map + torch.log(uncertainty_map)).mean()
    return uncerted_l1_loss
    

class SL1Loss(nn.Module):
    def __init__(self, levels=3):
        super(SL1Loss, self).__init__()
        self.levels = levels
        self.loss = nn.SmoothL1Loss(reduction='mean')

    def forward(self, inputs, targets, masks):
        loss = 0
        for l in range(self.levels):
            depth_pred_l = inputs[f'depth_{l}']
            depth_gt_l = targets[f'level_{l}']
            mask_l = masks[f'level_{l}']
            loss += self.loss(depth_pred_l[mask_l], depth_gt_l[mask_l]) * 2**(1-l)
        return loss


def psm_loss(disp_ests, disp_gt, mask):
    weights = [0.5, 0.7, 1.0]
    all_losses = []
    for disp_est, weight in zip(disp_ests, weights):
        all_losses.append(weight * F.smooth_l1_loss(disp_est[mask], disp_gt[mask], size_average=True))
    return sum(all_losses)



def gwc_loss(disp_ests, disp_gt, mask):
    weights = [0.5, 0.5, 0.7, 1.0]
    all_losses = []
    for disp_est, weight in zip(disp_ests, weights):
        all_losses.append(weight * F.smooth_l1_loss(disp_est[mask], disp_gt[mask], size_average=True))
    return sum(all_losses)

def model_loss(disp_ests, disp_gt, mask):
    weights = [0.5 * 0.5, 0.5 * 0.7, 0.5 * 1.0, 1 * 0.5, 1 * 0.7, 1 * 1.0, 2 * 0.5, 2 * 0.7, 2 * 1.0] #0.5,1,2
    all_losses = []
    for disp_est, weight in zip(disp_ests, weights):
        all_losses.append(weight * F.smooth_l1_loss(disp_est[mask], disp_gt[mask], size_average=True))
    return sum(all_losses)


def get_smooth_loss(disp, img):
    """Computes the smoothness loss for a disparity image
    The color image is used for edge-aware smoothness
    """
    grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
    grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

    grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)

    return grad_disp_x.mean() + grad_disp_y.mean()
    
def gcnet_loss(xx,loss_mul,gt):
   
    loss=torch.sum(torch.sqrt(torch.pow(torch.sum(xx.mul(loss_mul),1)-gt,2)+0.00000001)/256/(256+128))
    return loss    

    
class OrdinalRegressionLoss(object):

    def __init__(self, ord_num, beta, discretization="SID"):
        self.ord_num = ord_num
        self.beta = beta
        self.discretization = discretization

    def _create_ord_label(self, gt):
        N, H, W = gt.shape
        # print("gt shape:", gt.shape)

        ord_c0 = torch.ones(N, self.ord_num, H, W).to(gt.device)
        gt = torch.from_numpy(np.array(gt.detach().cpu(), dtype = np.float32))
        if self.discretization == "SID":
            label = self.ord_num * torch.log(gt) / np.log(self.beta)
        else:
            label = self.ord_num * (gt - 1.0) / (self.beta - 1.0)
        label = label.long()
        mask = torch.linspace(0, self.ord_num - 1, self.ord_num, requires_grad=False) \
            .view(1, self.ord_num, 1, 1).to(gt.device)
        mask = mask.repeat(N, 1, H, W).contiguous().long()
        
        print(mask.size(), label.size())
        
        label = label.unsqueeze(1)
        label = label.repeat(1,self.ord_num, 1,1)
        
        mask = (mask > label)
        ord_c0[mask] = 0
        ord_c1 = 1 - ord_c0
        
        # implementation according to the paper.
        # ord_label = torch.ones(N, self.ord_num * 2, H, W).to(gt.device)
        # ord_label[:, 0::2, :, :] = ord_c0
        # ord_label[:, 1::2, :, :] = ord_c1
        # reimplementation for fast speed.
        
        ord_label = torch.cat((ord_c0, ord_c1), dim=1)
        return ord_label, mask

    def __call__(self, prob, gt):
        """
        :param prob: ordinal regression probability, N x 2*Ord Num x H x W, torch.Tensor
        :param gt: depth ground truth, NXHxW, torch.Tensor
        :return: loss: loss value, torch.float
        """
        # N, C, H, W = prob.shape
        valid_mask = gt > 0.
        ord_label, mask = self._create_ord_label(gt)
        # print("prob shape: {}, ord label shape: {}".format(prob.shape, ord_label.shape))
        entropy = -prob * ord_label
        loss = torch.sum(entropy, dim=1)[valid_mask]
        return loss.mean()
        
def robust_loss(x, a, c):
    abs_a_sub_2 = abs(a - 2)

    x = x / c
    x = x * x / abs_a_sub_2 + 1
    x = x ** (a / 2)
    x = x - 1
    x = x * abs_a_sub_2 / a
    return x


def calc_init_loss(cv, target, max_disp, k=1, tile_size=1):
    scale = target.size(3) // cv.size(3)
    scale_disp = max(1, scale // tile_size)

    target = target / scale_disp
    max_disp = max_disp / scale_disp

    target = F.max_pool2d(target, kernel_size=scale, stride=scale)
    mask = (target < max_disp) & (target > 1e-3)

    def rho(d):  # ¦Ñ(d)
        d = torch.clip(d, 0, cv.size(1) - 1)
        return torch.gather(cv, dim=1, index=d)

    def phi(d):  # ¦Õ(d)
        df = torch.floor(d).long()
        d_sub_df = d - df
        return d_sub_df * rho(df + 1) + (1 - d_sub_df) * rho(df)

    pixels = mask.sum() + 1e-6
    gt_loss = (phi(target) * mask).sum() / pixels

    d_range = torch.arange(0, max_disp, dtype=target.dtype, device=target.device)
    d_range = d_range.view(1, -1, 1, 1)
    d_range = d_range.repeat(target.size(0), 1, target.size(2), target.size(3))

    low = target - 1.5
    up = target + 1.5
    d_range_mask = (low <= d_range) & (d_range <= up) | (~mask)

    cv_nm = torch.masked_fill(cv, d_range_mask, float("inf"))
    cost_nm = torch.topk(cv_nm, k=k, dim=1, largest=False).values

    nm_loss = torch.clip(1 - cost_nm, min=0)
    nm_loss = (nm_loss * mask).sum() / pixels
    return gt_loss + nm_loss


def calc_multi_scale_loss(pred, target, max_disp, a=0.8, c=0.5, A=1, tile_size=1):
    scale = target.size(3) // pred.size(3)
    scale_disp = max(1, scale // tile_size)

    target = target / scale_disp
    max_disp = max_disp / scale_disp

    target = F.max_pool2d(target, kernel_size=scale, stride=scale)
    mask = (target < max_disp) & (target > 1e-3)
    diff = (pred - target).abs()

    if tile_size > 1 and scale_disp > 1:
        mask = (diff < A) & mask
    loss = robust_loss(diff, a=a, c=c)
    return (loss * mask).sum() / (mask.sum() + 1e-6)


def calc_slant_loss(dxy, dxy_gt, pred, target, max_disp, B=1, tile_size=1):
    scale = target.size(3) // pred.size(3)
    scale_disp = max(1, scale // tile_size)

    target = target / scale_disp
    max_disp = max_disp / scale_disp

    target, index = F.max_pool2d(
        target, kernel_size=scale, stride=scale, return_indices=True
    )
    mask = (target < max_disp) & (target > 1e-3)
    diff = (pred - target).abs()

    def retrieve_elements_from_indices(tensor, indices):
        flattened_tensor = tensor.flatten(start_dim=2)
        output = flattened_tensor.gather(
            dim=2, index=indices.flatten(start_dim=2)
        ).view_as(indices)
        return output

    dxy_gt = retrieve_elements_from_indices(dxy_gt, index.repeat(1, 2, 1, 1))

    mask = (diff < B) & mask
    loss = (dxy - dxy_gt).abs()
    return (loss * mask).sum() / (mask.sum() + 1e-6)


def calc_w_loss(w, pred, target, max_disp, C1=1, C2=1.5, tile_size=1):
    scale = target.size(3) // pred.size(3)
    scale_disp = max(1, scale // tile_size)

    target = target / scale_disp
    max_disp = max_disp / scale_disp

    target = F.max_pool2d(target, kernel_size=scale, stride=scale)
    mask = (target < max_disp) & (target > 1e-3)
    diff = (pred - target).abs()

    mask_c1 = (diff < C1) & mask
    loss_c1 = torch.clip(1 - w, min=0)
    loss_c1 = (loss_c1 * mask_c1).sum() / (mask_c1.sum() + 1e-6)

    mask_c2 = (diff > C2) & mask
    loss_c2 = torch.clip(w, min=0)
    loss_c2 = (loss_c2 * mask_c2).sum() / (mask_c2.sum() + 1e-6)
    return loss_c1 + loss_c2


def calc_loss(pred, batch, args):
    loss_dict = {}
    tile_size = pred.get("tile_size", 1)

    # multi scale loss
    for ids, d in enumerate(pred.get("multi_scale", [])):
        loss_dict[f"disp_loss_{ids}"] = calc_multi_scale_loss(
            d,
            batch["disp"],
            args.max_disp,
            a=args.robust_loss_a,
            c=args.robust_loss_c,
            A=args.HITTI_A,
            tile_size=tile_size,
        )

    # init loss
    for ids, cv in enumerate(pred.get("cost_volume", [])):
        loss_dict[f"init_loss_{ids}"] = calc_init_loss(
            cv,
            batch["disp"],
            args.max_disp,
            k=args.init_loss_k,
            tile_size=tile_size,
        )

    # slant loss
    for ids, (d, dxy) in enumerate(pred.get("slant", [])):
        loss_dict[f"slant_loss_{ids}"] = calc_slant_loss(
            dxy,
            batch["dxy"],
            d,
            batch["disp"],
            args.max_disp,
            B=args.HITTI_B,
            tile_size=tile_size,
        )

    # select loss
    for ids, sel in enumerate(pred.get("select", [])):
        w0, d0 = sel[0]
        w1, d1 = sel[1]
        loss_0 = calc_w_loss(
            w0,
            d0,
            batch["disp"],
            args.max_disp,
            C1=args.HITTI_C1,
            C2=args.HITTI_C2,
            tile_size=tile_size,
        )
        loss_1 = calc_w_loss(
            w1,
            d1,
            batch["disp"],
            args.max_disp,
            C1=args.HITTI_C1,
            C2=args.HITTI_C2,
            tile_size=tile_size,
        )
        loss_dict[f"select_loss_{ids}"] = loss_0 + loss_1

    return loss_dict
 
 
 
 
 
 
from kornia.filters.sobel import spatial_gradient, sobel 
 
    
class MultiScaleGradient(torch.nn.Module):
    def __init__(self, start_scale, num_scales, mask):
        super(MultiScaleGradient,self).__init__()
        print('Setting up Multi Scale Gradient loss...')

        self.start_scale = start_scale
        self.num_scales = num_scales

        self.multi_scales = [torch.nn.AvgPool2d(self.start_scale * (2**scale), self.start_scale * (2**scale)) for scale in range(self.num_scales)]
        print('Done')
        self.mask = mask.unsqueeze(1)
        

    def forward(self, prediction, target, preview = False):
        # helper to remove potential nan in labels
        #target = target.unsqueeze(1)
        #prediction = prediction.unsqueeze(1)
        #print(prediction.size())
        def nan_helper(y):
            return torch.isnan(y), lambda z: z.nonzero()[0]
        
        loss_value = 0
        loss_value_2 = 0
        diff = prediction - target
        #print(self.mask.size(), target.size(), prediction.size(), self.mask)
        _,_,H,W = target.shape
        upsample = torch.nn.Upsample(size=(2*H,2*W), mode='bicubic', align_corners=True)
        record = []

        for m in self.multi_scales:
            # input and type are of the type [B x C x H x W]
            if preview:
                record.append(upsample(sobel(m(diff))))
            else:
                # Use kornia spatial gradient computation
                delta_diff = spatial_gradient(m(diff))
                #is_nan = torch.logical_or(torch.isinf(delta_diff), torch.isnan(delta_diff))
                
                is_nan = torch.isnan(delta_diff)
                is_not_nan_sum = (~is_nan).sum()
                #is_not_nan = (~is_nan).sum() 
                # output of kornia spatial gradient is [B x C x 2 x H x W]
                loss_value += torch.abs(delta_diff[~is_nan]).sum()/is_not_nan_sum*target.shape[0]*2
                # * batch size * 2 (because kornia spatial product has two outputs).
                # replaces the following line to be able to deal with nan's.
                # loss_value += torch.abs(delta_diff).mean(dim=(3,4)).sum()

        if preview:
            return record
        else:
            return (loss_value/self.num_scales)
