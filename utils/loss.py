import torch
import torch.nn as nn
import torch.nn.functional as F

import utils.util as uts

class DiscreteLoss(nn.Module):
    '''
    alpha: linear variable for  segmentation loss
    beta: linear variable for kl_divergence
    '''
    def __init__(self, alpha, beta, gamma, device, eps=1e-20):
        super(DiscreteLoss, self).__init__()
        
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.mark_index = [0, 29, 88, 117]
        
        self.device = device
        self.eps = eps
        
    def forward(self, zs, decs, qy, logits, best,
                pts_gt, masks_gt, best_pt_gt, best_mask_gt, vector_dims):
        '''
        kld: kl divergence, control entropy of P(Z), function(qy, vector_dims)
        seg: segmentation loss, take mse of partial mask, function(masks, masks_gt), function(best_mask, best_masks_gt)
        reg: regression loss, contains 2 parts:
                (1) autoencoder loss for z and rz, function(rzs, zs), function(best_rz, logits)
                (2) regression loss on points, function(pts, pts_gt), function(best_pt, best_pt_gt)
        '''
        pts, masks, rzs = decs
        best_pt, best_mask, best_rz = best
        
        kld = self.kld_loss(qy, vector_dims)
        
        seg = self.segmentation_loss(masks, masks_gt)
        best_seg = self.segmentation_loss(best_mask, best_mask_gt)
        
        auto, reg = self.regression_loss(rzs, zs, pts, pts_gt)
        best_auto, best_reg = self.regression_loss(best_rz, logits, best_pt, best_pt_gt)
        
        ret = self.gamma * (best_reg + best_auto + self.alpha * best_seg) +\
                (reg + auto + self.alpha * seg) + \
                self.beta * kld
        
        ret_dict = {
            "regression_loss": reg.item(),
            "autoencoder_loss": auto.item(),
            "segmentation_loss": seg.item(),
            "kl_divergence_loss": kld.item(),
            "best_regression_loss": best_reg.item(),
            "best_autoencoder_loss": best_auto.item(),
            "best_segmentation_loss": best_seg.item()
        }

        return ret * pts.shape[0], uts.dict_mul(ret_dict, pts.shape[0])
    
    def regression_loss(self, rz, z, x, gt):
        # autoencoder loss
        auto = F.mse_loss(rz, z, reduction="mean")
        
        # regression loss
        disk = F.mse_loss(x, gt, reduction="sum") / (x.shape[0] * x.shape[1])
        landmark = F.mse_loss(x[..., self.mark_index, :], gt[..., self.mark_index, :], reduction="sum") / (x.shape[0] * x.shape[1])
        
        return auto, (disk + landmark)
        
    def kld_loss(self, qy, vector_dims):
        log_qy = torch.log(qy + self.eps)
        g = torch.log(torch.Tensor([1.0 / vector_dims])).to(self.device)
        kld = torch.sum(qy * (log_qy - g), dim=-1).mean()
        
        return kld
    
    def segmentation_loss(self, x, gt):
        x, gt = x[..., 32:96, :], gt[..., 32:96, :]
        return F.mse_loss(x, gt, reduction="mean")
        
