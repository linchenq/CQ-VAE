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
        mapping = uts.batch_match_shape(pts, pts_gt)
        
        kld = self.kld(qy, vector_dims)
        
        seg = self.segmentation_wmap(masks, masks_gt, mapping)
        # seg = self.segmentation(masks, masks_gt)
        best_seg = self.segmentation(best_mask, best_mask_gt)
        
        auto, reg = self.regression_wmap(rzs, zs, pts, pts_gt, mapping)
        # auto, reg = self.regression(rzs, zs, pts, pts_gt)
        best_auto, best_reg = self.regression(best_rz, logits, best_pt, best_pt_gt)
        
        ret = self.gamma * (best_reg + best_auto + self.alpha * best_seg) +\
                (reg + auto + self.alpha * seg) + \
                self.beta * kld
                
        # ret = self.gamma * (best_reg + self.alpha * best_seg) +\
        #         (reg + self.alpha * seg) + \
        #         self.beta * kld
        
        ret_dict = {
            "total_loss": ret.item(),
            "regression_loss": reg.item(),
            "autoencoder_loss": auto.item(),
            "segmentation_loss": seg.item(),
            "kl_divergence_loss": kld.item(),
            "best_regression_loss": best_reg.item(),
            "best_autoencoder_loss": best_auto.item(),
            "best_segmentation_loss": best_seg.item()
        }

        return ret, uts.dict_mul(ret_dict, pts.shape[0])
    
    def regression(self, rz, z, x, gt):
        # autoencoder loss
        auto = F.mse_loss(rz, z)
        
        # regression loss
        disk = F.mse_loss(x, gt)
        landmark = F.mse_loss(x[..., self.mark_index, :], gt[..., self.mark_index, :])
        
        return auto, (disk + landmark)
     
    def regression_wmap(self, rz, z, x, gt, mapping):
        if x.shape[0] != gt.shape[0] or rz.shape[0] != z.shape[0] or x.shape[0] != z.shape[0]:
            raise ValueError('Unmatched batch size for regression loss')
        ret_auto, ret_disk, ret_landmark = 0, 0, 0
        
        for batch_i in range(gt.shape[0]):
            for len_i in range(gt.shape[1]):                
                # autoencoder loss
                ret_auto += F.mse_loss(rz[batch_i, mapping[batch_i, len_i], ...],
                                       z[batch_i, len_i, ...])
                # regression loss
                ret_disk += F.mse_loss(x[batch_i, mapping[batch_i, len_i], ...],
                                       gt[batch_i, len_i, ...])
                
                ret_landmark += F.mse_loss(x[batch_i, mapping[batch_i, len_i], self.mark_index, :],
                                           gt[batch_i, len_i, self.mark_index, :])
        
        ret_auto /= (gt.shape[0] * gt.shape[1])
        ret_disk /= (gt.shape[0] * gt.shape[1])
        ret_landmark /= (gt.shape[0] * gt.shape[1])
        
        return ret_auto, (ret_disk + ret_landmark)
    
    def kld(self, qy, vector_dims):
        log_qy = torch.log(qy + self.eps)
        g = torch.log(torch.Tensor([1.0 / vector_dims])).to(self.device)
        kld_loss = torch.sum(qy * (log_qy - g), dim=-1).mean()
        
        return kld_loss
    
    def segmentation(self, x, gt):
        x, gt = x[..., 32:96, :], gt[..., 32:96, :]
        return F.mse_loss(x, gt)
    
    def segmentation_wmap(self, x, gt, mapping):
        if x.shape[0] != gt.shape[0]:
            raise ValueError('Unmatched batch size for segmentation loss')
        ret_loss = 0
        
        for batch_i in range(gt.shape[0]):
            for len_i in range(gt.shape[1]):
                ret_loss += F.mse_loss(x[batch_i, mapping[batch_i, len_i], ...],
                                       gt[batch_i, len_i, ...])
        
        ret_loss /= (gt.shape[0] * gt.shape[1])
       
        return ret_loss     
        
