import torch
import torch.nn as nn
import torch.nn.functional as F

class DiscreteLoss(nn.Module):
    '''
    alpha: linear variable for  segmentation loss
    beta: linear variable for kl_divergence
    '''
    def __init__(self, alpha, beta, device, eps=1e-20):
        super(DiscreteLoss, self).__init__()
        
        self.alpha = alpha
        self.beta = beta
        self.mark_index = [0, 29, 88, 117]
        
        self.device = device
        self.eps = eps
        
    def forward(self,
                pts, pts_gt,
                mask, mask_gt,
                qy, vector_dims):
        
        reg = self.regression_loss(pts, pts_gt)
        kld = self.kld_loss(qy, vector_dims)
        seg = self.segmentation_loss(mask, mask_gt)
        
        ret = reg + self.beta * kld + self.alpha * seg
        
        return ret * pts.shape[0]
    
    def regression_loss(self, pts, pts_gt):
        disk = F.mse_loss(pts, pts_gt, reduction='sum') / pts.shape[0]
        landmark = F.mse_loss(pts[:, self.mark_index], pts_gt[:, self.mark_index], reduction='sum') / pts.shape[0]
        
        return landmark + disk
        
    def kld_loss(self, qy, vector_dims):
        log_qy = torch.log(qy + self.eps)
        g = torch.log(torch.Tensor([1.0 / vector_dims])).to(self.device)
        kld = torch.sum(qy * (log_qy - g), dim=-1).mean()
        
        return kld
    
    def segmentation_loss(self, mask, mask_gt):
        mask, mask_gt = mask[..., 32:96, :], mask_gt[..., 32:96, :]
        return F.mse_loss(mask, mask_gt)
        
