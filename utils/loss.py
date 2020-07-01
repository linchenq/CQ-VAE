import torch
import torch.nn as nn
import torch.nn.functional as F

import utils.util as uts

class CQLoss(nn.Module):
    '''
    alpha: linear variable for  segmentation loss
    beta: linear variable for kl_divergence
    '''
    def __init__(self, alpha, beta, gamma, device, eps=1e-20):
        super(CQLoss, self).__init__()
        
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.mark_index = [0, 29, 88, 117]
        
        self.device = device
        self.eps = eps
        
    def forward(self, zs, decs, qy, logits, best,
                      pts_gt, best_gt, vector_dims):
        pts, rzs = decs
        mapping = uts.batch_match_shape(pts, pts_gt)
        
        # kl-divergence
        kld = self.kld(qy, vector_dims)
        
        # shape-regression
        best_pts_loss = self.regression(best, best_gt)
        recon, pts_loss = self.regression_wmap(rzs, zs, pts, pts_gt, mapping)
        
        # ret loss
        ret = self.beta * kld + \
              self.gamma * recon + pts_loss + \
              best_pts_loss
        
        ret_dict = {
            "total_loss": ret.item(),
            "regression_loss": pts_loss.item(),
            "reconstruction_loss": recon.item(),
            "kl_divergence_loss": kld.item(),
            "best_regression_loss": best_pts_loss.item(),
        }

        return ret, uts.dict_mul(ret_dict, pts.shape[0])
    
    def regression(self, x, gt):
        # regression loss
        disk = F.mse_loss(x, gt)
        landmark = F.mse_loss(x[..., self.mark_index, :], gt[..., self.mark_index, :])
        pts = disk + self.alpha * landmark
        
        return pts
     
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
        ret_pts = ret_disk + self.alpha * ret_landmark
        
        return ret_auto, ret_pts
    
    def kld(self, qy, vector_dims):
        log_qy = torch.log(qy + self.eps)
        g = torch.log(torch.Tensor([1.0 / vector_dims])).to(self.device)
        kld_loss = torch.sum(qy * (log_qy - g), dim=-1).mean()
        
        return kld_loss
    
