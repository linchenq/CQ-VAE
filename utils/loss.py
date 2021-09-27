import torch
import torch.nn as nn
import torch.nn.functional as F


class CQVAELoss(nn.Module):
    def __init__(self, alpha, beta, gamma, device, eps=1e-20):
        super(CQVAELoss, self).__init__()
        
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
        self.mark_index = [0, 29, 88, 117]
        
        self.device = device
        self.eps = eps
    
    def forward(self, output, gs_logits, gts, best_gt, vector_dims,
                      kld=True, best_bool=True, autoe=True, regress=True, mark=False):
        
        (zs, rzs, pts, best), (logits, qy) = output, gs_logits
        
        # generate mapping function
        from utils.util import _batch_matching_shapes
        mapping = _batch_matching_shapes(pts, gts)
        
        # kl-divergence
        kld_loss = self._kld(qy, vector_dims) if kld else torch.tensor(0)
        
        # MSE_loss between best shape and its ground truth
        best_mse = self._bias(best, best_gt, mark) if best_bool else torch.tensor(0)
        
        # autoencoder reconstruction error
        # MSE_loss between generated shapes and ground truths' linear combination
        ae_loss, bias_loss = self._bias_mapping(rzs, zs, pts, gts, mapping, autoe, regress, mark)
        
        loss_ret = self.beta * kld_loss + self.gamma * ae_loss + best_mse + bias_loss
        
        loss_dict = {
            'KLD': kld_loss.item(),
            'AELoss': ae_loss.item(),
            'BIASLoss': bias_loss.item(),
            'BESTMse': best_mse.item()
            }
        
        return loss_ret, loss_dict
      
    def _bias(self, pts, gt, mark):
        disk_bias = F.mse_loss(pts, gt)
        mark_bias = F.mse_loss(pts[..., self.mark_index, :], gt[..., self.mark_index, :]) if mark else torch.tensor(0)
        
        return (disk_bias + self.alpha * mark_bias)
    
    def _bias_mapping(self, rzs, zs, pts, gts, mapping,
                            autoe, regress, mark):
        # mean mode
        auto_loss, bias_loss = 0, 0
        
        for b_i in range(gts.shape[0]):
            for i in range(gts.shape[1]):
                
                # autoencoder reconstruction error
                auto_loss += self._aeloss(rzs[b_i, mapping[b_i, i], ...], zs[b_i, i, ...]) \
                    if autoe else torch.tensor(0)
                        
                # MSE_loss between generated shapes and ground truths' linear combination   
                bias_loss += self._bias(pts[b_i, mapping[b_i, i], ...], gts[b_i, i, ...], mark) \
                    if regress else torch.tensor(0)
        
        auto_loss /= (gts.shape[0] * gts.shape[1])
        bias_loss /= (gts.shape[0] * gts.shape[1])
                       
        return auto_loss, bias_loss 
    
    def _kld(self, qy, vector_dims):
        log_qy = torch.log(qy + self.eps)
        g = torch.log(torch.Tensor([1.0 / vector_dims])).to(self.device)
        kld_loss = torch.sum(qy * (log_qy - g), dim=-1).mean()
        
        return kld_loss
    
    def _aeloss(self, zs, rzs):
        return F.mse_loss(zs, rzs)
