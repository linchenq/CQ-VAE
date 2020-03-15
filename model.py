import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.ops import DiscreteEncoder, DiscreteDecoder, RegressAutoEncoder
from utils.loss import DiscreteLoss

'''
I -> Encoder -> P(Z) -gumbel-softmax sample-> z(one-hot) -> decoder -> segmentation mask
                 |                                              |
                 v                                           ^  -----------> regression points
                 z_mean                                      |                   |
                                                               <---- enc <------
'''

class DiscreteVAE(nn.Module):
    ''' Encoder Structure: ResEncoder -> flatten -> discrete distribution -> logits
                                                                              | softmax
                                                                             qy
        
        # ResEncoder: (batch_size, 1, 64, 128) -> (batch_size, 512, 2, 4)
        # Discrete: (batch_size, 512 * 2 * 4) -> (batch_size, lat_dims * vec_dims)
        
    '''
        
    ''' Sample and decoder Structure: logits -(gumbel softmax sampling)-> decoder 
                            | softmax
                           qy -> kl-divergence between U(0,1) and softmaxed-logits
                           |
                          log_qy
        # logits: (batch_size, lat_dims * vec_dims)
    '''
    def __init__(self, in_channels=1,
                       out_channels=176*2,
                       seg_channels=1,
                       
                       latent_dims=64,
                       vector_dims=11,
                       
                       alpha=1.,
                       beta=1.,
                       gamma=1.,
                       
                       tau=1.,
                       device=None,
                       
                       sample_step=128
                ):
        
        super(DiscreteVAE, self).__init__()
        
        self.latent_dims = latent_dims
        self.vector_dims = vector_dims
        
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
        self.tau = tau
        self.device = device
        
        self.sample_step = sample_step
        
        self.encoder = DiscreteEncoder(in_ch=in_channels)
        self.discrete = nn.Linear(8192, latent_dims * vector_dims)
        
        self.decoder = DiscreteDecoder(in_ch=latent_dims * vector_dims, out_ch=out_channels, seg_ch=seg_channels)
        self.autoencoder = RegressAutoEncoder(in_ch=out_channels, latent_dims=latent_dims, vector_dims=vector_dims)
        
        self.softmax = nn.Softmax(dim=-1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def gumbel_sample(self, shape, eps=1e-20):
        ''' Sample from Gumbel(0, 1) '''
        U = torch.rand(shape)
        return -torch.log(-torch.log(U + eps) + eps)
    
    def reparametrize(self, logits):
        ''' Draw a sample from the Gumbel-Softmax distribution '''
        sample = self.gumbel_sample(logits.size())
        sample = sample.to(self.device)
        pi = logits + sample
        
        out = self.softmax(pi / self.tau)
        return out
    
    def encode(self, x):
        enc = self.encoder(x)
        enc = enc.view(-1, 8192)
        enc = self.discrete(enc)
        enc = self.sigmoid(enc) 
        
        logits = enc.view(-1, self.latent_dims, self.vector_dims)
        qy = self.softmax(logits)
        
        return logits, qy

    def decode(self, x):
        x = x.view(-1, self.latent_dims * self.vector_dims)
        reg, seg = self.decoder(x)
        rz = self.autoencoder(reg)
        reg = reg.view(-1, 176, 2)
        
        return reg, seg, rz
    
    def forward(self, x, step=None):
        '''
            zs: list of z, sampled z from P(Z)
            decs: list of tuple(pts, mask, rz), regarding loss
        '''
        zs, regs, segs, rzs = [], [], [], []
        logits, qy = self.encode(x)
        
        if step is None:
            step = self.sample_step
        
        '''
        SAMPLING METHOD:
            [1] x: drop with prob rate; gt: generate matched lr ground truth with the same amount
            [2] x: generate fixed number of x; gt: generate half amount of gt
        '''
        
        '''
        # # [1] SAMPLING METHOD
        # for i in range(step):
        #     if torch.rand(1) > 0.5:
        #         z = self.reparametrize(logits)
        #         reg, seg, rz = self.decode(z)
        #         zs, regs, segs, rzs = zs + [z], regs + [reg], segs + [seg], rzs + [rz]
        '''
        
        # [2] SAMPLING METHOD
        for i in range(step):
            z = self.reparametrize(logits)
            reg, seg, rz = self.decode(z)
            zs, regs, segs, rzs = zs + [z], regs + [reg], segs + [seg], rzs + [rz]
        
        zs, rzs = torch.stack(zs, dim=0).permute(1, 0, 2, 3), torch.stack(rzs, dim=0).permute(1, 0, 2, 3)
        regs= torch.stack(regs, dim=0).permute(1, 0, 2, 3)
        segs = torch.stack(segs, dim=0).permute(1, 0, 2, 3, 4).squeeze(dim=2)
        decs = (regs, segs, rzs)
        
        best_reg, best_seg, best_rz = self.decode(logits)
        best_seg = best_seg.squeeze(dim=1)
        best = (best_reg, best_seg, best_rz)
        
        return zs, decs, qy, logits, best


if __name__ == '__main__':
    debug, summary = True, False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = DiscreteVAE(in_channels=1,
                        out_channels=176*2,
                        seg_channels=1,
                        
                        latent_dims=64,
                        vector_dims=11,
                        
                        alpha=1.,
                        beta=1.,
                        gamma=1.,
                        
                        tau=3.,
                        device=device,
                        sample_step=512)
    
    model = model.to(device)
    model_loss = DiscreteLoss(alpha=model.alpha,
                              beta=model.beta,
                              gamma=model.gamma,
                              device=device)
    
    if summary:
        from torchsummary import summary
        summary(model, input_size=(1, 128, 128))
    
    if debug:
        from torch.autograd import Variable
        img = Variable(torch.rand(2, 1, 128, 128)).to(device)
        zs, decs, qy, logits, best = model(img)
        pts, masks, zrs = decs
        print(f"tensor shape: {pts.shape}, {masks.shape}, {zrs.shape}, {zs.shape}")
        