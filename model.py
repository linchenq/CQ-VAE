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
                       tau=1.,
                       device=None
                ):
        
        super(DiscreteVAE, self).__init__()
        
        self.latent_dims = latent_dims
        self.vector_dims = vector_dims
        
        self.alpha = alpha
        self.beta = beta
        self.tau = tau
        self.device = device
        
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
        
        logits = enc.view(-1, self.latent_dims, self.vector_dims)
        qy = self.softmax(logits)
        
        return logits, qy

    def decode(self, x):
        x = x.view(-1, self.latent_dims * self.vector_dims)
        reg, seg = self.decoder(x)
        rz = self.autoencoder(reg)
        reg = reg.view(-1, 176, 2)
        
        return reg, seg, rz
    
    def forward(self, x):
        logits, qy = self.encode(x)
        z = self.reparametrize(logits)
        pts, mask, rz = self.decode(z)
        best, _, _ = self.decode(logits)
        
        return pts, mask, qy, z, rz, best


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
                        tau=1.,
                        device=device)
    
    model = model.to(device)
    model_loss = DiscreteLoss(alpha=model.alpha, beta=model.beta, device=device)
    
    if summary:
        from torchsummary import summary
        summary(model, input_size=(1, 128, 128))
    
    if debug:
        from torch.autograd import Variable
        img = Variable(torch.rand(2, 1, 128, 128))
        img = img.to(device)
        pts, mask, qy, z, rz, best = model(img)
        print(pts.shape, mask.shape, qy.shape, z.shape, rz.shape, best.shape)
        
        pts_gt = Variable(torch.rand(*pts.size()))
        pts_gt = pts_gt.cuda()
        mask_gt = Variable(torch.rand(*mask.size()))
        mask_gt = mask_gt.cuda()
        loss = model_loss.forward(pts, pts_gt, mask, mask_gt, qy, model.vector_dims)
        print(loss)