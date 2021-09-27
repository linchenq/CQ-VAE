import torch
import torch.nn as nn
# import torch.nn.functional as F

from utils.ops import DiscreteEncoder, ShapeDist, BackLoopEnc, ShapeEst

'''
I -> Encoder -> P(Z) -gumbel-softmax sample-> z(one-hot) -> decoder -> regression points
                 |                                           ^                |
                 v                                           | <--- enc <----
                 z_mean -> seperated_decoder -> shape estimation
'''

class CQVAE(nn.Module):
    ''' Encoder Structure: ResEncoder -> flatten -> discrete distribution -> logits
                                                                              | softmax
                                                                             qy
        
        # ResEncoder: (batch_size, 1, 128, 128) -> (batch_size, 512, 4, 4)
        # Discrete: (batch_size, 512 * 4 * 4) -> (batch_size, lat_dims * vec_dims)
        
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
                       latent_dims=64,
                       vector_dims=11,
                       
                       tau=1.,
                       device=None,
                       
                       num_sample=128):
        
        super(CQVAE, self).__init__()
        
        self.latent_dims = latent_dims
        self.vector_dims = vector_dims
        self.tau = tau
        self.device = device
        self.num_sample = num_sample
        
        # model
        self.encoder = DiscreteEncoder(in_ch=in_channels)
        self.discrete = nn.Linear(512*4*4, latent_dims * vector_dims)
        self.dist = ShapeDist(out_ch=out_channels, lat_dim=latent_dims, vec_dim=vector_dims)
        self.est = ShapeEst(out_ch=out_channels, lat_dim=latent_dims, vec_dim=vector_dims)
        self.backloop = BackLoopEnc(in_ch=out_channels, lat_dim=latent_dims, vec_dim=vector_dims)
        
        self.softmax = nn.Softmax(dim=-1)
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
        enc = enc.view(-1, 512*4*4)
        z = self.discrete(enc)
        z = self.sigmoid(z)
                
        logits = z.view(-1, self.latent_dims, self.vector_dims)
        qy = self.softmax(logits)
        
        return logits, qy
    
    def shape_dist(self, x):
        # x = x.view(-1, self.latent_dims * self.vector_dims)
        pt = self.dist(x)
        rz = self.backloop(pt)
        pt = pt.view(-1, 176, 2)
        
        return pt, rz
    
    def shape_est(self, x):
        x = x.view(-1, self.latent_dims * self.vector_dims)
        best_pt = self.est(x)
        best_pt = best_pt.view(-1, 176, 2)
        
        return best_pt
    
    def forward(self, x):
        pts, zs, rzs = [], [], []
        logits, qy = self.encode(x)
                
        # sampling: generate fixed number of x
        for i in range(self.num_sample):
            z = self.reparametrize(logits)
            pt, rz = self.shape_dist(z)
            zs, pts, rzs = zs + [z], pts + [pt], rzs + [rz]
        
        zs = torch.stack(zs, dim=0).permute(1, 0, 2, 3)
        rzs = torch.stack(rzs, dim=0).permute(1, 0, 2, 3)
        pts = torch.stack(pts, dim=0).permute(1, 0, 2, 3)
        
        best = self.shape_est(logits)
        
        return (zs, rzs, pts, best), (logits, qy)


if __name__ == '__main__':
    debug, summary = True, False
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = CQVAE(in_channels=1,
                  out_channels=176*2,
                  latent_dims=64,
                  vector_dims=11,

                  tau=5.,
                  device=device,
                  num_sample=32)
    
    model = model.to(device)
    
    if summary:
        from torchsummary import summary
        summary(model, input_size=(1, 128, 128))
    
    if debug:
        from torch.autograd import Variable
        img = Variable(torch.rand(3, 1, 128, 128)).to(device)
        output, gs_logits = model(img)
        (zs, rzs, pts, best), (logits, qy) = output, gs_logits
        print(f"tensor shape: {pts.shape}, {rzs.shape}, {zs.shape}, {best.shape}")
        print(f"tensor shape: {logits.shape}, {qy.shape}")
        
        from utils.loss import CQVAELoss
        best_gt = Variable(torch.rand(3, 176, 2)).to(device)
        gts = Variable(torch.rand(3, 16, 176, 2)).to(device)
        model_loss = CQVAELoss(alpha=1.0, beta=1.0, gamma=1.0, device=device)
        loss, loss_dict = model_loss.forward(output, gs_logits, gts, best_gt, 11, True, True, True, True, True)
        
        print(f"loss is {loss}")
        print(f"detail is {loss_dict}")
        