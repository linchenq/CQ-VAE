import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.ops import ResEncoder, ResDecoder

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
                       
                       latent_dims=64,
                       vector_dims=11,
                       
                       beta=1.,
                       tau=1.,
                       device=None
                ):
        
        super(DiscreteVAE, self).__init__()
        
        self.latent_dims = latent_dims
        self.vector_dims = vector_dims
        
        self.tau = tau
        self.beta = beta
        self.device = device
        
        self.encoder = ResEncoder(in_ch=in_channels)
        self.discrete = nn.Linear(4096, latent_dims * vector_dims)
        
        self.decoder = MLPDecoder(in_ch=latent_dims * vector_dims, out_ch=out_channels)
        
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
        enc = x.view(-1, 4096)
        enc = self.discrete(enc)
        
        logits = enc.view(-1, self.latent_dims, self.vector_dims)
        qy = self.softmax(logits)
        
        return logits, qy

    def decode(self, x):
        x = x.view(-1, self.latent_dims * self.vector_dims)
        dec = self.decoder(x)
        dec = dec.view(-1, 176, 2)
        
        return dec
    
    def forward(self, x):
        logits, qy = self.encode(x)
        z = self.reparametrize(logits)
        pts = self.decode(z)
        
        return pts, qy
    
    def loss(self, pts, gt, qy, eps=1e-20):
        recon_loss = F.mse_loss(pts, gt, reduction='sum') / pts.shape[0]
        
        log_qy = torch.log(qy + eps)
        g = torch.log(torch.Tensor([1.0/self.vec_dims])).to(self.device)
        kld = torch.sum(qy * (log_qy - g), dim=-1).mean()
        
        return (recon_loss + self.beta * kld) * pts.shape[0]
        

if __name__ == '__main__':
    debug = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = DiscreteVAE(in_ch=1, out_ch=176*2, latent_dims=64, vec_dims=11, temperature=1., beta=1., device=device)
    model = model.to(device)
    
    if debug:
        from torchsummary import summary
        summary(model, input_size=(1, 64, 128))
    
    if debug:
        from torch.autograd import Variable
        img = Variable(torch.rand(2, 1, 64, 128))
        img = img.to(device)
        pts, qy = model(img)
        print(pts.shape, qy.shape)
        
        gt = Variable(torch.rand(*pts.size()))
        gt = gt.cuda()
        loss = model.loss(pts, gt, qy)
        print(loss)