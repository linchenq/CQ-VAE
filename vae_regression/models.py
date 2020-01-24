import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append("..") 

from utils.ops import ResEncoder, VAEDeconv

class BetaVAE(nn.Module):
    def __init__(self, in_ch=1, out_ch=176*2, latent_dims=64, beta=1.):
        super(BetaVAE, self).__init__()
        self.latent_dims = latent_dims
        self.beta = beta
        
        self.encoder = ResEncoder(in_ch=in_ch)
        self.decoder = nn.Sequential(
            VAEDeconv(512, 256, 2, 2, 0),
            VAEDeconv(256, 128, 2, 2, 0)
        )
        self.dec_dense = nn.Linear(128*8*16, 4096)
        self.regression = nn.Linear(4096, out_ch)
        
        self.fc_mu = nn.Linear(4096, self.latent_dims)
        self.fc_var = nn.Linear(4096, self.latent_dims)
        self.fc_z = nn.Linear(self.latent_dims, 4096)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)       # logvar = log(std^2)
        eps = torch.randn_like(std)         # eps ~ N(0,1)
        return (mu + std * eps)
    
    def encode(self, x):
        enc = self.encoder(x)
        enc = enc.view(-1, 4096)
        return self.fc_mu(enc), self.fc_var(enc)
    
    def decode(self, x):
        z = self.relu(self.fc_z(x))
        z = z.view(-1, 512, 2, 4)
        z = self.decoder(z)
        
        z = z.view(-1, 128*8*16)
        z = self.relu(self.dec_dense(z))
        z = self.regression(z)
        
        z = z.view(-1, 176, 2)
        return z
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        pts = self.decode(z)
            
        return pts, mu, logvar
    
    def loss(self, pts, gt, mu, logvar):
        # recon_loss = F.binary_cross_entropy(xr, x, reduction='sum')
        recon_loss = F.mse_loss(pts, gt, reduction='sum')
        
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        return (recon_loss + self.beta * kld)
        

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BetaVAE(in_ch=1, out_ch=176*2, latent_dims=64, beta=1.)
    model = model.to(device)
    if True:
        from torchsummary import summary
        summary(model, input_size=(1, 64, 128))
    
    if True:
        from torch.autograd import Variable
        img = Variable(torch.rand(2, 1, 64, 128))
        img = img.to(device)
        xr, mu, logvar = model(img)
        print(xr.shape, mu.shape, logvar.shape)