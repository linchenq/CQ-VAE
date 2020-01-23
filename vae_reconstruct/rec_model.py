import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.ops import VAEConv, VAEDeconv

class BetaVAECon(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, latent_dims=256, beta=1.):
        super(BetaVAECon, self).__init__()
        self.latent_dims = latent_dims
        self.beta = beta
        
        self.encoder = nn.Sequential(
            VAEConv(1, 32),
            VAEConv(32, 32),
            VAEConv(32, 64),
            VAEConv(64, 64)
        )
        self.decoder = nn.Sequential(
            VAEDeconv(64, 64, 4, 2, 1),
            VAEDeconv(64, 32, 4, 2, 1),
            VAEDeconv(32, 32, 4, 2, 1),
            VAEDeconv(32, 1, 4, 2, 1)
        )

        self.fc_mu = nn.Linear(2048, self.latent_dims)
        self.fc_var = nn.Linear(2048, self.latent_dims)
        self.fc_z = nn.Linear(self.latent_dims, 2048)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)       # logvar = log(std^2)
        eps = torch.randn_like(std)         # eps ~ N(0,1)
        return (mu + std * eps)
    
    def encode(self, x):
        enc = self.encoder(x)
        enc = enc.view(-1, 2048)
        return self.fc_mu(enc), self.fc_var(enc)
    
    def decode(self, x):
        z = self.relu(self.fc_z(x))
        z = z.view(-1, 64, 4, 8)
        z = self.decoder(z)
        return self.sigmoid(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        pts = self.decode(z)
            
        return pts, mu, logvar
    
    def loss(self, xr, x, mu, logvar):
        recon_loss = F.binary_cross_entropy(xr, x, reduction='sum')
        # recon_loss = F.mse_loss(xr, x, reduction='sum')
        
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        return (recon_loss + self.beta * kld)
        

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BetaVAECon(in_ch=1, out_ch=1, latent_dims=64, beta=1.)
    model = model.to(device)
    if False:
        from torchsummary import summary
        summary(model, input_size=(1, 64, 128))
    
    if True:
        from torch.autograd import Variable
        img = Variable(torch.rand(2, 1, 64, 128))
        img = img.to(device)
        xr, mu, logvar = model(img)
        print(xr.shape, mu.shape, logvar.shape)