import torch
import torch.nn as nn
from torchsummary import summary

from utils.ops import VAEConv, VAEDeconv

class BetaVAE(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, latent_dims=64, beta=1.):
        super(BetaVAE, self).__init__()
        self.latent_dims = latent_dims
        self.beta = beta
        
        self.encoder = nn.Sequential(
            VAEConv(in_ch, 32),
            VAEConv(32, 32),
            VAEConv(32, 64),
            VAEConv(64, 64),
        )
        
        self.fc_mu = nn.Linear(2048, self.latent_dims)
        self.fc_var = nn.Linear(2048, self.latent_dims)
        
        self.decoder = nn.Sequential(
            VAEDeconv(64, 64),
            VAEDeconv(64, 32),
            VAEDeconv(32, 32),
            VAEDeconv(32, out_ch),
        )
        self.fc_z = nn.Linear(self.latent_dims, 2048)
        
    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps
    
    def encode(self, x):
        enc = self.encoder(x)
        enc = enc.view(-1, 256)
        return self.fc_mu(x), self.fc_var(x)
    
    def decode(self, x):
        z = self.fc_z(x)
        z = z.view(-1, 64, 8, 4)
        z = self.decoder(z)
        return z
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        xr = self.decode(z)
        return xr, mu, logvar

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BetaVAE(in_ch=1, out_ch=1, latent_dims=64, beta=1.)
    model = model.to(device)
    
    summary(model, input_size=(1, 128, 64))