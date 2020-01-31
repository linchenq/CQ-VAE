import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.ops import ResEncoder, ResDecoder

class DiscreteVAE(nn.Module):
    
    def __init__(self, in_ch=1,
                       out_ch=176*2,
                       latent_dims=64,
                       vec_dims=11,
                       beta=1.,
                       temperature=1.,
                       device=None
                ):
        
        super(DiscreteVAE, self).__init__()
        
        self.latent_dims = latent_dims
        self.vec_dims = vec_dims
        
        self.temperature = temperature
        self.beta = beta
        self.device = device
        
        # encoder
        self.encoder = ResEncoder(in_ch=in_ch)
        self.fc_enc = nn.Linear(512*2*4, latent_dims*vec_dims)
        
        # decoder
        self.fc_dec = nn.Linear(latent_dims*vec_dims, 512*2*4)
        self.decoder = ResDecoder(layers=[512, 256, 128])
        
        self.fc_dense1 = nn.Linear(128*8*16, 4096)
        self.fc_dense2 = nn.Linear(4096, 1024)
        self.fc_reg = nn.Linear(1024, out_ch)
        
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
        
        out = self.softmax(pi / self.temperature)
        out = out.view(-1, self.latent_dims*self.vec_dims)
        return out
    
    def encode(self, x):
        enc = self.encoder(x)
        enc = enc.view(-1, 512*2*4)
        enc = self.fc_enc(enc)
        
        logits = enc.view(-1, self.latent_dims, self.vec_dims)
        qy = self.softmax(logits)
        
        return logits, qy

    def decode(self, x):
        dec = self.relu(self.fc_dec(x))
        dec = dec.view(-1, 512, 2, 4)
        
        dec = self.decoder(dec)
        
        dec = dec.view(-1, 128*8*16)
        dec = self.relu(self.fc_dense1(dec))
        dec = self.relu(self.fc_dense2(dec))
        dec = self.fc_reg(dec)
        
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DiscreteVAE(in_ch=1, out_ch=176*2, latent_dims=64, vec_dims=11, temperature=1., beta=1., device=device)
    model = model.to(device)
    if True:
        from torchsummary import summary
        summary(model, input_size=(1, 64, 128))
    
    if True:
        from torch.autograd import Variable
        img = Variable(torch.rand(2, 1, 64, 128))
        img = img.to(device)
        pts, qy = model(img)
        print(pts.shape, qy.shape)
        
        gt = Variable(torch.rand(*pts.size()))
        gt = gt.cuda()
        loss = model.loss(pts, gt, qy)
        print(loss)