import torch
import torch.nn as nn
from torchvision import models

class DiscreteEncoder(nn.Module):
    def __init__(self, in_ch):  
        super(DiscreteEncoder, self).__init__()
        self.resnet = models.resnet34(pretrained=False)
        self.enc0 = nn.Sequential(
            nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.GroupNorm(32, 64),
            nn.ReLU(inplace=True)
        )
        self.downsample = nn.Conv2d(64, 64, kernel_size=2, stride=2, padding=0, bias=False)
        
        self.enc1 = self.resnet.layer1
        self.enc2 = self.resnet.layer2
        self.enc3 = self.resnet.layer3 
        self.enc4 = self.resnet.layer4
            
    def forward(self, x):
        out = self.downsample(self.enc0(x))
        out = self.enc1(out)
        out = self.enc2(out)
        out = self.enc3(out)
        out = self.enc4(out)
        
        return out


# Problistic Path for Shape Distribution
class ShapeDist(nn.Module):
    def __init__(self, out_ch, lat_dim=64, vec_dim=11):
        super(ShapeDist, self).__init__()
        C = torch.linspace(-1, 1, vec_dim).view(1, 1, vec_dim)
        self.register_buffer('C', C)
        
        # upsampling
        planes = [UpsDist(lat_dim, 128, 1, 5, 1, 0)]
        for ch in [64, 32, 16]:
            planes.append(UpsDist(ch*2, ch, ch, 5, 2, 2, 1))
        self.ups = nn.Sequential(*planes)
        
        # regression
        self.fcs = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16*40*40, 6400),
            nn.ReLU(inplace=True),
            nn.Linear(6400, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, out_ch)
        )        
        
    def forward(self, x):
        z = torch.sum(x * self.C, dim=2)
        z = z.view(z.size(0), -1, 1, 1)
        
        dec = self.ups(z)
        dec = self.fcs(dec)
        
        return dec


# Backward encoder looped with **ShapeDist**
class BackLoopEnc(nn.Module):
    def __init__(self, in_ch, lat_dim=64, vec_dim=11):
        super(BackLoopEnc, self).__init__()
        self.lat_dim, self.vec_dim = lat_dim, vec_dim
        
        self.fcs = nn.Sequential(
            nn.Linear(in_ch, 1024),
            nn.Softplus()
        )
        self.down = nn.Sequential(
            nn.Conv2d(1, 64, 5, 2, 2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, 128, 5, 2, 2, bias=False),
            nn.GroupNorm(128, 128),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(128, 256, 5, 1, 0, bias=False),
            nn.GroupNorm(1, 256),
            nn.LeakyReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(256*4*4, lat_dim*vec_dim)
        )
    
    def forward(self, x):
        uped = self.fcs(x)
        uped = uped.view(-1, 1, 32, 32)
        
        out = self.down(uped)
        out = out.view(-1, self.lat_dim, self.vec_dim)
        
        return out
 
    
# Determinstic Path for Shape Estimation
class ShapeEst(nn.Module):
    def __init__(self, out_ch, lat_dim=64, vec_dim=11):
        super(ShapeEst, self).__init__()
        self.equaldist = nn.Sequential(
            nn.Linear(lat_dim*vec_dim, lat_dim*vec_dim),
            nn.ReLU(inplace=True),
        )
        
        # upsampling
        planes = [UpsDist(lat_dim*vec_dim, 128, 1, 5, 1, 0)]
        for ch in [64, 32]:
            planes.append(UpsDist(ch*2, ch, ch, 5, 2, 2, 1))
        self.ups = nn.Sequential(*planes)
        
        # regression
        self.fcs = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32*20*20, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, out_ch)
        )
        
    def forward(self, x):
        z = self.equaldist(x)
        z = z.view(z.size(0), -1, 1, 1)
        
        dec = self.ups(z)
        dec = self.fcs(dec)
        
        return dec


# Base Class for class **ShapeDist**
class UpsDist(nn.Module):
    def __init__(self, in_ch, out_ch, group_ch, kernel_size, stride, padding, out_padding=0):
        super(UpsDist, self).__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size, stride, padding, out_padding, bias=False),
            nn.GroupNorm(group_ch, out_ch),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.block(x)
    