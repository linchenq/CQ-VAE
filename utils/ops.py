import torch.nn as nn
from torchvision import models

class DiscreteEncoder(nn.Module):
    def __init__(self, in_ch):  
        super(DiscreteEncoder, self).__init__()
        self.resnet = models.resnet34(pretrained=False)
        self.enc0 = nn.Sequential(
            nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
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

class DiscreteDecoder(nn.Module):
    def __init__(self, in_ch, out_ch, seg_ch):
        super(DiscreteDecoder, self).__init__()
        self.recon = nn.Linear(in_ch, 8192)
        
        # 512, 4, 4 -> 256, 8, 8 -> 128, 16, 16
        planes = []
        for ch in [256, 128]:
            planes.append(Upsample(ch*2, ch, 2, 2, 0))
        self.upsample = nn.Sequential(*planes)
        
        self.regress = RegressDecoder(32768, out_ch)
        self.segment = SegmentDecoder(seg_ch)
        
    def forward(self, x):
        rec = self.recon(x)
        rec = rec.view(-1, 512, 4, 4)
        upsample = self.upsample(rec)
        
        # segmentation
        seg_out = self.segment(upsample)
        
        # regression
        reg_up = upsample.view(-1, 32768)
        reg_out = self.regress(reg_up)

        return reg_out, seg_out

class RegressDecoder(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(RegressDecoder, self).__init__()
        
        self.fcs = nn.Sequential(
            nn.Linear(in_ch, 8192),
            nn.ReLU(inplace=True),
            nn.Linear(8192, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, out_ch)
        )
        
    def forward(self, x):
        return self.fcs(x)
        
class SegmentDecoder(nn.Module):
    def __init__(self, out_ch):
        super(SegmentDecoder, self).__init__()
        
        self.planes = []
        for ch in [64, 32, 16]:
            self.planes.append(Upsample(ch*2, ch, 2, 2, 0))
        self.ups = nn.Sequential(*self.planes)
        
        self.conv = nn.Conv2d(in_channels=16, out_channels=out_ch, kernel_size=1)
        
    def forward(self, x):
        up = self.ups(x)
        out = self.conv(up)
        return out

class RegressAutoEncoder(nn.Module):
    def __init__(self, in_ch, latent_dims, vector_dims):
        super(RegressAutoEncoder, self).__init__()
        self.latent_dims, self.vector_dims = latent_dims, vector_dims
        
        self.fcs = nn.Sequential(
            nn.Linear(in_ch, 1024),
            nn.Softplus(),
            nn.Linear(1024, 8192)
        )
        self.conv = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.Softplus(),
            nn.Conv2d(128, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
        )        
        self.down = nn.Conv2d(256, 256, kernel_size=2, stride=2, padding=0, bias=False)
        self.regress = nn.Linear(4096, latent_dims * vector_dims)
    
    def forward(self, x):
        uped = self.fcs(x)
        uped = uped.view(-1, 128, 8, 8)
        
        out = self.down(self.conv(uped))
        out = out.view(-1, 4096)
        out = self.regress(out)
        out = out.view(-1, self.latent_dims, self.vector_dims)
        
        return out    

class Upsample(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=2, stride=2, padding=0):
        super(Upsample, self).__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.block(x)