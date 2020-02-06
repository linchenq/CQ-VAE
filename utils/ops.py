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
        self.downsample = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        
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
    def __init__(self, in_ch, out_ch):
        super(DiscreteDecoder, self).__init__()
        self.recon = nn.Linear(in_ch, 8192)
        
        self.planes = []
        for ch in [256, 128]:
            self.planes.append(Upsample(ch*2, ch, 2, 2, 0))
        self.ups = nn.Sequential(*self.planes)
        
        self.fcs = nn.Sequential(
            nn.Linear(32768, 8192),
            nn.ReLU(inplace=True),
            nn.Linear(8192, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, out_ch)
        )
        
    def forward(self, x):
        rec = self.recon(x)
        rec = rec.view(-1, 512, 4, 4)
        up = self.ups(rec)
        up = up.view(-1, 32768)
        out = self.fcs(up)
        
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
        

