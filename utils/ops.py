import torch.nn as nn
from torchvision import models

class VAEConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(VAEConv, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.block(x)
    
class VAEDeconv(nn.Module):
    def __init__(self, in_ch, out_ch, ks=2, s=2, p=0):
        super(VAEDeconv, self).__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=ks, stride=s, padding=p),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.block(x)
    
class ResEncoder(nn.Module):
    def __init__(self, in_ch):
        super(ResEncoder, self).__init__()
        self.resnet = models.resnet34(pretrained=False)
        self.enc0 = nn.Sequential(
            nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.enc1 = self.resnet.layer1
        self.enc2 = self.resnet.layer2
        self.enc3 = self.resnet.layer3
        self.enc4 = self.resnet.layer4
        
    def forward(self, x):
        out = self.maxpool(self.enc0(x))
        out = self.enc1(out)
        out = self.enc2(out)
        out = self.enc3(out)
        out = self.enc4(out)
        
        return out