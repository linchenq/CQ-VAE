import torch.nn as nn

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
    def __init__(self, in_ch, out_ch):
        super(VAEDeconv, self).__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.block(x)
    