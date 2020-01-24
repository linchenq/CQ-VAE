import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

import sys
sys.path.append("..") 

from models import BetaVAECon
from utils.datasets import SpineDataset
import utils.utils as uts
    
class Inference(object):
    def __init__(self, args, dataset, model):
        self.args = args
        self.model = model
        self.dataloader = {
            'test': DataLoader(dataset['test'], batch_size=self.args.batch_size, shuffle=True)
        }
        if self.args.pretrained_weights is not None:
            if self.args.pretrained_weights.endswith(".pth"):
                self.model.load_state_dict(torch.load(self.args.pretrained_weights))
            else:
                self.logger.log("WAR", "Unknown weights file")
                raise NotImplementedError
        
        # DL prepared
        self.device = torch.device(self.args.device if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
    
    def get_z(self, x):
        self.model.eval()
        with torch.no_grad():
            mu, logvar = self.model.encode(x)
            z = self.model.reparametrize(mu, logvar)
        return z
    
    def interpolate(self, x):
        z = self.get_z(x)        
        interpolation = torch.arange(-3, 3.1, 1)
        
        result = []
        for row in range(self.model.latent_dims):
            for val in interpolation:
                z[:, row] = val
                sample = self.model.decode(z)
                sample = torch.squeeze(sample.cpu())
                result.append(sample)
        
        uts.show_images(result, ncols=len(interpolation))
        
        
    
    def infer(self, recon=False, interpolate=False):
        self.model.eval()
        
        for batch_i, (x, mesh) in enumerate(self.dataloader['test']):
            x, mesh = x.float(), mesh.float()
            x, mesh = x.to(self.device), mesh.to(self.device)
            
            if interpolate:
                with torch.no_grad():
                    self.interpolate(x)
                    break
                
            if recon:
                with torch.no_grad():
                    xr, mu, logvar = self.model(x)
                    fig, ax = plt.subplots()
                    ax.plot()
                    ax.imshow(torch.squeeze(xr).cpu().numpy(), cmap='gray')
                    break

if __name__ == '__main__':
        
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--pretrained_weights", type=str, default="./saves/ckpt_140.pth")
    
    args = parser.parse_args()
    
    dataset = {}
    dataset['test'] = SpineDataset(f"../dataset/test.txt")
    model = BetaVAECon(in_ch=1, out_ch=1, latent_dims=64, beta=1.)
    
    infer = Inference(args, dataset, model)
    infer.infer(interpolate=True)
    
    