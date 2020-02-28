import numpy as np
import argparse

import torch
from torch.utils.data import DataLoader

import utils.util as uts
from model import DiscreteVAE
from utils.datasets import SpineDataset


class Inference(object):
    def __init__(self, args, dataset, model):
        self.args = args
        self.model = model
        self.dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        
        if self.args.pretrained_weights is not None:
            self.model.load_state_dict(torch.load(self.args.pretrained_weights))
        
        self.device = torch.device(self.args.device if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
    
    def get_best(self, x):
        self.model.eval()
       
        with torch.no_grad():
            logits, _ = self.model.encode(x)
            best_reg, best_seg, _ = self.model.decode(logits)
        return best_reg, best_seg
        
    def get_z(self, x):
        self.model.eval()
        
        with torch.no_grad():
            logits, _ = self.model.encode(x)
            z = self.model.reparametrize(logits)
        return z
    
    def interpolate(self, x, step):
        z = self.get_z(x)
        
        points = []
        for row in range(self.model.latent_dims):
            selections = np.random.choice(self.model.vector_dims, step, p=z[:, row])
            for target in selections:
                val = torch.zeros(self.model.vector_dims, device=self.device)
                val[target] = 1
                
                replaced = z.clone().detach()
                replaced[:, row] = val
                
                sample, _, _ = self.model.decode(z)
                sample = torch.squeeze(sample).cpu()
                points.append(sample.numpy())
            
            if row == 5:
                break
            
        images = [torch.squeeze(x).cpu().numpy() for i in range(len(points))]
        
        uts.show_images(images, step, points)

    def infer(self, best=True, interpolate=True):
        self.model.eval()
        
        for batch_i, (x, _, best_mesh, best_mask) in enumerate(self.dataloader):
            x = torch.unsqueeze(x, dim=1)
            x = x.float().to(self.device)
            best_mesh, best_mask = best_mesh.float().to(self.device), best_mask.float().to(self.device)
            
            if best:
                with torch.no_grad():
                    pts, mask = self.get_best(x)
                    pts, mask = torch.squeeze(pts).cpu().numpy(), torch.squeeze(mask).cpu().numpy()
                    
                    fig, ax = plt.subplots()
                    ax.plot()
                    ax.imshow(torch.squeeze(x).cpu().numpy(), cmap='gray')
                    ax.plot(pts[:, 0], pts[:, 1], 'g-')
                    
                    fig_best, ax_best = plt.subplots()
                    ax_best.plot()
                    ax_best.imshow(torch.squeeze(x).cpu().numpy(), cmap='gray')
                    ax_best.plot(best_mesh[:, 0], best_mesh[:, 1], 'g-')
                    
            if interpolate:
                with torch.no_grad():
                    self.interpolate(x, step=5)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--sample_step", type=int, default=1)
    parser.add_argument("--pretrained_weights", type=str, default=None)
    
    args = parser.parse_args()
    
    dataset = SpineDataset(f"dataset/test.txt")
    model = DiscreteVAE(in_channels=1, out_channels=176*2, seg_channels=1,
                        latent_dims=64, vector_dims=11, 
                        alpha=1., beta=1., tau=1., 
                        device=args.device,
                        sample_step=args.sample_step)
    infer = Inference(args, dataset, model)
    infer.infer(True, False)
    