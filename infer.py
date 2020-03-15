import numpy as np
import argparse
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import utils.visualize as vis
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
        with torch.no_grad():
            logits, _ = self.model.encode(x)
            best_reg, best_seg, _ = self.model.decode(logits)
        
        return best_reg, best_seg
    
    def get_soften(self, x):
        with torch.no_grad():
            logits, _ = self.model.encode(x)
            soften = F.softmax(logits, dim=-1)
        
        return soften
    
    def interpolate_local(self, x, meshes, best_mesh, step):
        soften = self.get_soften(x)
        
        prob_pts = {}
        points, masks = [], []
        
        for row in range(self.model.latent_dims):
            selections = np.random.choice(self.model.vector_dims, step, p=torch.squeeze(soften[:, row]).cpu().numpy())
            probability = torch.squeeze(soften[:, row]).cpu().numpy()[selections]
            
            for target, prob in zip(selections, probability):
                val = torch.zeros(self.model.vector_dims, device=self.device)
                val[target] = 1
                replaced = soften.clone().detach()
                replaced[:, row] = val
                
                sample, segment, _ = self.model.decode(replaced)
                sample = torch.squeeze(sample).cpu().numpy() + 64
                
                points.append(sample)
                masks.append((segment.squeeze(dim=1).squeeze(dim=0).cpu().numpy() >= 0.5).astype(int))
                
                points_key = round(prob, 5)
                if points_key in prob_pts:
                    prob_pts[points_key].append(sample)
                else:
                    prob_pts[points_key] = [sample]
        
        # visualization
        image = torch.squeeze(x).cpu().numpy()
    
    def interpolate_global(self, x, meshes, best_mesh, tau, sample_num):
        soften = self.get_soften(x)
        soften = soften.squeeze(dim=0)
        
        prob_pts = {}
        points, masks = [], []
        
        with torch.no_grad():
            logits, _ = self.model.encode(x)
            self.model.tau = tau
            
            for i in range(sample_num):
                z = self.model.reparametrize(logits)
                sample, segment, _ = self.model.decode(z)
                sample = torch.squeeze(sample).cpu().numpy() + 64
                
                index = torch.argmax(z, dim=-1).squeeze(dim=0)
                logprob = torch.sum(torch.log(soften[range(len(index)), index]))
                
                points.append(sample)
                masks.append((segment.squeeze(dim=1).squeeze(dim=0).cpu().numpy() >= 0.5).astype(int))
                
                if logprob in prob_pts:
                    prob_pts[logprob.cpu()].append(sample)
                else:
                    prob_pts[logprob.cpu()] = [sample]
        
        # visualization
        image = torch.squeeze(x).cpu().numpy()
        vis.heatmap_regression(image, points, prob_pts, best_mesh)
        vis.heatmap_boundary(image, points, prob_pts, best_mesh)

    def infer(self, best=True, interpolate=True):
        self.model.eval()
        
        for batch_i, (x, meshes, best_mesh, best_mask) in enumerate(self.dataloader):
            x = torch.unsqueeze(x, dim=1)
            x = x.float().to(self.device)
            meshes = torch.squeeze(meshes, dim=0)
            meshes = [mesh.float() for mesh in meshes]
            best_mesh, best_mask = best_mesh.float().to(self.device), best_mask.float().to(self.device)
            best_mesh, best_mask = torch.squeeze(best_mesh).cpu().numpy(), torch.squeeze(best_mask).cpu().numpy()
            best_mesh += 64
            
            if best:
                with torch.no_grad():
                    pts, mask = self.get_best(x)
                    pts, mask = torch.squeeze(pts).cpu().numpy(), torch.squeeze(mask).cpu().numpy()
                    
                    fig, ax = plt.subplots()
                    ax.imshow(torch.squeeze(x).cpu().numpy(), cmap='gray')
                    mask = (mask >= 0.5)
                    ax.imshow(mask, cmap='gray', alpha=0.5)
                    ax.plot(pts[:, 0] + 64, pts[:, 1] + 64, 'r')
                    
                    ax.plot(best_mesh[:, 0] + 64, best_mesh[:, 1] + 64, 'g--', alpha=0.8)
                    
                    for mesh in meshes:
                        ax.plot(mesh[:, 0] + 64, mesh[:, 1] + 64, 'b--', alpha=0.3)
                    
            if interpolate:
                with torch.no_grad():
                    # self.interpolate_local(x, meshes, best_mesh, step=5)
                    self.interpolate_global(x, meshes, best_mesh, tau=0.5, sample_num=100)
                    
            break

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--sample_step", type=int, default=1)
    parser.add_argument("--pretrained_weights", type=str, default="./saves_tau1/ckpt_tau1_40.pth")
    
    args = parser.parse_args()
    
    dataset = SpineDataset(f"dataset/test.txt")
    model = DiscreteVAE(in_channels=1, out_channels=176*2, seg_channels=1,
                        latent_dims=64, vector_dims=11, 
                        alpha=1., beta=1., tau=1, 
                        device=args.device,
                        sample_step=args.sample_step)
    infer = Inference(args, dataset, model)
    infer.infer(best=False, interpolate=True)
    