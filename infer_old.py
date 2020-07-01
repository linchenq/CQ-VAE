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
    
    def get_logits(self, x):
        with torch.no_grad():
            logits, _ = self.model.encode(x)
            return logits
    
    def interpolate_local(self, x, meshes, best_mesh, step, batch_i):
        soften = self.get_soften(x)
        
        prob_pts = {}
        points, masks = [], []
        image = torch.squeeze(x).cpu().numpy()        
        
        # for row in range(self.model.latent_dims):
        #     selections = np.random.choice(self.model.vector_dims, step, p=torch.squeeze(soften[:, row]).cpu().numpy())
        #     probability = torch.squeeze(soften[:, row]).cpu().numpy()[selections]
            
        #     for target, prob in zip(selections, probability):
        #         val = torch.zeros(self.model.vector_dims, device=self.device)
        #         val[target] = 1
        #         replaced = soften.clone().detach()
        #         replaced[:, row] = val
                
        #         sample, segment, _ = self.model.decode(replaced)
        #         sample = torch.squeeze(sample).cpu().numpy() + 64
                
        #         points.append(sample)
        #         masks.append((segment.squeeze(dim=1).squeeze(dim=0).cpu().numpy() >= 0.5).astype(int))
                
        #         points_key = round(prob, 5)
        #         if points_key in prob_pts:
        #             prob_pts[points_key].append(sample)
        #         else:
        #             prob_pts[points_key] = [sample]
        # points, masks, probs = [], [], []
        if True:
            for row in range(self.model.latent_dims):
                points, masks, probs = [], [], []
                selections = range(self.model.vector_dims)
                probability = torch.squeeze(soften[:, row]).cpu().numpy()[selections]
                for i, (target, prob) in enumerate(zip(selections, probability)):
                    if i==4:
                        break
                    val = torch.zeros(self.model.vector_dims, device=self.device)
                    val[target] = 1
                    replaced = soften.clone().detach()
                    replaced[:, row] = val
                    
                    for j in range(replaced.shape[1]):
                        if j == row:
                            continue
                        else:
                            replaced[:, j] = torch.from_numpy(np.array([1.0/11] * 11))
                    
                    sample, segment, _ = self.model.decode(replaced)
                    sample = torch.squeeze(sample).cpu().numpy() + 64
                    
                    points.append(sample)
                    masks.append((segment.squeeze(dim=1).squeeze(dim=0).cpu().numpy() >= 0.5).astype(int))
                    probs.append(prob)
                    
                vis.show_images_tight_local(image, points, probs, meshes, best_mesh, row, alpha=0)

            # image = torch.squeeze(x).cpu().numpy()
        
        
        points, masks, probs = [], [], []
        if False:
            for row in range(self.model.latent_dims):
                tmp = []
                selections = range(self.model.vector_dims)
                probability = torch.squeeze(soften[:, row]).cpu().numpy()[selections]
                for i, (target, prob) in enumerate(zip(selections, probability)):
                    val = torch.zeros(self.model.vector_dims, device=self.device)
                    val[target] = 1
                    replaced = soften.clone().detach()
                    replaced[:, row] = val
                    
                    sample, segment, _ = self.model.decode(replaced)
                    sample = torch.squeeze(sample).cpu().numpy() + 64
                    tmp.append(sample)
                tmp = np.array(tmp)
                points.append(tmp.mean(axis=0))
            vis.show_images_tight_local_mean(image, points, meshes, best_mesh, row)

            # image = torch.squeeze(x).cpu().numpy()

        
        if False:
            for row in range(self.model.latent_dims):
                points, masks, probs = [], [], []
                selections = range(self.model.vector_dims)
                probability = torch.squeeze(soften[:, row]).cpu().numpy()[selections]
                for i, (target, prob) in enumerate(zip(selections, probability)):
                    val = torch.zeros(self.model.vector_dims, device=self.device)
                    val[target] = 1
                    replaced = soften.clone().detach()
                    replaced[:, row] = val
                    
                    sample, segment, _ = self.model.decode(replaced)
                    sample = torch.squeeze(sample).cpu().numpy() + 64
                    
                    points.append(sample)
                    masks.append((segment.squeeze(dim=1).squeeze(dim=0).cpu().numpy() >= 0.5).astype(int))
                    probs.append(prob)
                vis.show_images_tight_local(image, points, probs, meshes, best_mesh, row)

            # image = torch.squeeze(x).cpu().numpy()
            # vis.show_images(image, points, probs, mode="tight")
            # vis.show_images(image, points, probs, mode="subplot")
            # vis.show_images(image, points, probs, mode="sort")
        
        # visualization
        # image = torch.squeeze(x).cpu().numpy()
        # vis.heatmap_regression(image, points, prob_pts, best_mesh)
        # vis.heatmap_boundary(image, points, prob_pts, best_mesh)
    def first_plot(self, x, meshes, best_mesh, tau, sample_num):
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
        
        if True:
            points, masks, probs = [], [], []
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
                probs.append(logprob)
            
            image = torch.squeeze(x).cpu().numpy()
            vis.plot_1(image, points, prob_pts, best_mesh)
    
    def second_plot(self, x, meshes, best_mesh, tau, sample_num):
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
        
        if True:
            points, masks, probs = [], [], []
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
                probs.append(logprob)
            
            image = torch.squeeze(x).cpu().numpy()
            vis.plot_2(image, points, prob_pts, best_mesh)
        
    def interpolate_global(self, x, meshes, best_mesh, tau, sample_num, batch_i):
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
        
        if True:
            points, masks, probs = [], [], []
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
                probs.append(logprob)
            
            image = torch.squeeze(x).cpu().numpy()
            #vis.show_images(image, points, probs, "tight", batch_i)
            #vis.show_images(image, points, probs, "subplot", batch_i)
            #vis.show_images(image, points, probs, "sort", batch_i)
            
            vis.show_images_tight(image, points, probs, meshes, best_mesh)
            vis.show_images_tmp(image, points, probs, "tight", meshes, best_mesh)
            vis.show_images_tmp(image, points, probs, "subplot", meshes, best_mesh)
            vis.show_images_tmp(image, points, probs, "sort", meshes, best_mesh)
        
        # visualization
        image = torch.squeeze(x).cpu().numpy()
        vis.heatmap_regression(image, points, prob_pts, best_mesh, batch_i)
        vis.heatmap_boundary(image, points, prob_pts, best_mesh, batch_i)
    
    def run_best(self, x, meshes, best_mesh, batch_i):
        with torch.no_grad():
            pts, mask = self.get_best(x)
            pts, mask = torch.squeeze(pts).cpu().numpy(), torch.squeeze(mask).cpu().numpy()
            
            fig, ax = plt.subplots()
            ax.imshow(torch.squeeze(x).cpu().numpy(), cmap='gray')
            ax.set_title(f"{batch_i}_image")
            ax.plot(pts[:, 0]+64, pts[:, 1]+64, 'r')
            
            ax.plot(best_mesh[:, 0], best_mesh[:, 1], 'g--', alpha=0.8)
            for mesh in meshes:
                ax.plot(mesh[:, 0]+64, mesh[:, 1]+64, 'b--', alpha=0.3)
                        
    def infer(self, best=True, interpolate=True, accuracy=True, grid=True):
        self.model.eval()
        
        for batch_i, (x, meshes, best_mesh, best_mask)  in enumerate(self.dataloader):
            if batch_i == 52:
                x = torch.unsqueeze(x, dim=1)
                x = x.float().to(self.device)
                meshes = torch.squeeze(meshes, dim=0)
                meshes = [mesh.float() for mesh in meshes]
                best_mesh, best_mask = best_mesh.float().to(self.device), best_mask.float().to(self.device)
                best_mesh, best_mask = torch.squeeze(best_mesh).cpu().numpy(), torch.squeeze(best_mask).cpu().numpy()
                best_mesh += 64
                
                if best:
                    with torch.no_grad():
                        self.run_best(x, meshes, best_mesh, batch_i, tau=0.5, sample_num=100)
                        
                if interpolate:
                    with torch.no_grad():
                        # self.interpolate_local(x, meshes, best_mesh, step=4, batch_i=0)
                        # self.interpolate_global(x, meshes, best_mesh, 1, 100, batch_i)
                        self.first_plot(x, meshes, best_mesh, tau=1, sample_num=100)
                        self.second_plot(x, meshes, best_mesh, tau=1, sample_num=100)
                if accuracy:
                    pass
                
                if grid:
                    with torch.no_grad():
                        logits = self.get_logits(x)
                        soften = self.get_soften(x)
                        logits = torch.squeeze(logits).cpu().numpy()
                        soften = torch.squeeze(soften).cpu().numpy()
                        
                        
                        import seaborn as sns
                        fig, ax = plt.subplots()
                        ax = sns.heatmap(soften)
                        
                        fig1, ax1 = plt.subplots()
                        ax1 = sns.heatmap(logits)
                        
            
            

if __name__ == '__main__':
    for i in range(5, 90, 5):
        parser = argparse.ArgumentParser()
        parser.add_argument("--device", type=str, default="cuda:0")
        parser.add_argument("--sample_step", type=int, default=1)
        # parser.add_argument("--pretrained_weights", type=str, default="./sftp/saves_tau0/ckpt_tau0_20.pth")
        parser.add_argument("--pretrained_weights", type=str, default=f"./sftp/saves_tau0/ckpt_tau0_{i}.pth")
        
        args = parser.parse_args()
        
        dataset = SpineDataset(f"dataset/test.txt")
        model = DiscreteVAE(in_channels=1, out_channels=176*2, seg_channels=1,
                            latent_dims=64, vector_dims=11, 
                            alpha=1., beta=1., tau=1, 
                            device=args.device,
                            sample_step=args.sample_step)
        infer = Inference(args, dataset, model)
        infer.infer(best=False, interpolate=True, accuracy=False, grid=True)
    