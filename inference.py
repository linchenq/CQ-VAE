import argparse
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import scipy.stats as stats
from sklearn import linear_model

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model import CQVAE
from utils.datasets import SpineDataset

class Infer(object):
    def __init__(self, args, dataset, model, device):
        self.args = args
        self.dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        self.device = device
        self.model = model.to(device)
        
        self.model.load_state_dict(torch.load(args.load_weights))
        
    def __get_logits(self, x):
        self.model.eval()
        with torch.no_grad():
            logits, _ = self.model.encode(x)
            soften = F.softmax(logits, dim=-1)
        return logits, soften
    
    def __generate_pts(self, x, tau):
        self.model.eval()
        self.model.tau = tau
        with torch.no_grad():
            logits, soften = self.__get_logits(x)
            z = self.model.reparametrize(logits)
            pts, _ = self.model.shape_dist(z)
            return pts
        
    def __generate_random_Z(self, mode, batch=1, row=64, col=11):
        if mode == "one-hot-vector":
            mask = torch.arange(0, col, 1).expand(batch, row, col)
            idx = torch.randint(0, col, (batch, row, 1)).expand(batch, row, col)
            Z = (mask == idx).to(torch.float32)
        elif mode == "random":
            Z = torch.rand(row, col)
            Z = F.softmax(Z, dim=-1)
        elif mode == "randomWOSoftmax":
            Z = torch.rand(row, col)
        else:
            raise NotImplementedError("CHECKING ERROR: Only one-hot-vector and random mode is supported for Zs")
        
        return Z
    
    def __generate_random_shape(self, mode):
        z = self.__generate_random_Z(mode).to(self.device)
        ret, _ = self.model.shape_dist(z)
        
        return ret.squeeze().detach().cpu().numpy()
        
    def __entropy(self, P, reduction='mean'):
        # P: batch_size x M x N
        P = torch.clamp(P, min=1e-8, max=1-1e-5)
        E = torch.mean(torch.sum(-P*torch.log(P), dim=2),dim=1)
        if reduction == 'mean':
            E = E.mean()
        return E
    
    def __pts_variance(self, S_all):
        # S_all: Nx176x2
        # Savg:  1x176x2
        Savg = np.mean(S_all, axis=0, keepdims=True)    
        pv = np.mean(np.sqrt(np.sum((S_all-Savg)**2, axis=2)), axis=0)
        return pv

    def __spe_variance(self, S_all):
        pv = self.__pts_variance(S_all)
        sv = np.mean(pv)
        return sv
    
    def __bias(self, S, Sbest):
        # S:     176x2
        # Sbest: 176x2
        b = np.mean(np.sqrt(np.sum((S-Sbest)**2, axis=1)), axis=0)
        return b

    def __draw_image(self, x, pts):
        x = torch.squeeze(x).cpu().numpy()
        pts = torch.squeeze(pts).cpu().numpy() + 64
        fig, ax = plt.subplots()
        ax.plot(pts[:, 0], pts[:, 1], 'g-', alpha=0.5)
        ax.imshow(x, cmap='gray', alpha=1)
    
    # x, y: the array of x and y
    # x/y_title: {ground-truth/model-computed}_{entropy/loss/bias}
    def __draw_sns(self, x, y, x_title, y_title, debug=False):
        arr = np.array([x, y]).T
        df = pd.DataFrame(arr, columns=[x_title, y_title])
        
        if debug:
            con_mat = np.array([x, y])
            print("{} vs {}".format(x_title, y_title))
            print("Cov: {}".format(np.cov(con_mat)))
            print("Corrcoef: {}".format(np.corrcoef(con_mat)))
            
            # r, p value calculation
            r, p = stats.pearsonr(x, y)
            print("r is {} and p is {}".format(r, p))
        
        g = sns.jointplot(x_title, y_title, data=df, kind='kde', space=0, color='b')
        
        ### regression formula
        f_model = linear_model.HuberRegressor()
        f_x, f_y = df[x_title].values.reshape(-1, 1), df[y_title].values.reshape(-1, 1)
        f_model.fit(f_x, f_y)
        f_k, f_b = f_model.coef_[0].astype(float), f_model.intercept_.astype(float)
        
        if debug:
            print("y={0:.3f}x+{1:.3f}".format(f_k, f_b))
        
        # plt phase
        g.plot_joint(sns.regplot, scatter=False, ci=None, 
                     line_kws={'linewidth':2,
                                'linestyle':'--',
                                'color':'red',
                                'label':'y={0:.3f}x+{1:.3f}'.format(f_k, f_b)
                                })
        g.fig.suptitle("y={0:.3f}x+{1:.3f}".format(f_k, f_b))        
        # g.ax_joint.text(x=0.3, y=6, s='y={0:.3f}x+{1:.3f}'.format(f_k, f_b),
        #                 fontstyle='oblique', fontsize='large', fontweight='medium')
        
        plt.savefig("{}_vs_{}.svg".format(x_title, y_title))
    
    # The module is designed to test the generation capability for CQ-VAE
    def __draw_random_shapes(self, num_sample, mode):
        for i in range(num_sample):
            shape = self.__generate_random_shape(mode)
            fig, ax = plt.subplots()
            ax.plot(shape[:, 0], shape[:, 1], 'r-')
    
    # The module is designed to visualize Z-space
    def __draw_Z_distribution(self, z):
        fig, ax = plt.subplots()
        ax = sns.heatmap(z)
    
    # The module is designed to visualize the points distribution around the boundary
    def __draw_single_image_boundary_distribution(self, img, pts, best, detail=False):
        # img: H x W, pts: N x 176 x 2, best: 176 x 2
        mean_shape = []
        for pt_i in range(best.shape[0]):
            cluster = np.vstack(pt[pt_i] for pt in pts)
            mean_shape.append([np.mean(cluster[:, 0]), np.mean(cluster[:, 1])])
        np_mean_shape = np.array(mean_shape)    
        pt_vars = self.__pts_variance(pts)
        
        # plot the "boundary" using the mean shape from all generate shapes
        # draw the distribution using the heatmap trick (sort by point variance, use different color to show variance)
        idx = np.argsort(pt_vars)[::-1]
        fig, ax = plt.subplots()
        cm = matplotlib.cm.jet(np.linspace(0, 1, 11))
        f_x, f_y, f_clr = [], [], []
        for pt_i in range(best.shape[0]):
            f_x.append(np_mean_shape[idx[pt_i], 0])
            f_y.append(np_mean_shape[idx[pt_i], 1])
            f_clr.append(cm[ int(pt_i * 10 / best.shape[0]) ])
        pcm = ax.scatter(f_x, f_y, s=3, marker='o', color=f_clr, alpha=1)
        fig.colorbar(pcm, ax=ax, boundaries=np.linspace(0, 1, 11))
        ax.imshow(img, cmap='gray', alpha=1)
        
        # plot the details (zoomed image/points) regarding the min/max point variance parts
        if detail:
            min_idx, max_idx, mid_idx = idx[-1], idx[0], idx[best.shape[0]//2]
            for i in (min_idx, max_idx, mid_idx):
                gathered = np.vstack(pt[i, :] for pt in pts)
                ax.plot(gathered[:, 0], gathered[:, 1])
            
            for i in (min_idx, max_idx, mid_idx):
                fig, ax = plt.subplots()
                gathered = np.vstack(pt[i, :] for pt in pts)
                
                # define the sub area with [xmin, ymin, xmax, ymax]
                x_min, x_max = int(np.min(gathered[:, 0])), int(np.max(gathered[:, 0])) + 1
                y_min, y_max = int(np.min(gathered[:, 1])), int(np.max(gathered[:, 1])) + 1
                
                sns.kdeplot(gathered[:, 0], gathered[:, 1],
                            cbar=True,
                            shade=True,
                            cmap=matplotlib.cm.jet,
                            shade_lowest=False,
                            n_levels=5,
                            alpha=0.5)
                
                cliped_best = np.vstack([pt for pt in best if (x_min <= pt[0] <= x_max and y_min <= pt[1] <= y_max)])
                cliped_mean = np.vstack([pt for pt in np_mean_shape if (x_min <= pt[0] <= x_max and y_min <= pt[1] <= y_max)])
                
                # ax.invert_yaxis()
                ax.scatter(gathered[:, 0], gathered[:, 1], s=5, marker='2', color='b')
                ax.plot(cliped_best[:, 0], cliped_best[:, 1], 'k-', color='red')
                ax.plot(cliped_mean[:, 0], cliped_mean[:, 1], 'k-', color='blue')
                ax.imshow(img[x_min:x_max, y_min:y_max], extent=[x_min, x_max, y_min, y_max], origin='upper', aspect='equal', cmap='gray')
    
    def __draw_boundary_distribution(self, num_sample, tau):
        for batch_i, (x, gt, best) in enumerate(self.dataloader):
            x = x.to(device)
            
            # remove batch channel, convert to numpy
            gt = torch.squeeze(gt, dim=0).cpu().numpy()
            best = torch.squeeze(best).cpu().numpy() + 64
            img = torch.squeeze(x).cpu().numpy()
            
            pts = []
            for sample_i in range(num_sample):
                sample = self.__generate_pts(x, tau)
                pts.append(torch.squeeze(sample).cpu().numpy() + 64)
            
            self.__draw_single_image_boundary_distribution(img, pts, best, detail=False)
    
    def __get_model_computed_metrics(self, num_sample, tau):
        entro, bias, vari = [], [], []
        self.model.tau = tau
        self.model.eval()
        
        for batch_i, (x, gt, best) in enumerate(self.dataloader):
            x = x.to(self.device)
            in_bias, in_pts = [], []
            
            with torch.no_grad():
                logits, soften = self.__get_logits(x)
                entro.append(self.__entropy(logits).item())
                
                for i in range(num_sample):
                    z = self.model.reparametrize(logits)
                    pts, _ = self.model.shape_dist(z)
                    in_pts.append(torch.squeeze(pts).cpu().numpy())
                    in_bias.append(self.__bias(torch.squeeze(pts).cpu().numpy(), torch.squeeze(best).cpu().numpy()))
                
                bias.append(np.mean(in_bias))
                vari.append(self.__spe_variance(in_pts))
        
        return entro, bias, vari
    
    def __get_gt_variance(self):
        gt_vari = []
        for batch_i, (x, gt, best) in enumerate(self.dataloader):
            np_gt = torch.squeeze(gt, dim=0).cpu().numpy()
            gt_vari.append(self.__spe_variance(np_gt))
        return gt_vari

    
    def forward(self):
        
        # 1. Maually design random Z with 3 different modes, w/o any prior knowledge on test set
        self.__draw_random_shapes(num_sample=10, mode="random")
        
        # 2. Generate possible shapes on x in test set
        for batch_i, (x, gt, _) in enumerate(self.dataloader):
            x = x.to(self.device)
            self.__draw_image(x, self.__generate_pts(x, 0.5))
        
        # 3. Visualize the Z-space distribution
        for batch_i, (x, gt, _) in enumerate(self.dataloader):
            x = x.to(self.device)
            logits, soften = self.__get_logits(x)
            self.__draw_Z_distribution(torch.squeeze(logits).cpu().numpy())
            self.__draw_Z_distribution(torch.squeeze(soften).cpu().numpy())
        
        # 4. Visualize the distribution of points around the boundary
        # For example, if the points gather around some parts of boundary, the 'uncertainty' low while result accurate
        self.__draw_boundary_distribution(num_sample=100, tau=self.model.tau)
        
        # 5. Draw comparsion between different metrics
        for tau in [0.01, 0.1, 0.5, 1, 5, 10]:
            entro, bias, vari = self.__get_model_computed_metrics(128, tau)
            gt_vari = self.__get_gt_variance()
            
            self.__draw_sns(vari, bias, "model-computed_variance", "model-computed_bias", debug=False)
            self.__draw_sns(vari, entro, "model-computed_variance", "model-computed_entropy", debug=False)
            self.__draw_sns(entro, bias, "model-computed_entropy", "model-computed_bias", debug=False)
            
            self.__draw_sns(gt_vari, vari, "ground-truth_variance", "model-computed_bias", debug=False)
            self.__draw_sns(gt_vari, entro, "ground-truth_variance", "model-computed_entropy", debug=False)
            self.__draw_sns(gt_vari, bias, "ground-truth_variance", "model-computed_bias", debug=False)
        
        pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num_sample", type=int, default=1)
    parser.add_argument("--load_weights", type=str, default="./saves/saves_debug/ckpt_100_debug.pth")
    
    args = parser.parse_args()
    dataset = SpineDataset("dataset/test.txt")
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    model = CQVAE(in_channels=1,
                  out_channels=176*2,
                  latent_dims=64,
                  vector_dims=11,
                        
                  tau=1.,
                  device=device,
                  num_sample=args.num_sample)
    infer = Infer(args, dataset, model, device)
    infer.forward()