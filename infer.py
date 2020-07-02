import numpy as np
import argparse
import matplotlib
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import utils.visualization as vis
import utils.util as uts
from model import CQVAE
from utils.datasets import SpineDataset

def get_best(x, model):
    model.eval()
    with torch.no_grad():
        logits, _ = model.encode(x)
        best = model.shape_est(logits)
    return best

def get_logits(x, model):
    model.eval()
    with torch.no_grad():
        logits, _ = model.encode(x)
        soften = F.softmax(logits, dim=-1)
    return logits, soften

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num_sample", type=int, default=1)
    parser.add_argument("--load_weights", type=str, default=f"./saves_db/ckpt_db_90.pth")
    
    args = parser.parse_args()
    
    dataset = SpineDataset(f"dataset/test.txt")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = CQVAE(in_channels=1,
              out_channels=176*2,
              latent_dims=64,
              vector_dims=11,
                    
              alpha=1.,
              beta=1.,
              gamma=1.,
                    
              tau=1,
              device=args.device,
              num_sample=args.num_sample)
    model = model.to(device)
    model.load_state_dict(torch.load(args.load_weights))

# %%
'''
# Visualize the learning/evaluating phase with different ratios
'''
# TODO: Wait for result
IDX = [1, 2, 4, 8, 16, 32, 64, 128]
T_1, T_2, T_4, T_8, T_16, T_32, T_64, T_128 = [], [], [], [], [], [], [], []
E_1, E_2, E_4, E_8, E_16, E_32, E_64, E_128 = [], [], [], [], [], [], [], []
T_PARAMS = [T_1, T_2, T_4, T_8, T_16, T_32, T_64, T_128]
E_PARAMS = [E_1, E_2, E_4, E_8, E_16, E_32, E_64, E_128]
X_T = []
X_E = []
first = True

for idx, t_param, e_param in zip(IDX, T_PARAMS, E_PARAMS):
    fin = f"./results/logs_aws_{idx}/model_aws_{idx}.log"
    with open(fin, "r") as f:
        for line in f.readlines()[1231:]:
            if line.startswith('+'):
                continue
            elif line.startswith('|'):
                units = line.split('|')
                # training loss
                epoch = int(units[1].strip())
                t_loss = float(units[2].strip())
                if first:
                    X_T.append(epoch)
                t_param.append(t_loss)
                
                # validing loss
                text = units[3].strip()
                if not text.startswith('-'):
                    if first:
                        X_E.append(epoch)
                    e_param.append(float(text))
            else:
                continue
    first = False

# plot for training phase
if True:
    plt.figure()
    plt.grid(linestyle="--")
    
    # ax = plt.gca()
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    
    clrmap = matplotlib.cm.jet(np.linspace(0, 1, len(IDX)))
    for j, (param, index) in enumerate(zip(T_PARAMS, IDX)):
        # lines
        plt.plot(X_T[1:], param[1:],
                 color=clrmap[j],
                 label=f"ratio:{index}/128",
                 linewidth=1.5,
                 linestyle="--")
        # markers
        for k in [21, 41, 61, 81]:
            plt.scatter(X_T[k], param[k],
                        color=clrmap[j],
                        marker='o')
            
        
    plt.xticks(np.arange(1, 101, 20))
    # plt.yticks([])
    # plt.xlim(0, 101)
    # plt.ylim(0, 10)
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    
    # set legend
    plt.legend(loc=1, numpoints=1)
    # leg = plt.gca().get_legend()
    # ltext = leg.get_texts()
    # plt.setp(ltext, fontsize=12, fontweight='bold')
    
    # plt.savefig('./filename.svg', format='svg')
    plt.show()
    
# plot for evaluating phase
if True:
    plt.figure()
    plt.grid(linestyle="--")
    
    # ax = plt.gca()
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    
    clrmap = matplotlib.cm.jet(np.linspace(0, 1, len(IDX)))
    for j, (param, index) in enumerate(zip(E_PARAMS, IDX)):
        # lines
        plt.plot(X_E[1:], param[1:],
                 color=clrmap[j],
                 label=f"ratio:{index}/128",
                 linewidth=1.5,
                 linestyle="--")
        # markers
        for k in [2, 4, 6, 8]:
            plt.scatter(X_E[k], param[k],
                        color=clrmap[j],
                        marker='o')
            
        
    plt.xticks(np.arange(1, 101, 20))
    # plt.yticks([])
    # plt.xlim(0, 101)
    # plt.ylim(0, 10)
    plt.xlabel("Epoch")
    plt.ylabel("Evaluating Loss")
    
    # set legend
    plt.legend(loc=1, numpoints=1)
    # leg = plt.gca().get_legend()
    # ltext = leg.get_texts()
    # plt.setp(ltext, fontsize=12, fontweight='bold')
    
    # plt.savefig('./filename.svg', format='svg')
    plt.show()


    
# %%
'''
Histrogram: Completed
'''
def histogram(model, num, tau):    
    
    for batch_i, (x, meshes, best_mesh) in enumerate(dataloader):
        # data prepared
        x = torch.unsqueeze(x, dim=1)
        x = x.float().to(device)
        
        meshes = torch.squeeze(meshes, dim=0)
        meshes = [mesh.float() for mesh in meshes]
        
        best_mesh = best_mesh.float().to(device)
        best_mesh = torch.squeeze(best_mesh).cpu().numpy()
        best_mesh += 64
        
        # initial variables
        points, probs = [], []
        image = torch.squeeze(x).cpu().numpy()
        
        # testing phase
        model.eval()
        with torch.no_grad():
            logits, soften = get_logits(x, model)
            soften = soften.squeeze(dim=0)
            model.tau = tau
            
            for i in range(num):
                z = model.reparametrize(logits)
                sample, _ = model.shape_dist(z)
                sample = torch.squeeze(sample).cpu().numpy() + 64
                
                index = torch.argmax(z, dim=-1).squeeze(dim=0)
                logprob = torch.sum(torch.log(soften[range(len(index)), index]))
                
                points.append(sample)
                probs.append(logprob.item())
        
        # visualize
        vis.heatmap(image, points, probs, best_mesh)
        
        break

histogram(model, 100, 0.5)

# %%
'''
Z space visualization
'''
def z_visualize():
    for batch_i, (x, meshes, best_mesh) in enumerate(dataloader):
        # data prepared
        x = torch.unsqueeze(x, dim=1)
        x = x.float().to(device)
        
        logits, soften = get_logits(x, model)
        logits = torch.squeeze(logits).cpu().numpy()
        soften = torch.squeeze(soften).cpu().numpy()
        
        vis.z_heatmap(logits)
        vis.z_heatmap(soften)
        
        break
z_visualize()

# %%
'''
Variance visualization around the boundary
[1] General Variance
[2] Local Variance
'''
def vari_boundary(model, num, tau):    
    
    for batch_i, (x, meshes, best_mesh) in enumerate(dataloader):
        # data prepared
        x = torch.unsqueeze(x, dim=1)
        x = x.float().to(device)
        
        meshes = torch.squeeze(meshes, dim=0)
        meshes = [mesh.float() for mesh in meshes]
        
        best_mesh = best_mesh.float().to(device)
        best_mesh = torch.squeeze(best_mesh).cpu().numpy()
        best_mesh += 64
        
        # initial variables
        points = []
        image = torch.squeeze(x).cpu().numpy()
        
        # testing phase
        model.eval()
        with torch.no_grad():
            logits, soften = get_logits(x, model)
            soften = soften.squeeze(dim=0)
            model.tau = tau
            
            for i in range(num):
                z = model.reparametrize(logits)
                sample, _ = model.shape_dist(z)
                sample = torch.squeeze(sample).cpu().numpy() + 64
                
                points.append(sample)
        
        # visualize
        vis.heatboundary(image, points, best_mesh)
        
        break
vari_boundary(model, 100, 0.5)

# %%
'''
Variance Comparision between ground truth and test set
'''
def vari_compare(model, num, tau):    
    
    for batch_i, (x, meshes, best_mesh) in enumerate(dataloader):
        # data prepared
        x = torch.unsqueeze(x, dim=1)
        x = x.float().to(device)
        
        meshes = [mesh.float() for mesh in meshes]
        gen_meshes = uts.batch_linear_combination(cfg="cfgs/cfgs_table.npy",
                                                  target=32, 
                                                  meshes=meshes,
                                                  random_seed=0)
        gen_meshes = torch.squeeze(gen_meshes, dim=0)
        gen_meshes += 64
        
        best_mesh = best_mesh.float().to(device)
        best_mesh = torch.squeeze(best_mesh).cpu().numpy()
        best_mesh += 64
        
        # initial variables
        points = []
        image = torch.squeeze(x).cpu().numpy()
        
        # testing phase
        model.eval()
        with torch.no_grad():
            logits, soften = get_logits(x, model)
            soften = soften.squeeze(dim=0)
            model.tau = tau
            
            for i in range(num):
                z = model.reparametrize(logits)
                sample, _ = model.shape_dist(z)
                sample = torch.squeeze(sample).cpu().numpy() + 64
                
                points.append(sample)
        
        # visualize
        vis.compare_gt_test(image, points, best_mesh, gen_meshes)
        
        break
vari_compare(model, 100, 0.5)

# %%
'''
Uncertainty (sum of logprob) and regression loss
'''
def relation_search(model, num, tau):    
    
    for batch_i, (x, meshes, best_mesh) in enumerate(dataloader):
        # data prepared
        x = torch.unsqueeze(x, dim=1)
        x = x.float().to(device)
        
        best_mesh = best_mesh.float().to(device)
        
        # initial variables
        probs, losses = [], []
        image = torch.squeeze(x).cpu().numpy()
        
        # testing phase
        model.eval()
        with torch.no_grad():
            logits, soften = get_logits(x, model)
            soften = soften.squeeze(dim=0)
            model.tau = tau
            
            for i in range(num):
                z = model.reparametrize(logits)
                sample, _ = model.shape_dist(z)
                loss = F.mse_loss(sample, best_mesh)
                
                index = torch.argmax(z, dim=-1).squeeze(dim=0)
                logprob = torch.sum(torch.log(soften[range(len(index)), index]))
                
                probs.append(logprob.item())
                losses.append(loss.item())
        # cov and corrcoef
        con_mat = np.array([probs, losses])
        print(np.cov(con_mat))
        print(np.corrcoef(con_mat))
        
        # r, p value
        import scipy.stats as stats
        r, p = stats.pearsonr(probs, losses)
        print(f"r is {r} and p is {p}")
    
        # visualization
        vis.relation_vis(probs, losses)
        
        break

relation_search(model, 100, 0.5)