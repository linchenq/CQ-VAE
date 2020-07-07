import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import seaborn as sns
from scipy.stats.kde import gaussian_kde
from scipy.ndimage.filters import gaussian_filter

from terminaltables import AsciiTable

def heatmap(image, points, probs, best):
    index = np.argsort(probs)
    min_prob, max_prob = probs[index[0]], probs[index[-1]]
    
    x, y = [pt[:, 0] for pt in points], [pt[:, 1] for pt in points]
    x, y = np.hstack(x), np.hstack(y)
    
    set_cmap = matplotlib.cm.jet
    
    if True:
        Z, xedges, yedges = np.histogram2d(x, y)
        fig, ax = plt.subplots()
        pcm = ax.pcolormesh(xedges, yedges, Z.T, cmap=set_cmap, alpha=0.5)
        fig.colorbar(pcm, ax=ax)
        
        ax.plot(points[index[0]][:, 0], points[index[0]][:, 1], 'g-', alpha=0.5)
        ax.plot(points[index[-1]][:, 0], points[index[-1]][:, 1], 'g-', alpha=0.5)
        ax.plot(best[:, 0], best[:, 1], 'w-')
        ax.imshow(image, cmap='gray')
        
    if True:
        fig, ax = plt.subplots()
        sns.kdeplot(x, y,
                    cbar=True,
                    shade=True,
                    cmap=set_cmap,
                    shade_lowest=False,
                    n_levels=5,
                    alpha=0.5)
        
        ax.grid(linestyle='--')
        # ax.scatter(x, y, s=5, alpha=0.5, color='k')        
        ax.imshow(image, cmap='gray', alpha=1)

    if True:
        fig, ax = plt.subplots()
        pcm = ax.hexbin(x, y, gridsize=15, cmap=set_cmap, alpha=0.5)
        fig.colorbar(pcm, ax=ax)
        ax.imshow(image, cmap='gray', alpha=1)
        
    if True:
        xi, yi = np.mgrid[x.min():x.max():x.size**0.5*1j, y.min():y.max():y.size**0.5*1j]
        k = gaussian_kde(np.vstack([x, y]))
        zi = k(np.vstack([xi.flatten(), yi.flatten()]))
        zi = (zi - np.min(zi)) / (np.max(zi) - np.min(zi))
        
        fig, ax = plt.subplots()
        pcm = ax.contourf(xi, yi, zi.reshape(xi.shape), cmap=set_cmap, alpha=0.5)
        fig.colorbar(pcm, ax=ax)
        ax.imshow(image, cmap='gray', alpha=1)

def z_heatmap(z):
    fig, ax = plt.subplots()
    ax = sns.heatmap(z)
        
def heatboundary(image, points, best):
    pt_vars, mean_pt_x, mean_pt_y = [], [], []
    for i in range(len(best)):
        cluster = np.vstack(pt[i, :] for pt in points)
        mean_pt_x.append(np.mean(cluster[:, 0]))
        mean_pt_y.append(np.mean(cluster[:, 1]))
        pt_vars.append(np.var(cluster))
        
    index = np.argsort(pt_vars)
    index = index[::-1]
    
    fig, ax = plt.subplots()
    colormap = matplotlib.cm.jet(np.linspace(0, 1, 11))  
    f_x, f_y, f_clr = [], [], []
    
    for i in range(len(best)):
        clr_idx = int(i*10/len(best))
        f_x.append(mean_pt_x[index[i]])
        f_y.append(mean_pt_y[index[i]])
        f_clr.append(colormap[clr_idx])
    pcm = ax.scatter(f_x, f_y, s=3, marker='o', color=f_clr, alpha=1)
    fig.colorbar(pcm, ax=ax, boundaries=np.linspace(0, 1, 11))
    ax.plot(best[:, 0], best[:, 1], 'w-')
    ax.imshow(image, cmap='gray', alpha=1)
    
    # Local image
    if True:
        min_var_idx, mid_var_idx, max_var_idx = index[-1], index[len(best) // 2], index[0]
        for idx in (min_var_idx, mid_var_idx, max_var_idx):
            plot_points = np.vstack(pt[idx, :] for pt in points)
            ax.plot(plot_points[:, 0], plot_points[:, 1])
    
        for idx in (min_var_idx, mid_var_idx, max_var_idx):
            fig, ax = plt.subplots()
            plot_points = np.vstack(pt[idx, :] for pt in points)
            x_min, x_max = int(np.min(plot_points[:, 0])), int(np.max(plot_points[:, 0])) + 1
            y_min, y_max = int(np.min(plot_points[:, 1])), int(np.max(plot_points[:, 1])) + 1
            
            sns.kdeplot(plot_points[:, 0], plot_points[:, 1],
                        cbar=True,
                        shade=True,
                        cmap=matplotlib.cm.jet,
                        shade_lowest=False,
                        n_levels=5,
                        alpha=0.5)
            
            cliped_best = np.vstack([pt for pt in best if (x_min <= pt[0] <= x_max and y_min <= pt[1] <= y_max)])
            # cliped_best = np.vstack([[f_x[j], f_y[j]] for j in range(len(f_x)) if (x_min <= f_x[j] <= x_max and y_min <= f_y[j] <= y_max)])        
            ax.invert_yaxis()
            ax.scatter(plot_points[:, 0], plot_points[:, 1], s=5, marker='2', color='b')
            ax.plot(cliped_best[:, 0], cliped_best[:, 1], 'k-')
            ax.imshow(image[x_min:x_max, y_min:y_max], extent=[x_min, x_max, y_min, y_max], origin='upper', aspect='equal', cmap='gray')
            
def compare_gt_test(image, points, best, meshes, var_plot=True):
    # for points, test set
    pt_vars, mean_pt_x, mean_pt_y = [], [], []
    for i in range(len(best)):
        cluster = np.vstack(pt[i, :] for pt in points)
        mean_pt_x.append(np.mean(cluster[:, 0]))
        mean_pt_y.append(np.mean(cluster[:, 1]))
        pt_vars.append(np.var(cluster))
        
    index = np.argsort(pt_vars)
    index = index[::-1]
    
    
    f_x, f_y, f_clr = [], [], []
    
    if var_plot:
        fig, ax = plt.subplots()
        colormap = matplotlib.cm.jet(np.linspace(0, 1, 11))  
        for i in range(len(best)):
            clr_idx = int(i*10/len(best))
            f_x.append(mean_pt_x[index[i]])
            f_y.append(mean_pt_y[index[i]])
            f_clr.append(colormap[clr_idx])
        pcm = ax.scatter(f_x, f_y, s=3, marker='o', color=f_clr, alpha=1)
        fig.colorbar(pcm, ax=ax, boundaries=np.linspace(0, 1, 11))
        # ax.plot(best[:, 0], best[:, 1], 'w-')
        ax.imshow(image, cmap='gray', alpha=1)
        ax.set_xticks([])
        ax.set_yticks([])
        
        if False:
            min_var_idx, mid_var_idx, max_var_idx = index[-1], index[len(best) // 2], index[0]
            for idx in (min_var_idx, mid_var_idx, max_var_idx):
                plot_points = np.vstack(pt[idx, :] for pt in points)
                ax.plot(plot_points[:, 0], plot_points[:, 1])
    
    # for meshes, ground truth
    pt1_vars, mean1_pt_x, mean1_pt_y = [], [], []
    for i in range(len(best)):
        cluster = np.vstack(pt[i, :] for pt in meshes)
        mean1_pt_x.append(np.mean(cluster[:, 0]))
        mean1_pt_y.append(np.mean(cluster[:, 1]))
        pt1_vars.append(np.var(cluster))
        
    index1 = np.argsort(pt1_vars)
    index1 = index1[::-1]
    
    if var_plot:
        fig1, ax1 = plt.subplots()
        colormap = matplotlib.cm.jet(np.linspace(0, 1, 11))  
        f_x1, f_y1, f_clr1 = [], [], []
        
        for i in range(len(best)):
            clr_idx = int(i*10/len(best))
            f_x1.append(mean1_pt_x[index1[i]])
            f_y1.append(mean1_pt_y[index1[i]])
            f_clr1.append(colormap[clr_idx])
            
        pcm = ax1.scatter(f_x1, f_y1, s=3, marker='o', color=f_clr1, alpha=1)
        fig1.colorbar(pcm, ax=ax1, boundaries=np.linspace(0, 1, 11))
        # ax1.plot(best[:, 0], best[:, 1], 'w-')
        ax1.imshow(image, cmap='gray', alpha=1)
        ax1.set_xticks([])
        ax1.set_yticks([])
        
        if False:
            min_var_idx, mid_var_idx, max_var_idx = index1[-1], index1[len(best) // 2], index1[0]
            for idx in (min_var_idx, mid_var_idx, max_var_idx):
                plot_points = np.vstack(pt[idx, :] for pt in meshes)
            ax1.plot(plot_points[:, 0], plot_points[:, 1])
    
    # print variance comparision with details
    metrics = [['No.', 'GT', 'Test']]
    for j in range(len(pt_vars)):
        metrics.append([j, "%.2f" % pt1_vars[j], "%.2f" % pt_vars[j]])
    # print(AsciiTable(metrics).table)
    return AsciiTable(metrics).table, pt1_vars, pt_vars
    
def relation_vis(probs, losses):
    import pandas as pd
    arr = np.array([probs, losses]).T
    df = pd.DataFrame(arr, columns=['probability', 'loss'])
    sns.jointplot("probability", "loss", data=df, kind='reg')