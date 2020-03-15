import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import seaborn as sns
from scipy.ndimage.filters import gaussian_filter
from scipy.stats.kde import gaussian_kde    
from scipy.interpolate import make_interp_spline

'''
Function: Regression task
'''
def heatmap_regression(image, points, pts_dict, best):
    sorted_keys = sorted(pts_dict.keys())
    min_key, max_key = sorted_keys[0], sorted_keys[-1]
    x, y = [pt[:, 0] for pt in points], [pt[:, 1] for pt in points]
    x, y = np.hstack(x), np.hstack(y)
    intergrate_points = np.vstack((x, y))
    set_cmap = matplotlib.cm.jet
    
    # formula 1: vinilla histogram2d: density map
    if True:
        Z, xedges, yedges = np.histogram2d(x, y)
        # Z = (Z - np.min(Z)) / (np.max(Z) - np.min(Z))

        fig, ax = plt.subplots()
        pcm = ax.pcolormesh(xedges, yedges, Z.T, 
                            cmap=set_cmap,
                            # norm=matplotlib.colors.Normalize(vmin=0., vmax=1.),
                            alpha=0.5)
        fig.colorbar(pcm, ax=ax)
        # ax.plot(pts_dict[min_key][0][:, 0], pts_dict[min_key][0][:, 1], 'g-', alpha=0.5)
        # ax.plot(pts_dict[max_key][0][:, 0], pts_dict[max_key][0][:, 1], 'g-', alpha=0.5)        
        # ax.plot(best[:, 0], best[:, 1], 'w-')
        ax.imshow(image, cmap='gray')
        
    # formula 2： 
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
        ax.imshow(image, cmap='gray')
    
    # formula 3:
    if True:
        xi, yi = np.mgrid[x.min():x.max():x.size**0.5*1j, y.min():y.max():y.size**0.5*1j]
        k = gaussian_kde(np.vstack([x, y]))
        zi = k(np.vstack([xi.flatten(), yi.flatten()]))
        zi = (zi - np.min(zi)) / (np.max(zi) - np.min(zi))
        
        fig, ax = plt.subplots()
        pcm = ax.contourf(xi, yi, zi.reshape(xi.shape), cmap=set_cmap, alpha=0.5)
        fig.colorbar(pcm, ax=ax)
        ax.imshow(image, cmap='gray')
    
    # formula 4:
    if True:
        fig, ax = plt.subplots()
        pcm = ax.hexbin(x, y, gridsize=15, cmap=set_cmap, alpha=0.5)
        fig.colorbar(pcm, ax=ax)
        ax.imshow(image, cmap='gray')
    
    # formula 5:
    if False:
        step_sigma = 45
        
        Z, xedges, yedges = np.histogram2d(x, y, bins=len(x))
        Z = gaussian_filter(Z, sigma=step_sigma)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        
        fig, ax = plt.subplots()
        ax.pcolormesh(xedges, yedges, Z.T, alpha=0.5)
        ax.imshow(Z, cmap=set_cmap)
        
    # formula 6:
    # TBD

def heatmap_boundary(image, points, pts_dict, best):
    pt_vars = []
    for index in range(len(best)):
        cluster = np.vstack(pt[index, :] for pt in points)
        pt_vars.append(np.var(cluster))
    sort_index = sorted(range(len(pt_vars)), key=lambda k: pt_vars[k], reverse=True)
    
    fig, ax = plt.subplots()
    colormap = matplotlib.cm.jet(np.linspace(0, 1, 11))
    f_x, f_y, f_clr = [], [], []
    for i, idx in enumerate(sort_index):
        clr_idx = int(i*10/len(best))
        f_x.append(best[idx, 0])
        f_y.append(best[idx, 1])
        f_clr.append(colormap[clr_idx])
    pcm = ax.scatter(f_x, f_y, s=3, marker='o', color=f_clr, alpha=0.5)
    fig.colorbar(pcm, ax=ax, boundaries=np.linspace(0, 1, 11))
    ax.imshow(image, cmap='gray')
    
    min_idx, mid_idx, max_idx = sort_index[0], sort_index[len(best) // 2], sort_index[-1]
    for idx in (min_idx, mid_idx, max_idx):
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
        max_prob = pts_dict[max(pts_dict.keys())][0]
        cliped_best = np.vstack([pt for pt in max_prob if (x_min <= pt[0] <= x_max and y_min <= pt[1] <= y_max)])        
        ax.invert_yaxis()
        ax.scatter(plot_points[:, 0], plot_points[:, 1], s=5, marker='2', color='b')
        ax.plot(cliped_best[:, 0], cliped_best[:, 1], 'k-')
        ax.imshow(image[x_min:x_max, y_min:y_max], extent=[x_min, x_max, y_min, y_max], origin='upper', aspect='equal', cmap='gray')

# '''
# Function: Regression task
#     heatmap_gen, heatmap_cor: Generate histogram2d form for discrete points, gaussian filter to soften them
# Argument:
#     As required
# '''

# def heatmap_gen(x, y, sig, bins):
#     heatmap, xedges, yedges = np.histogram2d(x, y, bins=bins)
#     heatmap = gaussian_filter(heatmap, sigma=sig)
#     extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    
#     return heatmap.T, extent

# def heatmap_cor(image, points, pts_dict):
#     sort_keys = sorted(pts_dict.keys())
#     min_key, max_key = sort_keys[0], sort_keys[-1]
    
#     x_cors, y_cors = np.array([]), np.array([])
#     for pt in points:
#         x_cors = np.append(x_cors, pt[:, 0])
#         y_cors = np.append(y_cors, pt[:, 1])
    
#     heatmap, extent = heatmap_gen(x_cors, y_cors, sig=64, bins=len(x_cors))
#     fig, ax = plt.subplots()
#     ax.imshow(heatmap, extent=extent, origin='lower', cmap=matplotlib.cm.jet)
#     fig1, ax1 = plt.subplots()
#     ax1.imshow(heatmap, extent=extent, origin='upper', cmap=matplotlib.cm.jet)


# '''
# Function: Regression task
#     heatmap: Plot all the boundaries in only one image
# Arguments:
#     image: original image
#     pts_dict: dictionary where key is the probability and value is a list containing all possible shape
    
# '''
# def heatmap(image, pts_dict, n_class=None):
#     sort_keys = sorted(pts_dict.keys())
#     min_key, max_key = sort_keys[0], sort_keys[-1]
#     if n_class is None:
#         n_class = len(sort_keys)
    
#     colormap = cm.jet(np.linspace(0, 1, n_class+1))
    
#     # Equalize the number of container   
#     fig, ax = plt.subplots()
#     ax.imshow(image, cmap='gray')
#     # ax.imshow(np.zeros((1000, 100)))
#     for i, pts_key in enumerate(sort_keys):
#         clr_idx = int(i * n_class / len(sort_keys))
#         clr_alpha = clr_idx / n_class
#         for mesh in pts_dict[pts_key]:
#             ax.plot(mesh[:, 0], mesh[:, 1], color=colormap[clr_idx], alpha=clr_alpha)
    
#     # Mimic the true distribution
#     fig1, ax1 = plt.subplots()
#     ax1.imshow(image, cmap='gray')
#     # ax1.imshow(np.zeros((1000, 100)))
#     for pts_key in sort_keys:
#         clr_idx = int((pts_key - min_key) * n_class / (max_key - min_key))
#         clr_alpha = clr_idx / n_class
#         for mesh in pts_dict[pts_key]:
#             ax1.plot(mesh[:, 0], mesh[:, 1], color=colormap[clr_idx], alpha=clr_alpha)


'''
Function: Classification task
'''
def heatmap_seg(image, masks):
    pass