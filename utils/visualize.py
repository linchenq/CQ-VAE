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

def plot_1(image, points, pts_dict, best, alpha=0):
    sorted_keys = sorted(pts_dict.keys())
    min_key, max_key = sorted_keys[0], sorted_keys[-1]
    x, y = [pt[:, 0] for pt in points], [pt[:, 1] for pt in points]
    x, y = np.hstack(x), np.hstack(y)
    intergrate_points = np.vstack((x, y))
    set_cmap = matplotlib.cm.jet
    
    fig, ax = plt.subplots()
    sns.kdeplot(x, y, cbar=True,shade=True,cmap=set_cmap,shade_lowest=False,n_levels=5,alpha=1)
    ax.grid(linestyle='--')       
    ax.imshow(image, cmap='gray', alpha=alpha)
    ax.set_xticks([])
    ax.set_yticks([])
    
    # xmin, ymin = pts_dict[min_key][0][:, 0], pts_dict[min_key][0][:, 1]
    # xmax, ymax = pts_dict[max_key][0][:, 0], pts_dict[max_key][0][:, 1]
    # xmid, ymid = pts_dict[sorted_keys[len(sorted_keys)//2]][0][:, 0], pts_dict[sorted_keys[len(sorted_keys)//2]][0][:, 1]
    
    # for x, y in zip([xmin, xmid, xmax], [ymin, ymid, ymax]):
    #     fig, ax = plt.subplots()
    #     ax.plot(x, y, 'r-')
    #     ax.plot(best[:,0], best[:,1], 'g--')
    #     ax.imshow(image, cmap='gray')
    #     ax.set_xticks([])
    #     ax.set_yticks([])
        
def plot_2(image, points, pts_dict, best, alpha=1):
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
    ax.imshow(image, cmap='gray', alpha=alpha)
    ax.set_xticks([])
    ax.set_yticks([])
    
    if False:
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
        ax.imshow(image, cmap='gray', alpha=alpha)
        ax.set_xticks([])
        ax.set_yticks([])
        
        max_idx, mid_idx, min_idx = sort_index[0], sort_index[len(best) // 2], sort_index[-1]
        for idx in (min_idx, mid_idx, max_idx):
            plot_points = np.vstack(pt[idx, :] for pt in points)
            ax.plot(plot_points[:, 0], plot_points[:, 1])
    if False:
        sorted_keys = sorted(pts_dict.keys())
        min_key, max_key = sorted_keys[0], sorted_keys[-1]
        x, y = [pt[:, 0] for pt in points], [pt[:, 1] for pt in points]
        x, y = np.hstack(x), np.hstack(y)
        intergrate_points = np.vstack((x, y))
        set_cmap = matplotlib.cm.jet
        
        fig, ax = plt.subplots()
        pcm = ax.hexbin(x, y, gridsize=15, cmap=set_cmap, alpha=0.5)
        fig.colorbar(pcm, ax=ax)
        ax.imshow(image, cmap='gray', alpha=alpha)
        ax.set_xticks([])
        ax.set_yticks([])
        
    # max_idx, mid_idx, min_idx = sort_index[0], sort_index[len(best) // 2], sort_index[-1]
    # for idx in (min_idx, mid_idx, max_idx):
    #     fig, ax = plt.subplots()
    #     plot_points = np.vstack(pt[idx, :] for pt in points)
    #     x_min, x_max = int(np.min(plot_points[:, 0])), int(np.max(plot_points[:, 0])) + 1
    #     y_min, y_max = int(np.min(plot_points[:, 1])), int(np.max(plot_points[:, 1])) + 1
        
    #     sns.kdeplot(plot_points[:, 0], plot_points[:, 1],
    #                 cbar=True,
    #                 shade=True,
    #                 cmap=matplotlib.cm.jet,
    #                 shade_lowest=False,
    #                 n_levels=5,
    #                 alpha=0.5)
    #     max_prob = pts_dict[max(pts_dict.keys())][0]
    #     cliped_best = np.vstack([pt for pt in max_prob if (x_min <= pt[0] <= x_max and y_min <= pt[1] <= y_max)])        
    #     ax.invert_yaxis()
    #     ax.scatter(plot_points[:, 0], plot_points[:, 1], s=5, marker='2', color='b')
    #     ax.plot(cliped_best[:, 0], cliped_best[:, 1], 'k-')
    #     ax.imshow(image[x_min:x_max, y_min:y_max], extent=[x_min, x_max, y_min, y_max], origin='upper', aspect='equal', cmap='gray')
    #     ax.set_xticks([])
    #     ax.set_yticks([])
        
        
def heatmap_regression(image, points, pts_dict, best, alpha=0):
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
        ax.imshow(image, cmap='gray', alpha=alpha)
        
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
        ax.imshow(image, cmap='gray', alpha=alpha)
    
    # formula 3:
    if True:
        xi, yi = np.mgrid[x.min():x.max():x.size**0.5*1j, y.min():y.max():y.size**0.5*1j]
        k = gaussian_kde(np.vstack([x, y]))
        zi = k(np.vstack([xi.flatten(), yi.flatten()]))
        zi = (zi - np.min(zi)) / (np.max(zi) - np.min(zi))
        
        fig, ax = plt.subplots()
        pcm = ax.contourf(xi, yi, zi.reshape(xi.shape), cmap=set_cmap, alpha=0.5)
        fig.colorbar(pcm, ax=ax)
        ax.imshow(image, cmap='gray', alpha=alpha)
    
    # formula 4:
    if True:
        fig, ax = plt.subplots()
        pcm = ax.hexbin(x, y, gridsize=15, cmap=set_cmap, alpha=0.5)
        fig.colorbar(pcm, ax=ax)
        ax.imshow(image, cmap='gray', alpha=alpha)
    
    # formula 5:
    if False:
        step_sigma = 45
        
        Z, xedges, yedges = np.histogram2d(x, y, bins=len(x))
        Z = gaussian_filter(Z, sigma=step_sigma)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        
        fig, ax = plt.subplots()
        ax.pcolormesh(xedges, yedges, Z.T, alpha=0.5)
        ax.imshow(Z, cmap=set_cmap, alpha=alpha)
        
    # formula 6:
    # TBD

def heatmap_regression_probs(image, points, probs, best, alpha=0):
    sorted_keys = sorted(range(len(probs)), key=lambda k: probs[k], reverse=False)
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
        ax.imshow(image, cmap='gray', alpha=alpha)
        
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
        ax.imshow(image, cmap='gray', alpha=alpha)
    
    # formula 3:
    if True:
        xi, yi = np.mgrid[x.min():x.max():x.size**0.5*1j, y.min():y.max():y.size**0.5*1j]
        k = gaussian_kde(np.vstack([x, y]))
        zi = k(np.vstack([xi.flatten(), yi.flatten()]))
        zi = (zi - np.min(zi)) / (np.max(zi) - np.min(zi))
        
        fig, ax = plt.subplots()
        pcm = ax.contourf(xi, yi, zi.reshape(xi.shape), cmap=set_cmap, alpha=0.5)
        fig.colorbar(pcm, ax=ax)
        ax.imshow(image, cmap='gray', alpha=alpha)
    
    # formula 4:
    if True:
        fig, ax = plt.subplots()
        pcm = ax.hexbin(x, y, gridsize=15, cmap=set_cmap, alpha=0.5)
        fig.colorbar(pcm, ax=ax)
        ax.imshow(image, cmap='gray', alpha=alpha)
    
    # formula 5:
    if False:
        step_sigma = 45
        
        Z, xedges, yedges = np.histogram2d(x, y, bins=len(x))
        Z = gaussian_filter(Z, sigma=step_sigma)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        
        fig, ax = plt.subplots()
        ax.pcolormesh(xedges, yedges, Z.T, alpha=0.5)
        ax.imshow(Z, cmap=set_cmap, alpha=alpha)
        
    # formula 6:
    # TBD


def heatmap_boundary(image, points, pts_dict, best, batch_i, alpha=0):
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
    ax.imshow(image, cmap='gray', alpha=alpha)
    ax.set_title(f"{batch_i}")
    
    if True:
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
        ax.imshow(image, cmap='gray', alpha=alpha)
        ax.set_title(f"{batch_i}")
        
        max_idx, mid_idx, min_idx = sort_index[0], sort_index[len(best) // 2], sort_index[-1]
        for idx in (min_idx, mid_idx, max_idx):
            plot_points = np.vstack(pt[idx, :] for pt in points)
            ax.plot(plot_points[:, 0], plot_points[:, 1])
       
    # this part omits because of efficiency of the first evaluation
    '''
    max_idx, mid_idx, min_idx = sort_index[0], sort_index[len(best) // 2], sort_index[-1]
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
        ax.set_title(f"{batch_i}")
    '''

def show_images(image, points, probs, mode, batch_i, alpha=1):
    if mode == "tight":
        fig, ax = plt.subplots(nrows=1, ncols=len(points), figsize=(128, 128))
        for idx in range(len(points)):
            ax[idx].plot(points[idx][:, 0], points[idx][:, 1], 'g-')
            ax[idx].imshow(image, cmap='gray')
            ax[idx].set_title("%.5f" % probs[idx])
            ax[idx].set_xticks([])
            ax[idx].set_yticks([])
        plt.tight_layout()
        
    elif mode == "subplot":
        for pt, prob in zip(points, probs):
            fig, ax = plt.subplots()
            ax.plot(pt[:, 0], pt[:, 1], 'g-')
            ax.imshow(image, cmap='gray')
            ax.set_title("%.5f" % prob)
            ax.set_xticks([])
            ax.set_yticks([])
    elif mode == "sort":
        sorted_image(image, points, probs)
    else:
        raise NotImplementedError
            
def sorted_image(image, points, probs, alpha=1):
    sorted_idx = sorted(range(len(probs)), key=lambda k: probs[k])
    for idx in sorted_idx:
        fig, ax = plt.subplots()
        ax.plot(points[idx][:, 0], points[idx][:, 1], 'g-')
        ax.imshow(image, cmap='gray')
        ax.set_title("%.5f" % probs[idx])
        ax.set_xticks([])
        ax.set_yticks([])

def show_images_tight_local_mean(image, points, meshes, best_mesh, row, alpha=0):
    for pt in points:
        fig, ax = plt.subplots()
        ax.plot(pt[:, 0], pt[:, 1]-32, 'r-')
        ax.plot(best_mesh[:, 0], best_mesh[:, 1]-32, 'b--', linewidth=0.5)
        ax.imshow(image[32:96], cmap='gray', alpha=alpha)
        
    

def show_images_tight_local(image, points, probs, meshes, best_mesh, row, alpha=0):
    fig, ax = plt.subplots(nrows=1, ncols=len(points))
    sort_idx = sorted(range(len(probs)), key=lambda k: probs[k])
    for idx in sort_idx:
        ax[idx].plot(points[idx][:, 0], points[idx][:, 1]-32, 'r-')
        ax[idx].plot(best_mesh[:, 0], best_mesh[:, 1]-32, 'b--', linewidth=0.5, alpha=0.5)
        ax[idx].imshow(image[32:96], cmap='gray', alpha=alpha)
        #ax[idx].set_title("%.5f" % probs[idx])
        ax[idx].set_xticks([])
        ax[idx].set_yticks([])
    ax[0].set_title(f"{row}")
    plt.tight_layout()
    plt.subplots_adjust(wspace =0, hspace =0)
    fig.savefig(f"{row}.svg")


def show_images_tight(image, points, probs, meshes, best_mesh, alpha=1):
    fig, ax = plt.subplots(nrows=1, ncols=len(points))
    sort_idx = sorted(range(len(probs)), key=lambda k: probs[k])
    for idx in sort_idx:
        ax[idx].plot(points[idx][:, 0], points[idx][:, 1], 'r-')
        for mesh in meshes:
            ax[idx].plot(mesh[:, 0] + 64, mesh[:, 1] + 64, 'g--', alpha=0.5)
        ax[idx].imshow(image, cmap='gray', alpha=alpha)
        #ax[idx].set_title("%.5f" % probs[idx])
        ax[idx].set_xticks([])
        ax[idx].set_yticks([])
    plt.tight_layout()
    fig.savefig('zzz.svg')
   
def show_images_tmp(image, points, probs, mode, meshes, best_mesh, alpha=1):
    if mode == "tight":           
        fig, axs = plt.subplots(nrows=1, ncols=len(points))
        for idx in range(len(points)):
            axs[idx].plot(points[idx][:, 0], points[idx][:, 1], 'r-')
            for mesh in meshes:
                axs[idx].plot(mesh[:, 0]+64, mesh[:, 1]+64, 'g--', alpha=0.5)
            axs[idx].imshow(image, cmap='gray', alpha=alpha)
            # axs[0][idx].set_title("%.5f" % probs[idx])
            axs[idx].set_xticks([])
            axs[idx].set_yticks([])
        plt.tight_layout()
        
    elif mode == "subplot":
        for pt, prob in zip(points, probs):
            fig, ax = plt.subplots()
            ax.plot(pt[:, 0], pt[:, 1], 'r-')
            for mesh in meshes:
                ax.plot(mesh[:, 0]+64, mesh[:, 1]+64, 'g--', alpha=0.5)
            ax.imshow(image, cmap='gray', alpha=alpha)
            ax.set_title("%.5f" % prob)
            ax.set_xticks([])
            ax.set_yticks([])
    elif mode == "sort":
        sorted_image_tmp(image, points, probs, meshes, best_mesh)
    else:
        raise NotImplementedError
            
def sorted_image_tmp(image, points, probs, meshes, best_mesh, alpha=0):
    sorted_idx = sorted(range(len(probs)), key=lambda k: probs[k])
    for idx in sorted_idx:
        fig, ax = plt.subplots()
        ax.plot(points[idx][:, 0], points[idx][:, 1], 'r-')
        for mesh in meshes:
            ax.plot(mesh[:, 0]+64, mesh[:, 1]+64, 'g--', alpha=0.5)
        ax.imshow(image, cmap='gray', alpha=alpha)
        ax.set_title("%.5f" % probs[idx])
        ax.set_xticks([])
        ax.set_yticks([])
     
    
    
    
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
