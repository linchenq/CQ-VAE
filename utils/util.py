import random
import numpy as np
import matplotlib.pyplot as plt
import torch

from skimage.draw import polygon2mask
from terminaltables import AsciiTable

'''
INDEX:
    [1] poly2mask : convert boundary points to mask
    [2] [batch_]linear_combination: return tensors of generated pts/masks, with (batch)
    [3] [batch_]match_shape: TBD
    [4] dict_add, dict_mul: rewrite add/mul operation for dictionary
    [5] print_metrics: TBD
    [6] summary_metrics: TBD
'''
def poly2mask(height, width, poly):
    return np.transpose(polygon2mask((height, width), poly)).astype(int)

def linear_combination(cfg, target, x_shape, meshes, random_seed, device):
    '''
    meshes: w/o batch dims
    '''
    selection = np.load(cfg)
    random.seed(random_seed)
    indexes = random.sample(range(0, selection.shape[0]), target)
    
    pts, masks = [], []
    for index in indexes:
        gen_pt = [selection[index, i] * meshes[i] for i in range(len(meshes))]
        pt = torch.stack(gen_pt, dim=0).sum(dim=0)
        pts.append(pt)
        
        mask = poly2mask(x_shape[0], x_shape[1], pt.cpu().numpy())
        mask = torch.from_numpy(mask).to(device)
        masks.append(mask)
        
    return torch.stack(pts, dim=0), torch.stack(masks, dim=0).float()

def batch_linear_combination(cfg, target, x_shape, meshes, random_seed, device):
    '''
    cfg: permutation based on step size, saved as .npy
    target: length of generated pts, the target size to generate
    meshes: (batch, len, pts.shape[0], pts.shape[1])
    '''
    pts, masks = [], []
    for batch in meshes:
        pt, mask = linear_combination(cfg, target, x_shape, batch, random_seed, device)
        pts.append(pt)
        masks.append(mask)
        
    return torch.stack(pts, dim=0), torch.stack(masks, dim=0)

def match_shape(pred, data):
    '''
    pred: a set of shapes predicted from NN, viewed as tensor (length, ...)
    data: a set of shapes from ground truth, viewed as tensor (length, ...)
    ret_map: return a mapping from ground truth(data) to pred, i.e. ret_map[n] -> k while data[n] -> pred[k]
    '''
    if pred.shape[0] < data.shape[0]:
        raise ValueError('more data than predicted')
    pred, data = pred.view(pred.shape[0], -1), data.view(data.shape[0], -1)
    dist_map = torch.zeros((data.shape[0], pred.shape[0]), dtype=pred.dtype, device=pred.device)
    ret_map = torch.zeros(data.shape[0], dtype=torch.int64, device=pred.device)
    inf = torch.tensor(float('inf'), dtype=torch.float32, device=pred.device)
    
    for i in range(data.shape[0]):
        shape_i = data[i, :].view(1, -1)
        dist_map[i] = torch.sum((shape_i - pred)**2, dim=1)
    
    for i in range(data.shape[0]):
        min_v, col_indices = dist_map.min(dim=1)
        _, line = min_v.min(dim=0)
        col = col_indices[line]
        ret_map[line] = col
        
        dist_map[line, :] = inf
        dist_map[:, col] = inf
    
    return ret_map

def batch_match_shape(pred, data):
    if pred.shape[0] != data.shape[0]:
        raise ValueError('Unmatched batch size while generating maps')
    
    ret_map = []
    for batch_pred, batch_data in zip(pred, data):
        mapping = match_shape(batch_pred, batch_data)
        ret_map.append(mapping)
    
    return torch.stack(ret_map, dim=0)

def dict_add(base, x):
    for key in base.keys():
        base[key] += x[key]
    return base

def dict_mul(base, rate):
    for key in base.keys():
        base[key] *= rate
    return base

def print_metrics(epoch, met_dict):
    metrics = [['Epoch'] + list(met_dict.keys())]
    met_v = ["%.2f" % i for i in met_dict.values()]
    index = [str(epoch)] + [str(i) for i in met_v]
    metrics.append(index)
    
    return AsciiTable(metrics).table
        
def summary_metrics(train, valid):
    metrics = ['Epoch']
    for key in train[0].keys():
        metrics = metrics + [f"T_{key}"] + [f"V_{key}"]
    metrics = [metrics]
    
    for epoch in train.keys():
        index = [str(epoch)]
        for loss in train[epoch].keys():
            index.append(train[epoch][loss])
            if epoch in valid.keys() and loss in valid[epoch].keys():
                index.append(valid[epoch][loss])
            else:
                index.append('---')
        metrics.append(index)
    
    return AsciiTable(metrics).table

# def plot_loss(epoch, train, valid, filename=None):
#     tx, ty = list(train.keys()), list(train.values())
#     vx, vy = list(valid.keys()), list(valid.values())
    
#     if epoch > 30:
#         tx, ty = zip(*[(x, y) for x, y in zip(tx, ty) if y < 1e5])
#         vx, vy = zip(*[(x, y) for x, y in zip(vx, vy) if y < 1e5])
    
#     plt.plot(tx, ty, 'r--', label='Train Loss')
#     plt.plot(vx, vy, 'b-', label='Valid Loss')
    
#     plt.legend()
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
    
#     if filename is not None:
#         plt.savefig(filename)
#     # plt.show()
        
def show_images(images, ncols, plts=None):
    n_images = len(images)
    fig = plt.figure()
    
    if plts is None:
        for i, image in enumerate(images):
            fig.add_subplot(np.ceil(n_images/float(ncols)), ncols, i+1)
            if image.ndim == 2:
                plt.gray()
            plt.imshow(image)
        # fig.set_size_inches(np.array(fig.get_size_inches())*n_images)
        plt.show()
    
    else:
        assert(len(images) == len(plts))
        for i, (image, pts) in enumerate(zip(images, plts)):
            fig.add_subplot(np.ceil(n_images/float(ncols)), ncols, i+1， figsize=(64,64))
            plt.subplots_adjust(wspace=0.5, hspace=0.5)
            plt.axis('off')
            if image.ndim == 2:
                plt.gray()
            plt.imshow(image)
            plt.plot(plts[i][:,0], plts[i][:,1], 'g-')
        # fig.set_size_inches(np.array(fig.get_size_inches())*n_images)
        plt.show()
        
def show_image_tmp(images, points):
    n_images = len(images)
    
    for i in range(n_images):
        fig, ax = plt.subplots()
        ax.imshow(images[i])
        ax.plot(points[i][:,0], points[i][:,1], 'g-')
    
    
    
