import random
import numpy as np
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

def linear_combination(cfg, target, meshes, random_seed):
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
        
        # mask = poly2mask(x_shape[0], x_shape[1], (pt+64).cpu().numpy())
        # mask = torch.from_numpy(mask).to(device)
        # masks.append(mask)
        
    return torch.stack(pts, dim=0)

def batch_linear_combination(cfg, target, meshes, random_seed):
    '''
    cfg: permutation based on step size, saved as .npy
    target: length of generated pts, the target size to generate
    meshes: (batch, len, pts.shape[0], pts.shape[1])
    '''
    pts = []
    for batch in meshes:
        pt = linear_combination(cfg, target, batch, random_seed)
        pts.append(pt)
        
    return torch.stack(pts, dim=0)

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