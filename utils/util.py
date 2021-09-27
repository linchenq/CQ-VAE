import random
import numpy as np
import torch


# mapping generation
def _matching_shapes(pred, data):
    # pred: a set of shapes predicted from NN, viewed as tensor without batch (length, H, W)
    # data: ground truth, viewed as (length, H, W)
    # mapping: mapping from data(gt) to pred, i.e. mapping[n]->k <-> data[n]->pred[k]
    
    if pred.shape[0] < data.shape[0]:
        raise ValueError('CHECKING ERROR: Generated shapes should be largely more than ground truth shapes')
        
    pred, data = pred.view(pred.shape[0], -1), data.view(data.shape[0], -1)
    distance = torch.zeros((data.shape[0], pred.shape[0]), dtype=pred.dtype, device=pred.device)
    mapping = torch.zeros(data.shape[0], dtype=torch.int64, device=pred.device)
    inf = torch.tensor(float('inf'), dtype=torch.float32, device=pred.device)
    
    # disance: distance[i][j] = mse distance between data[i] and pred[j]
    for i in range(data.shape[0]):
        gt = data[i, :].view(1, -1)
        distance[i] = torch.sum((gt - pred)**2, dim=1)
    
    # mapping
    for i in range(data.shape[0]):
        min_rows, col_indices = torch.min(distance, dim=1)
        _, row = torch.min(min_rows, dim=0)
        mapping[row] = col_indices[row]
        
        distance[row, :] = inf
        distance[:, col_indices[row]] = inf
        
    return mapping

# mapping generation via batch size
def _batch_matching_shapes(pred, data):
    if pred.shape[0] != data.shape[0]:
        raise ValueError("CHECKING ERROR: Batch size between generated shapes and ground truth are not the same")
    
    mapping = []
    for batch_pred, batch_data in zip(pred, data):
        batch_mapping = _matching_shapes(batch_pred, batch_data)
        mapping.append(batch_mapping)
        
    return torch.stack(mapping, dim=0)

# Rewrite add operation for dictionary
def dict_add(base, x):
    for key in base.keys():
        base[key] += x[key]
    return base

# ground truth linear combination without batch size, in other words, tensors(L x 176 x 2)
def _linear_combination(cfg, size, meshes, random_seed):
    selection = np.load(cfg)
    random.seed(random_seed)
    indices = random.sample(range(0, selection.shape[0]), size)
    
    gts = []
    for idx in indices:
        gt = [selection[idx, i] * meshes[i] for i in range(len(meshes))]
        gts.append(torch.stack(gt, dim=0).sum(dim=0))
    
    return torch.stack(gts, dim=0)

# ground truth linear combination with batch size
# cfg: permutation based on step size, saved as .npy
# size: length of generated gts, the target size to extend ground truth
# meshes: (batch, len, H, W)
def _batch_lc(cfg, size, meshes, random_seed):
    gts = []
    for batch in meshes:
        gts.append(_linear_combination(cfg, size, batch, random_seed))
    
    return torch.stack(gts, dim=0)