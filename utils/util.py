import random
import numpy as np
import matplotlib.pyplot as plt
import torch

from skimage.draw import polygon2mask
from terminaltables import AsciiTable

def poly2mask(height, width, poly):
    return np.transpose(polygon2mask((height, width), poly)).astype(int)

def linear_combination(cfg, target, x_shape, meshes, device):
    '''
    meshes: w/o batch dims
    '''
    selection = np.load(cfg)
    indexes = random.sample(range(0, selection.shape[0]), target)
    
    pts, masks = [], []
    for index in indexes:
        gen_pt = [selection[index, i] * meshes[i] for i in range(len(meshes))]
        pt = torch.stack(gen_pt, dim=0).sum(dim=0)
        pts.append(pt)
        
        mask = poly2mask(x_shape[0], x_shape[1], pt.cpu().numpy())
        mask = torch.from_numpy(mask).to(device)
        masks.append(mask)
        
    return torch.stack(pts, dim=0), torch.stack(masks, dim=0)

def batch_linear_combination(cfg, target, x_shape, meshes, device):
    '''
    cfg: permutation based on step size, saved as .npy
    target: length of generated pts, the target size to generate
    meshes: (batch, len, pts.shape[0], pts.shape[1])
    '''
    pts, masks = [], []
    for batch in meshes:
        pt, mask = linear_combination(cfg, target, x_shape, batch, device)
        pts.append(pt)
        masks.append(mask)
        
    return torch.stack(pts, dim=0), torch.stack(masks, dim=0)
    
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
            fig.add_subplot(np.ceil(n_images/float(ncols)), ncols, i+1)
            plt.subplots_adjust(wspace=0, hspace=-0.7)
            plt.axis('off')
            if image.ndim == 2:
                plt.gray()
            plt.imshow(image)
            plt.plot(plts[i][:,0], plts[i][:,1], 'g-')
        # fig.set_size_inches(np.array(fig.get_size_inches())*n_images)
        plt.show()
    
def print_metrics(train, valid):
    metrics = [['Epoch', 'Train Loss', 'Valid Loss']]
    for epoch, loss in train.items():
        index = [ str(epoch),
                  loss,
                  valid[epoch] if epoch in valid.keys() else '---' ]
        metrics.append(index)
    return AsciiTable(metrics).table

def plot_loss(epoch, train, valid, filename=None):
    tx, ty = list(train.keys()), list(train.values())
    vx, vy = list(valid.keys()), list(valid.values())
    
    if epoch > 30:
        tx, ty = zip(*[(x, y) for x, y in zip(tx, ty) if y < 1e5])
        vx, vy = zip(*[(x, y) for x, y in zip(vx, vy) if y < 1e5])
    
    plt.plot(tx, ty, 'r--', label='Train Loss')
    plt.plot(vx, vy, 'b-', label='Valid Loss')
    
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    if filename is not None:
        plt.savefig(filename)
    # plt.show()
    
    
    
