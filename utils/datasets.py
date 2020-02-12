import numpy as np
import scipy.io as sio

import torch
from torch.utils.data import Dataset

class SpineDataset:
    def __init__(self, pth):
        with open(pth, "r") as file:
            self.mat_list = file.readlines()
    
    def __len__(self):
        return len(self.mat_list)
    
    def __getitem__(self, index):
        mat = sio.loadmat(self.mat_list[index % len(self.mat_list)].rstrip())
        img, mask = mat['img'], mat['mask']
        landmarks, disk = mat['landmarks'], mat['disk']        
        
        # fit the channel requirement
        img = np.expand_dims(img, axis=0)
        # mask operations
        mask = np.expand_dims(mask, axis=0)
        
        return img, mask, disk

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    ds = SpineDataset(r"../dataset/train.txt")
    
    for i, (img, mask, pts) in enumerate(ds):
        
        fig, ax = plt.subplots()
        ax.plot()
        ax.imshow(img[0], cmap='gray')
        ax.plot(pts[:, 0], pts[:, 1], 'g-')
        ax.scatter(pts[[0,29,88,117], 0], pts[[0,29,88,117], 1], marker = 'o', color = 'b')
        
        fig1, ax1 = plt.subplots()
        ax1.plot()
        ax1.imshow(mask[0], cmap='gray')
        
        break
    