import numpy as np
import scipy.io as sio
import utils as uts

import torch
from torch.utils.data import Dataset

class SpineDataset:
    def __init__(self, pth):
        with open(pth, "r") as file:
            self.filenames = file.readlines()
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, index):
        cur_pth = self.filenames[index % len(self.filenames)].rstrip()
        mat = sio.loadmat(cur_pth)
        
        img, disks = mat['img'], mat['disk']
        return img, disks


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    ds = SpineDataset(r"../dataset/train.txt")
    
    for i, (img, pts) in enumerate(ds):
        for pt in pts:
            fig, ax = plt.subplots()
            ax.plot()
        
            ax.imshow(img, cmap='gray')
            ax.plot(pt[:, 0], pt[:, 1], 'g-')
            ax.scatter(pt[[0,29,88,117], 0], pt[[0,29,88,117], 1], marker = 'o', color = 'b')
            
            fig1, ax1 = plt.subplots()
            ax1.plot()
            mask = uts.poly2mask(128, 128, pt)
            ax1.imshow(mask, cmap='gray')
            
        break
    