import os
import numpy as np

import torch
from torch.utils.data import Dataset

class SpineDataset:
    def __init__(self, pth):
        with open(pth, "r") as file:
            self.img_files = file.readlines()
    
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, index):
        img_pth = self.img_files[index % len(self.img_files)].rstrip()
        pt_pth = img_pth.replace("image", "point", 2)
        
        image = np.load(img_pth)
        point = np.load(pt_pth)
        
        return image, point

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    ds = SpineDataset(r"../dataset/test.txt")
    for i, (img, pts) in enumerate(ds):
        fig, ax = plt.subplots()
        ax.plot()
        ax.imshow(img, cmap='gray')
        ax.plot(pts[:, 0], pts[:, 1], 'g-')
        