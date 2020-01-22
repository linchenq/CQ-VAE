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
        pts_pth = img_pth.replace("image", "point")
        
        img, pts = np.load(img_pth), np.load(pts_pth)
        # pts = np.expand_dims(pts, axis=0)
        
        # fit the channel requiremend
        img = np.expand_dims(img, axis=0)
        
        # normalize each data by min/max value
        img = (img - np.min(img)) / (np.max(img) - np.min(img))
        
        return img, pts

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    ds = SpineDataset(r"../dataset/test.txt")
    
    # 0， 29, 88， 117
    for i, (img, pts) in enumerate(ds):
        fig, ax = plt.subplots()
        ax.plot()
        ax.imshow(img[0], cmap='gray')
        ax.plot(pts[0, :, 0], pts[0, :, 1], 'g-')
        break
        