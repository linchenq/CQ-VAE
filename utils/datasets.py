import numpy as np
from PIL import Image
from scipy.io import loadmat

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
        pts_pth = img_pth.replace("image", "mesh").replace(".jpg", ".mat")
        
        img = np.array(Image.open(img_pth).convert('L'))
        
        # fit the channel requirement only if grayscale (H, W)
        img = np.expand_dims(img, axis=0)
        img = (img - np.min(img)) / (np.max(img) - np.min(img))
        
        mat = loadmat(pts_pth)
        pts = np.vstack((mat['disk_landmarks'],
                         mat['disk_right'], mat['disk_up'],
                         mat['disk_left'], mat['disk_bot']))

        return img, pts

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    ds = SpineDataset(r"../dataset/train.txt")
    
    for i, (img, pts) in enumerate(ds):
        fig, ax = plt.subplots()
        ax.plot()
        ax.imshow(img[0], cmap='gray')
        ax.plot(pts[4:, 0], pts[4:, 1], 'g-')
        ax.scatter(pts[0:4, 0], pts[0:4, 1], marker = 'o', color = 'b')
        break
        