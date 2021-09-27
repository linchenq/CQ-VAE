import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from skimage.draw import polygon2mask

class Preprocessor():
    def __init__(self, src, dst, norm=True):
        self.disks = []
        self.img, self.best = None, None
        self.src = src
        self.dst = dst
        
        self.init_mats(norm)
        check = self.init_best()
        if not check:
            raise ValueError("Best shape NOT found")
        
    def init_mats(self, norm):
        imgs = []
        
        for filenames in os.listdir(self.src):
            cur_pth = f"{self.src}/{filenames}"
            mat, pts = self._mat2pts(cur_pth)
            self.disks.append(pts)
            
            imgs.append(mat['img'])
        
        for pre, post in zip(imgs[:-1], imgs[1:]):
            diff = np.abs(pre - post).sum()
            if diff > 1e-10:
                raise ValueError("Images are not consistnet")
        
        self.img = imgs[0]

        if norm:
            self.img = (self.img - np.min(self.img)) / (np.max(self.img) - np.min(self.img))
    
    def init_best(self):
        for filenames in os.listdir(self.src):
            cur_pth = f"{self.src}/{filenames}"
            if cur_pth.endswith("@best.mat"):
                mat, pts = self._mat2pts(cur_pth)
                self.best = pts
                
                return True
        return False
    
    def _mat2pts(self, mat_pth):
        if not os.path.exists(mat_pth):
            raise FileNotFoundError
        
        mat = sio.loadmat(mat_pth)
        pts = np.vstack((mat['disk_left'],
                         mat['disk_bot'],
                         mat['disk_right'],
                         mat['disk_up']))
        pts -= 64
        return mat, pts
    
    def forward(self):
        self.dic = {
            'img': self.img,
            'disk': self.disks,
            'best': self.best
        }
        if self.dst is not None:
            sio.savemat(self.dst, self.dic)
    
    def output(self, mat_pth):
        mat = sio.loadmat(mat_pth)
        img, disk, best = mat['img'], mat['disk'], mat['best']
        mark_index = [0, 29, 88, 117]
        
        fig, ax = plt.subplots()
        ax.imshow(img, cmap='gray')
        ax.plot(best[:, 0] + 64, best[:, 1] + 64, 'g-')
        ax.scatter(best[mark_index, 0] + 64, best[mark_index, 1] + 64, marker = 'o', color = 'b')
        
        for i in range(len(disk)):
            fig1, ax1 = plt.subplots()
            ax1.imshow(img, cmap='gray')
            ax1.plot(disk[i][:, 0] + 64, disk[i][:, 1] + 64, 'g-')
            ax1.scatter(disk[i][mark_index, 0] + 64, disk[i][mark_index, 1] + 64, marker = 'o', color = 'b')
            
            mask = np.transpose(polygon2mask((128, 128), disk[i] + 64)).astype(int)
            fig2, ax2 = plt.subplots()
            ax2.imshow(mask, cmap='gray')
            ax2.plot(disk[i][:, 0] + 64, disk[i][:, 1] + 64, 'g-')
            ax2.scatter(disk[i][mark_index, 0] + 64, disk[i][mark_index, 1] + 64, marker = 'o', color = 'b')
            
              
if __name__ == '__main__':
    src = "../source/Arntzen_1/"
    dst = "../data/Arntzen_1_combined.mat"
    pre = Preprocessor(src, dst, norm=True)
    pre.forward()
    pre.output(dst)

