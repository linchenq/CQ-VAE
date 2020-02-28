import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from skimage.draw import polygon2mask
'''
Important Comment for difference between matlab and python
----------------------------------------------------------
I(matlab) -> I(matlab centered) -> I(python)
  (x,y)        (x-64, y-64)     (x-0.5, y-0.5)
----------------------------------------------------------
for training phase:
    1. mesh points (x-64, y-64)
    2. mask (x-0.5, y-0.5)
for inference phase:
    1. mesh points (x_pred - 0.5, y_pred - 0.5) for plot
    2. mask (x, y)

'''

'''
Preprocess:
    init_check: check if image is consistent, check file path
    init_norm: normalization on images
    init_boundary: inital boundary points, decide whether the mask contains bone or not
    init_difference: centered difference for points
'''
class Preprocessor():
    def __init__(self, src, dst, norm=True, keep_bone=False):
        self.mats, self.img = [], None
        self.src, self.dst = src, dst
    
        if not self.init_check():
            raise ValueError("Images are not consistent")
        
        self.init_norm(norm)
        
        self.init_boundary(keep_bone)
        
    def init_check(self):
        for filenames in os.listdir(self.src):
            if not os.path.exists(f"{self.src}/{filenames}"):
                raise FileNotFoundError
            self.mats.append(sio.loadmat(f"{self.src}/{filenames}"))
        
        # double check if images are the same
        imgs = []
        for mat in self.mats:
            if 'img' in mat:
                imgs.append(mat['img'])
        for index in range(0, len(imgs) - 1):
            if not (imgs[index] == imgs[index + 1]).all():
                return False
        self.img = imgs[0]
        
        return True  
    
    def init_norm(self, norm):
        if norm:
            self.img = (self.img - np.min(self.img)) / (np.max(self.img) - np.min(self.img))
    
    def init_boundary(self, keep_bone):
        self.landmarks, self.boundary, self.mask = [], [], []
        
        for mat in self.mats:
            self.landmarks.append(mat['disk_landmarks'])
            self.boundary.append(np.vstack((mat['disk_left'],
                                            mat['disk_bot'],
                                            mat['disk_right'],
                                            mat['disk_up'])))
            if keep_bone:
                out_bound = np.vstack((mat['up_bon_left'], mat['disk_left'], mat['bot_bon_left'],
                                   mat['bot_bon_low'],
                                   mat['bot_bon_right'], mat['disk_right'], mat['up_bon_right'],
                                   mat['up_bon_top']))
                self.mask.append(out_bound)
            else:
                self.mask.append(self.boundary[-1])
                
        self.landmarks = [ele - 64 for ele in self.landmarks]
        self.boundary = [ele - 64 for ele in self.boundary]
        
    def forward(self):
        self.dic = {
            "img": self.img,
            "landmarks": self.landmarks,
            "disk": self.boundary,
            "mask": self.mask
        }
        if self.dst is not None:
            sio.savemat(self.dst, self.dic)
    
    def output(self):
        for i in range(len(self.dic["disk"])):
            fig, ax = plt.subplots()
            ax.plot()
            
            ax.imshow(self.dic["img"], cmap='gray')
            ax.plot(self.dic["disk"][i][:, 0], self.dic["disk"][i][:, 1], 'g-')
            ax.scatter(self.dic["landmarks"][i][:, 0], self.dic["landmarks"][i][:, 1], marker='o', color='b')
            
            fig1, ax1 = plt.subplots()
            ax1.plot()
            mask = np.transpose(polygon2mask((128, 128), self.dic["mask"][i]))
            ax1.imshow(mask, cmap='gray')
            ax1.plot(self.dic["disk"][i][:, 0], self.dic["disk"][i][:, 1], 'g-')
            ax1.scatter(self.dic["landmarks"][i][:, 0], self.dic["landmarks"][i][:, 1], marker='o', color='b')

        
    
if __name__ == '__main__':
    src = "../source/Arntzen_1/"
    dst = "../data/Arntzen_1_combined.mat"
    pre = Preprocessor(src, dst, norm=True, keep_bone=False)
    pre.forward()
    pre.output()

