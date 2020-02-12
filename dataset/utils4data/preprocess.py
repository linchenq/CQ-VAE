import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from skimage.draw import polygon2mask

class Preprocessor():
    def __init__(self, src, dst, norm=True, bone=False, crop=True):
        self.mat = sio.loadmat(src)
        self.num_attr = len(self.mat.keys())
        self.img = self.mat['img']
        self.dst = dst
        
        # normalization on image
        if norm:
            self.img = (self.img - np.min(self.img)) / (np.max(self.img) - np.min(self.img))
        # generate landmarks, boundary, and its mask
        self.init_boundary(bone)
        # crop the whole 
        if crop:
            self.init_crop()
    
    def init_boundary(self, bone):
        self.landmarks = self.mat['disk_landmarks']        
        self.boundary = np.vstack((self.mat['disk_left'],
                                    self.mat['disk_bot'],
                                    self.mat['disk_right'],
                                    self.mat['disk_up']))
        if bone:
            out_bound = np.vstack((self.mat['up_bon_left'], self.mat['disk_left'], self.mat['bot_bon_left'],
                                   self.mat['bot_bon_low'],
                                   self.mat['bot_bon_right'], self.mat['disk_right'], self.mat['up_bon_right'],
                                   self.mat['up_bon_top']
                ))
            self.mask = np.transpose(polygon2mask((128, 128), out_bound))
        else:
            self.mask = np.transpose(polygon2mask((128, 128), self.boundary))
    
    def init_crop(self):
        self.img = self.img[32:96, :]
        self.landmarks[:, 1] -= 32.0
        self.boundary[:, 1] -= 32.0
        self.mask = self.mask[32:96, :]
        
        if (self.boundary < 0).any():
            raise ValueError("[CROP ERROR] Invalid boundary out of index")
    
    def forward(self):
        self.dic = {
            "img": self.img,
            "landmarks": self.landmarks,
            "disk": self.boundary,
            "mask": self.mask.astype(int)
        }
        if self.dst is not None:
            sio.savemat(self.dst, self.dic)
    
    def output(self):
        fig, ax = plt.subplots()
        ax.plot()
        
        ax.imshow(self.dic["img"], cmap='gray')
        ax.plot(self.dic["disk"][:, 0], self.dic["disk"][:, 1], 'g-')
        ax.scatter(self.dic["landmarks"][:, 0], self.dic["landmarks"][:, 1], marker='o', color='b')
        
        fig1, ax1 = plt.subplots()
        ax1.plot()
        ax1.imshow(self.dic["mask"], cmap='gray')
        
    
if __name__ == '__main__':
    src = "../source/Allison_1_liang.mat"
    dst = "../data/Allison_1_liang.mat"
    pre = Preprocessor(src, dst, norm=True, bone=True, crop=True)
    pre.forward()
    pre.output()

