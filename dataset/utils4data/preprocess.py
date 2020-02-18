import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from skimage.draw import polygon2mask

class Preprocessor():
    def __init__(self, src, dst, norm=True, bone=False, crop=False):
        self.mats, self.img = [], None
        self.src, self.dst = src, dst
    
        if not self.init_check():
            raise NotImplementedError
        
        self.init_norm(norm)
        self.init_boundary(bone)
        self.init_crop(crop)
        
    def init_check(self):
        for filenames in os.listdir(self.src):
            if not os.path.exists(f"{self.src}/{filenames}"):
                raise FileNotFoundError
            self.mats.append(sio.loadmat(f"{self.src}/{filenames}"))
            if filenames.endswith("@liang.mat"):
                self.img = self.mats[-1]['img']
        
        # double check if images are the same
        # imgs = []
        # for mat in self.mats:
        #     if 'img' in mat:
        #         imgs.append(mat['img'])
        # for index in range(0, len(imgs) - 1):
        #     if not (imgs[index] == imgs[index + 1]).all():
        #         return False
        # self.img = imgs[0]
        
        return True  
    
    def init_norm(self, norm):
        if norm:
            self.img = (self.img - np.min(self.img)) / (np.max(self.img) - np.min(self.img))
    
    def init_boundary(self, bone):
        self.landmarks, self.boundary, self.mask = [], [], []
        for mat in self.mats:
            self.landmarks.append(mat['disk_landmarks'])
            self.boundary.append(np.vstack((mat['disk_left'],
                                            mat['disk_bot'],
                                            mat['disk_right'],
                                            mat['disk_up'])))
            if bone:
                out_bound = np.vstack((mat['up_bon_left'], mat['disk_left'], mat['bot_bon_left'],
                                   mat['bot_bon_low'],
                                   mat['bot_bon_right'], mat['disk_right'], mat['up_bon_right'],
                                   mat['up_bon_top']))
                self.mask.append(out_bound)
                # self.mask.append(np.transpose(polygon2mask((128, 128), out_bound)))
            else:
                self.mask.append(self.boundary[-1])
                # self.mask.append(np.transpose(polygon2mask((128, 128), self.boundary[-1])))
    
    def init_crop(self, crop):
        if crop:
            self.img = self.img[32:96, :]
            for ele_l, ele_b, ele_m in zip(self.landmarks, self.boundary, self.mask):
                ele_l[:, 1] -= 32.0
                ele_b[:, 1] -= 32.0
                ele_m[:, 1] -= 32.0
            
                if (ele_b < 0).any():
                    raise ValueError("[CROP ERROR] Invalid boundary out of index")
        
    def forward(self):
        self.mask = [mask.astype(int) for mask in self.mask]
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
            # ax1.imshow(self.dic["mask"][i], cmap='gray')
        
    
if __name__ == '__main__':
    src = "../source/Arntzen_1/"
    dst = "../data/Allison_1_combined.mat"
    pre = Preprocessor(src, dst, norm=True, bone=False, crop=False)
    pre.forward()
    pre.output()

