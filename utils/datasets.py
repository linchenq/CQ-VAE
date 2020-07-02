import scipy.io as sio
import utils.util as uts

class SpineDataset:
    def __init__(self, pth):
        with open(pth, "r") as file:
            self.filenames = file.readlines()
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, index):
        cur_pth = self.filenames[index % len(self.filenames)].rstrip()
        cur_mat = sio.loadmat(cur_pth)
        
        img = cur_mat['img']
        disks = cur_mat['disk']
        best = cur_mat['best']
        # best_mask = uts.poly2mask(img.shape[0], img.shape[1], best + 64)
        
        return img, disks, best
    

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    ds = SpineDataset(r'../dataset/train.txt')
    # max_size = len(ds)
    # for i, (img, pts, best, best_mask, cur_pth) in enumerate(ds):
    #     print(i)
    #     if i >= max_size + 1:
    #         break
        
    #     fig, ax = plt.subplots()
    #     title = cur_pth.split('\\')[-1]
    #     fig.suptitle(f"{title}")
    #     ax.imshow(img, cmap='gray')
        
    #     for pt in pts:
    #         ax.plot(pt[:, 0] + 64, pt[:, 1] + 64, 'g-')
    
    for i, (img, pts, best) in enumerate(ds):
        import numpy as np
        for pt in pts:                
                fig1, ax1 = plt.subplots()
                ax1.imshow(img, cmap='gray')
                ax1.plot(pt[:, 0]+64, pt[:, 1]+64, 'r-')
                ax1.imshow(img, cmap='gray', alpha=0)
        if i>=3:
                
            break
    
    # for i, (img, pts, best) in enumerate(ds):
    #     fig, ax = plt.subplots()
    #     ax.imshow(img, cmap='gray')
    #     ax.plot(best[:, 0] + 64, best[:, 1] + 64, 'g-')
    #     # ax.scatter(best[[0,29,88,117], 0] + 64, best[[0,29,88,117], 1] + 64, marker = 'o', color = 'b')
        
    #     # ax.imshow(best_mask, cmap='gray', alpha=0.5)
        
    #     for pt in pts:
    #         fig1, ax1 = plt.subplots()
    #         ax1.imshow(img, cmap='gray')
    #         ax1.plot(pt[:, 0] + 64, pt[:, 1] + 64, 'g-')
    #         ax1.scatter(pt[[0,29,88,117], 0] + 64, pt[[0,29,88,117], 1] + 64, marker = 'o', color = 'b')
            
    #         mask = uts.poly2mask(img.shape[0], img.shape[1], pt + 64)
    #         ax1.imshow(mask, cmap='gray', alpha=0.5)
        
    #     break