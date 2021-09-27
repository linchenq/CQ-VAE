# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 17:44:26 2020

@author: liang
"""
import numpy as np
import matplotlib.pyplot as plt
import torch
from Lumbar_Dataset import DiskSet
from sklearn.decomposition import PCA
from PolyToImage2D import PolyToImage2D
#%%
class PCA_Aug():
    def __init__(self, n_components, c_max, flag):
        path='../../data/Lumbar/UM100_disk/'
        Dataset_train = DiskSet(path, '400_train.txt', return_bon=True)
        loader_train = torch.utils.data.DataLoader(dataset=Dataset_train,batch_size = 64, shuffle = False)    
        Sall=[]
        Xall=[]
        for batch_idx, (X, S) in enumerate(loader_train):
            Xall.append(X.numpy())
            Sall.append(S.numpy())            
        Xall=np.concatenate(Xall, axis=0) 
        Sall=np.concatenate(Sall, axis=0) 
        pca = PCA(n_components=n_components)
        pca.fit(Sall.reshape(Sall.shape[0], -1))
        self.pca=pca
        self.P=torch.tensor(pca.components_.reshape(1,n_components,-1), dtype=torch.float32)
        self.V=torch.tensor(np.sqrt(pca.explained_variance_).reshape(1,n_components,1), dtype=torch.float32)
        self.Smean=torch.tensor(np.mean(Sall, axis=0, keepdims=True), dtype=torch.float32)
        self.X=torch.tensor(Xall, dtype=torch.float32)
        self.S=torch.tensor(Sall, dtype=torch.float32)
        self.n_components=n_components
        self.c_max=c_max
        self.flag=flag
        border=[]
        for x in [1, 128]:
            for y in [1, 32, 64, 96, 128]:
                border.append([x, y])
                #pass
        for y in [1, 128]:
            for x in [32, 64, 96]:
                border.append([x, y])
                #pass
        self.border=torch.tensor(border, dtype=torch.float32).view(1,-1,2)-0.5
        self.rng1=torch.Generator()
        self.rng1.manual_seed(1)
        self.rng2=torch.Generator()
        self.rng2.manual_seed(2)
        
    def to(self, sth):
        #sth is device or dtype
        self.P=self.P.to(sth)
        self.V=self.V.to(sth)
        self.Smean=self.Smean.to(sth)
        self.border=self.border.to(sth)
        if isinstance(sth, torch.device):
            self.rng1=torch.Generator(sth)
            self.rng1.manual_seed(1)
            self.rng2=torch.Generator(sth)
            self.rng2.manual_seed(2)

    def add_boarder(self, s):        
        border= self.border.expand(s.shape[0], self.border.shape[1], self.border.shape[2])
        s=torch.cat([s, border], dim=1)
        return s

    def generate_shape(self, batch_size):    
        r=-self.c_max+2*self.c_max*torch.rand(batch_size,self.n_components,1, 
                                              dtype=self.P.dtype, device=self.P.device, generator=self.rng2)
        s=torch.sum(r*self.V*self.P, dim=1)
        s=s.view(batch_size,-1,2)+self.Smean
        return s

    def generate_image(self, s, x_real, s_real):
        poly2image=PolyToImage2D(self.add_boarder(s_real), x_real, origin=[0.5, 0.5], swap_xy=False)
        x=poly2image(self.add_boarder(s))
        return x

    def __call__(self, batch_size):
        dtype=self.P.dtype
        device=self.P.device
        if self.flag==0:
            idx=torch.randint(0, self.X.shape[0], (batch_size,), device=device, generator=self.rng1)
            x_real=self.X[idx].to(dtype).to(device)
            s_real=self.S[idx].to(dtype).to(device)
            s=self.generate_shape(batch_size)
            x=self.generate_image(s, x_real, s_real)
        elif self.flag==1:
            idx=torch.randint(0, self.X.shape[0], (1,), device=device, generator=self.rng1)
            x_real=self.X[idx].to(dtype).to(device)
            s_real=self.S[idx].to(dtype).to(device)
            s=self.generate_shape(batch_size)
            x=self.generate_image(s, x_real, s_real)
        elif self.flag==2:
            idx=torch.randint(0, self.X.shape[0], (batch_size,), device=device, generator=self.rng1)
            x_real=self.X[idx].to(dtype).to(device)
            s_real=self.S[idx].to(dtype).to(device)
            s=self.generate_shape(1)
            s=s.expand(batch_size, -1, 2)
            x=self.generate_image(s, x_real, s_real)
        else:
            raise ValueError('uknown flag')
        s=s[:,0:176,:]
        return x, s
#%%
from skimage.draw import polygon2mask
def poly_disk(S):
    device=S.device
    S=S.detach().cpu().numpy()
    Mask = np.zeros((S.shape[0],1,128,128), dtype=np.float32)
    for i in range(S.shape[0]):
        Mask[i,0]=polygon2mask((128,128), S[i])
        Mask[i,0]=np.transpose(Mask[i,0])
    Mask = torch.tensor(Mask, dtype=torch.float32, device=device)
    return Mask
#%%
def augXS(X, S):
    M=poly_disk(S)
    beta=torch.rand(1).item()
    eps=0.001
    Noise=torch.rand_like(X)
    X=X*(M>eps)+(beta*X+(1-beta)*Noise)*(M<=eps)
    X.clamp_(0,1)
    return X
#%%
if __name__ == '__main__':
    #%%
    pca_aug=PCA_Aug(n_components=20, c_max=2, flag=0)
    '''
    torch.save({'P':pca_aug.P,
                'V':pca_aug.V,
                'Smean':pca_aug.Smean,
                'S':pca_aug.S,
                'X':pca_aug.X},
               'result/PCAn5c2f0.pt')    
    '''
    #%%
    P=pca_aug.P
    V=pca_aug.V
    S=pca_aug.S
    S=S.view(S.shape[0], 1, -1)
    Smean=pca_aug.Smean.view(1, 1, -1)
    C=torch.sum((S-Smean)*P/V, dim=2).numpy().reshape(-1)
    fig, ax = plt.subplots()
    ax.hist(C, bins=100)
    print(pca_aug.pca.explained_variance_ratio_.sum())
    #%%
    fig, ax = plt.subplots()
    for n in range(100):
        ax.cla()
        x,s=pca_aug(1)        
        x=augXS(x,s)
        ax.plot(s[:,:,0], s[:,:,1], 'b.')
        ax.set_xlim(0.5,127.5)
        ax.set_ylim(0.5,127.5)
        ax.imshow(x[0].cpu().numpy().reshape(128,128), cmap='gray')
        ax.set_aspect('equal')
        plt.pause(1)
        plt.draw()
        if x.mean().item()<0.1:
            break # it means we need to use torch.float64