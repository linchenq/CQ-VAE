# import numpy as np
import scipy.io as sio
import torch

class SpineDataset(torch.utils.data.Dataset):
    def __init__(self, path):
        self.path = path
        with open(path, "r") as f:
            self.filelist = f.readlines()
    
    def __len__(self):
        return len(self.filelist)
    
    def __getitem__(self, index):
        cur_path = self.filelist[index % len(self.filelist)].rstrip()
        data = sio.loadmat(cur_path)
        
        # extract image and best, reshape as (C, H, W)
        img, meshes, best = data['img'], data['disk'], data['best']
        img = img.reshape(1, img.shape[0], img.shape[1])
        
        # normalization and torch
        img = (img - img.min())/(img.max() - img.min())
        img = torch.tensor(img, dtype=torch.float32)
        best = torch.tensor(best, dtype=torch.float32)
        meshes = torch.tensor(meshes, dtype=torch.float32)
        
        return img, meshes, best


if __name__ == '__main__':
    debug = True
    import matplotlib.pyplot as plt
    dataset = SpineDataset(r'../dataset/train.txt')
    print(f"len function: Dataset Length is {len(dataset)}")

    if not debug:
        for i, (img, meshes, best) in enumerate(dataset):
            print(f"img {i}: Image shape is {img.shape}")
            print(f"meshes {i}: Best shape is {meshes.shape}")
            print(f"best {i}: Best shape is {best.shape}")
            
            fig, ax = plt.subplots()
            ax.imshow(img[0, :, :], cmap='gray')
            ax.plot(best[:, 0] + 64, best[:, 1] + 64, 'r-')
        
            if i >= 2:
                print("Only plot limited iterations")
                break
    
    if debug:
        dataset = {}
        for param in ['train', 'valid', 'test']:
            dataset[param] = SpineDataset(f"../dataset/{param}.txt")
        from torch.utils.data import DataLoader
        dataloader = {
            'train': DataLoader(dataset['train'], batch_size=3, shuffle=True),
            'valid': DataLoader(dataset['valid'], batch_size=3, shuffle=True),
            'test': DataLoader(dataset['test'], batch_size=3, shuffle=False)
        }
        print(f"len function: Dataloader Length is {len(dataloader['train'])}")
        
        for batch_i, (data_A, data_B, data_C) in enumerate(zip(dataloader['train'], dataloader['valid'], dataloader['test'])):
            print(len(data_A))
            print(len(data_B))
            print(len(data_C))
            break