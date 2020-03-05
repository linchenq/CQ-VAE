import os
import tqdm
from sklearn.model_selection import train_test_split
from preprocess import Preprocessor

import cfgs

if __name__ == '__main__':
    augmented = False
    
    #%% preprocess
    src, dst = "../source", "../data"
    for folders in tqdm.tqdm(os.listdir(src)):
        if folders.startswith("AUGM"):
            if not augmented:
                continue

        pre = Preprocessor(f"{src}/{folders}",
                           f"{dst}/{folders}_combined.mat",
                           norm=True)
        pre.forward()
    
    #%% train/test split
    dst_pth = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    
    train, valid, _, _ = train_test_split(cfgs.TRAIN, range(len(cfgs.TRAIN)), test_size=0.1, random_state=17)
    
    test = cfgs.TEST
    name_list, para_list = ['train', 'valid', 'test'], [train, valid, test]
    out_dict = {'train': 0,
                'valid': 0,
                'test': 0}
    
    # Custom dataset
    for name, para in zip(name_list, para_list):
        src_pth = os.path.join(dst_pth, "data")
        
        with open(os.path.join(dst_pth, f"{name}.txt"), 'w') as fs:
            print(f"generating {name} dataset...")
            
            for filename in os.listdir(src_pth):
                if filename.split('_')[0] in para:
                    out = str(os.path.join(src_pth, filename))
                    print(out, file=fs)
                    out_dict[name] += 1
                    
            if name == 'train' and augmented:
                if filename.split('-')[0].startswith('AUGM'):
                    out = str(os.path.join(src_pth, filename))
                    print(out, file=fs)
                    out_dict[name] += 1
    
    print(out_dict)