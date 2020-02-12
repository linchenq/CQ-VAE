import os
import tqdm
from sklearn.model_selection import train_test_split
from preprocess import Preprocessor

import cfgs

if __name__ == '__main__':
    #%% preprocess
    src, dst = "../source/", "../data/"
    for filename in tqdm.tqdm(os.listdir(src)):
        pre = Preprocessor(f"{src}{filename}",
                           f"{dst}{filename}",
                           norm=True, bone=False, crop=True)
        pre.forward()
    
    #%% train/test split
    dst_pth = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    
    train, valid, _, _ = train_test_split(cfgs.TRAIN, range(len(cfgs.TRAIN)), test_size=0.1, random_state=17)
    
    test = cfgs.TEST
    name_list, para_list = ['train', 'valid', 'test'], [train, valid, test]
    
    for name, para in zip(name_list, para_list):
        src_pth = os.path.join(dst_pth, "data")
        
        with open(os.path.join(dst_pth, f"{name}.txt"), 'w') as fs:
            print(f"generating {name} dataset...")
            
            for filename in os.listdir(src_pth):
                if filename.split('_')[0] in para:
                    out = str(os.path.join(src_pth, filename))
                    print(out, file=fs)