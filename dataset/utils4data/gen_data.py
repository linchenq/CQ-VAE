import os
from sklearn.model_selection import train_test_split

import cfgs

          
if __name__ == '__main__':
    #%% train/test split
    dst_pth = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    
    train, valid, _, _ = train_test_split(cfgs.TRAIN, range(len(cfgs.TRAIN)), test_size=0.1, random_state=17)
    
    test = cfgs.TEST
    name_list, para_list = ['train', 'valid', 'test'], [train, valid, test]
    
    for name, para in zip(name_list, para_list):
        img_folder = os.path.join(dst_pth, "image")
        
        with open(os.path.join(dst_pth, f"{name}.txt"), 'w') as fs:
            print(f"generating {name} dataset...")
            
            for filename in os.listdir(img_folder):
                if filename.split('_')[0] in para:
                    out = str(os.path.join(img_folder, filename))
                    print(out, file=fs)