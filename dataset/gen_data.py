import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import util.cfgs as cfgs
from util.preprocess import Preprocess


if __name__ == '__main__':
    # %% generate image and mesh .npy
    csv_f = pd.read_csv(cfgs.PATHS['index'], index_col='ID')
    for patient_id, row in csv_f.iterrows():
        dcm_pth = os.path.join(cfgs.PATHS['data'], patient_id, row['Number'])
        vtk_name = f"{patient_id}_SpineMesh2D.json.vtk"
        
        for mesh_r, ext in zip(cfgs.PATHS['mesh'], cfgs.PATHS['ext']):
            vtk_pth = os.path.join(mesh_r, vtk_name)
            if not (os.path.isfile(dcm_pth) and os.path.isfile(vtk_pth)):
                print(f"patient data lost: {patient_id}")
                break
            
            pre = Preprocess(dcm_path=dcm_pth, vtk_path=vtk_pth)
            pre.save(f"{patient_id}@{ext}", cfgs.PATHS['pair'])
    # end
            
    # %% split the whole dataset to txt
    train, valid, _, _ = train_test_split(cfgs.TRAIN, range(len(cfgs.TRAIN)), test_size=0.1, random_state=17)
    test = cfgs.TEST
    
    dir_pth = os.path.abspath(os.path.dirname(__file__))
    for name, para in zip(['train', 'valid', 'test'], [train, valid, test]):
        with open(os.path.join(dir_pth, f"{name}.txt"), "w") as fs:
            print(f"generateing {name} dataset...")
            
            for filename in os.listdir(cfgs.PATHS['r_img']):
                if filename.split('@')[0] in para:
                    out = str(os.path.join(cfgs.PATHS['r_img'], filename))
                    print(out, file=fs)
    # end