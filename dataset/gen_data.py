import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from utils.preprocess import Preprocess

PATHS = {
    # IMAGE/MESH GENERATION
    'index': 'UM100/index.csv',
    'data': 'UM100/src/data/',
    'mesh': ['UM100/src/spinemesh@jason/', 'UM100/src/spinemesh@liang/'],
    'ext': ['jason', 'liang'],
    'regression': ['UM100/regression/image/', 'UM100/regression/point/']
}


if __name__ == '__main__':
    
    # %% generate image and mesh .npy
    csv_f = pd.read_csv(PATHS['index'], index_col='ID')
    for patient_id, row in csv_f.iterrows():
        dcm_pth = os.path.join(PATHS['data'], patient_id, row['Number'])
        vtk_name = f"{patient_id}_SpineMesh2D.json.vtk"
        
        for mesh_r, ext in zip(PATHS['mesh'], PATHS['ext']):
            vtk_pth = os.path.join(mesh_r, vtk_name)
            if not (os.path.isfile(dcm_pth) and os.path.isfile(vtk_pth)):
                print(f"patient data lost: {patient_id}")
                break                
            
            pre = Preprocess(dcm_path=dcm_pth, vtk_path=vtk_pth)
            pre.save(f"{patient_id}@{ext}", PATHS['regression'])
            
    # end
            
    # %% split the whole dataset to txt
    train, test, _, _ = train_test_split(os.listdir(PATHS['regression'][0]), range(0, len(os.listdir(PATHS['regression'][0]))), test_size=0.1, random_state=17)
    train, valid, _, _ = train_test_split(train, range(len(train)), test_size=0.1, random_state=17)
    dir_pth = os.path.abspath(os.path.dirname(__file__))
    for name, para in zip(['train', 'valid', 'test'], [train, valid, test]):
        with open(os.path.join(dir_pth, f"{name}.txt"), "w") as fs:
            print(f"generateing {name} dataset...")
            root = "C:/Research/LumbarSpine/Github/probalistic-unet/dataset/UM100/regression/image/"
            for filename in para:
                out = str(os.path.join(root, filename))
                print(out, file=fs)
                
    # end