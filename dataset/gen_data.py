import os
import numpy as np
import pandas as pd

from utils.preprocess import Preprocess

PATHS = {
    'index': 'UM100/index.csv',
    'data': 'UM100/src/data',
    'mesh': ['UM100/src/spinemesh@jason', 'UM100/src/spinemesh@liang'],
    'ext': ['jason', 'liang'],
    'dst': ['UM100/dst/image', 'UM100/dst/points']
}


if __name__ == '__main__':
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
            pre.save(f"{patient_id}@{ext}", PATHS['dst'])
        
        
        
    
    
    