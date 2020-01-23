import os
import numpy as np
import pydicom
import vtk
from util.MinimumBoundingBox import MinimumBoundingBox
from skimage.transform import warp

import matplotlib.pyplot as plt

class Preprocess():
    def __init__(self, dcm_path, vtk_path):
        self.plot = True
        self.dcm_path = dcm_path
        self.vtk_path = vtk_path
        
        ds = pydicom.dcmread(self.dcm_path)
        self.I = ds.pixel_array
        spacing = [float(ds.PixelSpacing[0]), float(ds.PixelSpacing[1])]
            
        self.mesh_point, self.mesh_element = self._load_mesh_vtk(self.vtk_path)
        #no need to add 1, index in Python starts from 0
        for i in range(0, 2):
            self.mesh_point[:, i] /= spacing[i]
            
    def _load_mesh_vtk(self, filename):
        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(filename)
        reader.Update()
        polydata = reader.GetOutput()
        pts = polydata.GetPoints()
        n_pts = pts.GetNumberOfPoints()
        mesh_point=np.zeros((n_pts,2))
        for n in range(0, n_pts):
            p = pts.GetPoint(n)
            mesh_point[n,0] = p[0]
            mesh_point[n,1] = p[1]
            n_elm = polydata.GetNumberOfCells()
            mesh_element = []
        for n in range(0, n_elm):
            cell=polydata.GetCell(n)
            mesh_element.append([])
            for m in range(0, cell.GetNumberOfPoints()):
                mesh_element[n].append(cell.GetPointId(m))
        return mesh_point, mesh_element        
    
    def _rescale(self, idx):
        disk_element = self.mesh_element[idx]
        disk_point = self.mesh_point[disk_element]
        disk_landmark = self.mesh_point[[idx*2, idx*2+1, idx*2+2, idx*2+3], :]
        
        rect = MinimumBoundingBox(disk_point)
        center = rect.rectangle_center
        corner_points = np.array(list(rect.corner_points))  # random order
        
        # re-order points
        dispa = corner_points - center
        dispa /= np.sqrt(dispa[:,0]**2 + dispa[:,1]**2).reshape(-1,1)
        dispb = disk_landmark[[0]] - center
        dispb /= np.sqrt(dispb[:,0]**2 + dispb[:,1]**2).reshape(-1,1)
        id0 = np.argmax(np.sum(dispa*dispb, axis=1))
        if id0 == 0:
            pass
        elif id0==1:
            corner_points = corner_points[[1,2,3,0]]
        elif id0==2:
            corner_points = corner_points[[2,3,0,1]]
        elif id0==3:
            corner_points = corner_points[[3,0,1,2]]
        
        #orientation
        direction_x = corner_points[3] - corner_points[0]
        direction_x /= np.sqrt(np.sum(direction_x**2))
        direction_y = corner_points[1] - corner_points[0]
        direction_y /= np.sqrt(np.sum(direction_y**2))
    
        # make a 2:1 square
        new_a = np.sqrt(np.sum((corner_points[0]-corner_points[3])**2))
        new_b = np.sqrt(np.sum((corner_points[0]-corner_points[1])**2))
        new_y = new_a / 5
        new_x = (2. * new_y) + new_b - 0.5 * new_a
        
        m03 = (corner_points[0] + corner_points[3]) / 2 
        m01 = (corner_points[0] + corner_points[1]) / 2
        m23 = (corner_points[2] + corner_points[3]) / 2
        m12 = (corner_points[1] + corner_points[2]) / 2
        
        unitX = corner_points[0] - m03
        unitX /= np.sqrt(np.sum(unitX**2))
        unitY = corner_points[0] - m01
        unitY /= np.sqrt(np.sum(unitY**2))
        corner_points[0] = corner_points[0] + ((new_x)*unitX) + (new_y*unitY)
        
        unitX = corner_points[3] - m03
        unitX /= np.sqrt(np.sum(unitX**2))
        unitY = corner_points[3] - m23
        unitY /= np.sqrt(np.sum(unitY**2))
        corner_points[3] = corner_points[3] + ((new_x)*unitX) + (new_y*unitY)
        
        unitX = corner_points[1] - m12
        unitX /= np.sqrt(np.sum(unitX**2))
        unitY = corner_points[1] - m01
        unitY /= np.sqrt(np.sum(unitY**2))
        corner_points[1] = corner_points[1] + ((new_x)*unitX) + (new_y*unitY)
        
        unitX = corner_points[2] - m12
        unitX /= np.sqrt(np.sum(unitX**2))
        unitY = corner_points[2] - m23
        unitY /= np.sqrt(np.sum(unitY**2))
        corner_points[2] = corner_points[2] + ((new_x)*unitX) + (new_y*unitY)
      
        # rescale to 128*64 pixels
        Nx, Ny = 128, 64
        spacing_x = np.sqrt(np.sum((corner_points[0]-corner_points[3])**2)) / Nx
        spacing_y = np.sqrt(np.sum((corner_points[0]-corner_points[1])**2)) / Ny
        mapping=np.zeros((Ny,Nx), dtype=np.object)
        
        grid_point = []
        for y in range(0, Ny):
            for x in range(0, Nx):
                #p is the grid point position 
                p = corner_points[0] + x*spacing_x*direction_x + y*spacing_y*direction_y
                mapping[y,x] = p
                grid_point.append(p)
        grid_point = np.array(grid_point)        

        # note I[y,x], so grid_point[x,y] => gp[y,x]
        gp = grid_point.copy()
        gp[:,0] = grid_point[:,1]
        gp[:,1] = grid_point[:,0]
        gp = gp.T
        grid_value = warp(self.I, gp)
        Inew = grid_value.reshape(Ny, Nx)
        
        # transform mesh_point from I to Inew
        mesh_point_new = np.zeros(self.mesh_point.shape)
        for n in range(0, self.mesh_point.shape[0]):
            d = self.mesh_point[n] - corner_points[0]
            mesh_point_new[n] = [np.sum(d*direction_x)/spacing_x, np.sum(d*direction_y)/spacing_y]
        
        return Inew, mesh_point_new[self.mesh_element[idx],0], mesh_point_new[self.mesh_element[idx],1]

    def img_plot(self):
        for disk_idx in range(1, len(self.mesh_element), 2):
            Inew, x, y = self._rescale(disk_idx)
            fig, ax = plt.subplots()
            ax.plot()
            ax.imshow(Inew, cmap='gray')
            ax.plot(x, y, 'g-')
    
    def save(self, filename, path):
        for disk_idx in range(1, len(self.mesh_element), 2):
            Inew, x, y = self._rescale(disk_idx)
            points = np.concatenate((x, y), axis=0).reshape(-1, 2, order='F')
            for ext, data, pth in zip(['image', 'point'], [Inew, points], path):
                save_pth = os.path.join(pth, f"{filename}_{disk_idx}_{ext}.npy")
                np.save(save_pth, data)
    