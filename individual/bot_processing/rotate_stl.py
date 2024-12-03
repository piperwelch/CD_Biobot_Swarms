import numpy as np
import os
import trimesh
import sys

def load_transform_matrix(path):
    tform = np.zeros((4,4))
    with open(path, 'r') as f:
        # Read contents of file
        lines = f.readlines()
    ln_ct=0
    for line in lines:
        if line[0]!='#':
            nbrs = line.strip().split(' ')
            for i in range(len(nbrs)):
                tform[ln_ct,i] = nbrs[i]
            ln_ct+=1
    return tform

if __name__=='__main__':

    BOT_ID = sys.argv[1]

    # Load transform matrix
    TFORM_PATH = os.path.abspath(__file__ + "/../../") + "/binvox/" + BOT_ID + "/tform.mat"
    tform = load_transform_matrix(TFORM_PATH)

    # Load and rotate body and cilia stls
    STL_DIR = os.path.abspath(__file__ + "/../../") + "/binvox/" + BOT_ID
    
    body_mesh = trimesh.load(STL_DIR+'/body.stl')        
    body_mesh.apply_transform(tform)
    trimesh.repair.fill_holes(body_mesh)    
    body_mesh.export(f"{STL_DIR}/body_tform.stl", "stl")

    cilia_mesh = trimesh.load(STL_DIR+'/cilia.stl')        
    cilia_mesh.apply_transform(tform)
    cilia_mesh.export(f"{STL_DIR}/cilia_tform.stl", "stl")