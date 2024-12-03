import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from stl import mesh
from mpl_toolkits import mplot3d

import os
import pickle
import matplotlib.pyplot as plt


PATH = os.path.abspath(__file__ + "/../../") + "/pickle/Run4group0subject2/Run4group0subject2_scale32.p"

with open(PATH, 'rb') as f:
    body = pickle.load(f)[0]

# for l in range(body.shape[2]):
#     layer = body[:,:,l]

#     plt.matshow(layer)
#     plt.title(l)
#     plt.show()

def front(x,y,z,d=1):
    return x,y-d,z 

def back(x,y,z,d=1):
    return x,y+d,z

def left(x,y,z,d=1):
    return x-1,y,z

def right(x,y,z,d=1):
    return x+1,y,z

def up(x,y,z,d=1):
    return x,y,z+d

def down(x,y,z,d=1):
    return x,y,z-d

def get_2d_neighbors(a):
    b = np.pad(a, pad_width=1, mode='constant', constant_values=0)
    neigh = np.concatenate((b[2:, 1:-1, None], b[:-2, 1:-1, None],
        b[1:-1, 2:, None], b[1:-1, :-2, None]), axis=2)
    return neigh

r_hole = 2

for z in range(body.shape[2]):
    neigh = get_2d_neighbors(body[:,:,z])
    for x in range(body.shape[0]):
        for y in range(body.shape[1]):
            # Fill from top and bottom
            if z!=0 and z!=body.shape[2]-1: 
                if body[up(x,y,z)]>0 and body[down(x,y,z)]>0:
                    body[x,y,z]=1
            if z>r_hole-1 and z<body.shape[2]-r_hole: 
                if body[up(x,y,z,r_hole)]>0 and body[down(x,y,z,r_hole)]>0:
                    body[x,y,z]=1

            # Fill holes in plane
            if np.sum(neigh[x,y,:])==4:
                body[x,y,z]=1

for l in range(body.shape[2]):
    layer = body[:,:,l]

    plt.matshow(layer)
    plt.title(l)
    plt.show()