'''
Miscellaneous helper functions
'''
import numpy as np


def normalize(x, xmin, xmax):
    return (x-xmin)/(xmax-xmin)

# Matrix functions
def front(x,y,z,d=1):
    return x,y-d,z 

def back(x,y,z,d=1):
    return x,y+d,z

def left(x,y,z,d=1):
    return x-d,y,z

def right(x,y,z,d=1):
    return x+d,y,z

def up(x,y,z,d=1):
    return x,y,z+d

def down(x,y,z,d=1):
    return x,y,z-d

def get_n_neighbors(self, a, n=1):
    b = np.pad(a, pad_width=n, mode='constant', constant_values=0)
    neigh = np.concatenate((
        b[n*2:, n:-n, n:-n, None], b[:-n*2, n:-n, n:-n, None],
        b[n:-n, n*2:, n:-n, None], b[n:-n, :-n*2, n:-n, None],
        b[n:-n, n:-n, n*2:, None], b[n:-n, n:-n, :-n*2, None]), axis=3)
    return neigh
