'''
Helper functions to process body arrays
'''

from sys import float_repr_style
import numpy as np

from scipy.ndimage.measurements import label, center_of_mass
from skimage.morphology import remove_small_objects
from scipy import ndimage


def spline_interp(array, scale, order=0): 
    if scale==1:
        return array
    resized = ndimage.zoom(array, scale, order=order)
    return resized

def move_cilia_to_surface(body):

    # Finds direction of nearest edge
    # Searches in that direction to find an unciliated voxel closest to the surface if one exists

    n_ciliated_cells = np.sum(body==2)

    binary_mask = body>0
    neigh = get_neighbors(binary_mask)

    # new_body = body.copy()
    # search_radius = 25 # max distance from ciliated cell to search outward for a surface cell

    for x in range(body.shape[0]):
        for y in range(body.shape[1]):
            for z in range(body.shape[2]):

                if body[x,y,z]==2 and np.sum(neigh[x,y,z,:])==6 and not cilia_part_of_patch(body, x, y, z): # cilia cell not on surface

                    body_mask = body==1 # ignoring cilia as part of surface of the bot (cilia can be layered on the surface)
                    body_neigh = get_neighbors(body_mask)

                    # print(x,y,z)

                    direction = find_nearest_edge(binary_mask, x, y, z)

                    # print(direction)

                    i=1
                    nx,ny,nz = direction(x,y,z,d=i)

                    while nx < body.shape[0] and ny < body.shape[1] and nz < body.shape[2] and body[nx,ny,nz]!=0:
                        i+=1
                        nx,ny,nz = direction(x,y,z,d=i)

                    # work backwards to find a free cell 
                    moved = False

                    while not moved and i >= 1:
                        i-=1
                        nx,ny,nz = direction(x,y,z,d=i)
                        # print(nx,ny,nz)

                        if nx < body.shape[0] and ny < body.shape[1] and nz < body.shape[2]:
                        
                            if body[nx,ny,nz] == 1:

                                body[nx,ny,nz]=2
                                body[x,y,z]=1
                                moved=True

    assert n_ciliated_cells==np.sum(body==2)
    
    return body

def cilia_part_of_patch(body, x, y, z):
    # returns true if cilia is already part of a surface patch of ciliated voxels
    body_bin = body==1

    voxels_up = np.sum(body[x,y,z:]) == np.sum(body_bin[x,y,z:])*2
    voxels_down = np.sum(body[x,y,:z]) == np.sum(body_bin[x,y,:z])*2
    voxels_left = np.sum(body[:x,y,z]) == np.sum(body_bin[:x,y,z])*2
    voxels_right = np.sum(body[x:,y,z]) == np.sum(body_bin[x:,y,z])*2
    voxels_front = np.sum(body[x,:y,z]) == np.sum(body_bin[x,:y,z])*2
    voxels_back = np.sum(body[x,y:,z]) == np.sum(body_bin[x,y:,z])*2

    if voxels_up or voxels_down or voxels_left or voxels_right or voxels_front or voxels_back:
        return True
    else: 
        return False

def fill_cilia(body, n=3):
    neigh = get_neighbors(body==2)
    for x in range(body.shape[0]):
        for y in range(body.shape[1]):
            for z in range(body.shape[2]):
                if np.sum(neigh[x,y,z,:])>=n: # fill hole
                    body[x,y,z]=2
    return body

def find_nearest_edge(body, x, y, z):
    
    body_bin=body>0 # binary matrix defining the body of the bot 

    voxels_up = np.sum(body_bin[x,y,z:])
    voxels_down = np.sum(body_bin[x,y,:z])
    voxels_left = np.sum(body_bin[:x,y,z])
    voxels_right = np.sum(body_bin[x:,y,z])
    voxels_front = np.sum(body_bin[x,:y,z])
    voxels_back = np.sum(body_bin[x,y:,z])
    
    min_distance = np.argmin([voxels_up, voxels_down, voxels_left, voxels_right, voxels_front, voxels_back])

    if min_distance==0:
        # print('up')
        return up
    elif min_distance==1:
        # print('down')
        return down
    elif min_distance==2:
        # print('left')
        return left
    elif min_distance==3:
        # print('right')
        return right
    elif min_distance==4:
        # print('front')
        return front
    elif min_distance==5:
        # print('back')
        return back
    else:
        print('no min direction detected')

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

def fill(body, n=3):
    neigh = get_neighbors(body)
    for x in range(body.shape[0]):
        for y in range(body.shape[1]):
            for z in range(body.shape[2]):
                if np.sum(neigh[x,y,z,:])>=n: # fill hole
                    body[x,y,z]=1
    return body

def get_neighbors(a):
    b = np.pad(a, pad_width=1, mode='constant', constant_values=0)
    neigh = np.concatenate((
        b[2:, 1:-1, 1:-1, None], b[:-2, 1:-1, 1:-1, None],
        b[1:-1, 2:, 1:-1, None], b[1:-1, :-2, 1:-1, None],
        b[1:-1, 1:-1, 2:, None], b[1:-1, 1:-1, :-2, None]), axis=3)
    return neigh

def shave(body, layers=3):
    # shave back layers layers 
    neigh = get_neighbors(body)
    neigh_mat = np.sum(neigh, axis=3)

    for l in range(layers):
        neigh = get_neighbors(body)
        neigh_mat = np.sum(neigh, axis=3)
        body[neigh_mat<6]=0

    return body

def shift_down(body):
    while True:  # shift down until in contact with surface plane
        if np.sum(body[:, :, 0]) == 0:
            body[:, :, :-1] = body[:, :, 1:]
            body[:, :, -1] = np.zeros_like(body[:, :, -1])
        else:
            break
    return body

def make_one_shape_only(output_state):
    """Find the largest continuous arrangement of True elements after applying boolean mask.
    Avoids multiple disconnected softbots in simulation counted as a single individual.
    Parameters
    ----------
    output_state : numpy.ndarray
        Network output
    Returns
    -------
    part_of_ind : bool
        True if component of individual
    """
    if np.sum(output_state) == 0:
        return output_state
    # find coordinates
    array = output_state > 0
    labeled, ncomponents = label(array)
    largest_count = 0
    largest_label = 0
    for n in range(ncomponents+1):
        this_count = np.sum(labeled == n)
        vox_count = np.sum(array[labeled == n])
        if (this_count > largest_count) and (vox_count > 0):
            largest_label = n
            largest_count = this_count
    return labeled == largest_label

def rotate(x):
    ret = np.rot90(x, k=3, axes=(0,1))
    ret = np.rot90(ret, k=2, axes=(0,2))
    return ret

def mirror(x):
    ret = np.fliplr(x)
    return ret

def remove_protruding_voxels(body):
    neigh = get_neighbors(body>0)
    for x in range(body.shape[0]):
        for y in range(body.shape[1]):
            for z in range(body.shape[2]):
                if np.sum(neigh[x,y,z,:])<=1: # remove protrude voxel
                    body[x,y,z]=0
                elif body[x,y,z]==0 and np.sum(neigh[x,y,z,:]) == 6: # fill holes
                    body[x,y,z]=1
    return body

def orient(arr):
    ret = rotate(arr)
    ret = mirror(ret)
    return ret

def remove_noise(arr, min_size):
    # print("Removing cilia noise...")
    return remove_small_objects(arr, min_size)

def hollow(arr):

    mask = arr>0 # ignore cell type

    neigh = get_neighbors(mask)
    n_neigh = np.sum(neigh, axis=3)
    assert np.min(n_neigh)==0
    assert np.max(n_neigh)==6

    # remove all voxels with all 6 neighbors
    arr[n_neigh==6]=0 # hollow 

    return arr

# def shave(arr, layers=1):
#     # Shave layers off the body to expose more cilia
#     inds = np.indices(arr.shape)

#     for i in range(layers):
#         neigh = get_neighbors(arr)

#         neigh_sums = np.sum(neigh,axis=3)

#         # Find indices of surface cells (cells with fewer than 6 neighbors)
#         x_surface_cell_inds = inds[0,:,:,:][neigh_sums<6]
#         y_surface_cell_inds = inds[1,:,:,:][neigh_sums<6]
#         z_surface_cell_inds = inds[2,:,:,:][neigh_sums<6]
#         assert len(x_surface_cell_inds) == len(y_surface_cell_inds) and len(y_surface_cell_inds) == len(z_surface_cell_inds) 
#         dim = len(x_surface_cell_inds)

#         all_surface_inds = np.concatenate((np.reshape(x_surface_cell_inds, (dim,1)),np.reshape(y_surface_cell_inds, (dim,1)),np.reshape(z_surface_cell_inds,(dim,1))), axis=1)

#         # Delete surface voxels
#         arr[all_surface_inds]=0
    
#     return arr

def shave(arr, layers=5):
    # Shave layers off the body to expose more cilia
    for i in range(layers):
        neigh = get_neighbors(arr)
        for x in range(arr.shape[0]):
            for y in range(arr.shape[1]):
                for z in range(arr.shape[2]):
                    # if arr[x,y,z]==1 and np.sum(neigh[x,y,z,:])<6:
                    if np.sum(neigh[x,y,z,:])<6:
                        arr[x,y,z]=0
        
    return arr

def expose_cilia(body):
    # print("Exposing cilia...")
    # For each ciliated voxel i
    #   Find direction of nearest edge
    #   If there is no ciliated voxel between voxel i and the edge 
    #   then replace the surface voxel in that row/column with voxel i

    neigh = get_neighbors(body>0)

    for x in range(body.shape[0]):
        for y in range(body.shape[1]):
            for z in range(body.shape[2]):

                if body[x,y,z]==2 and np.sum(neigh[x,y,z,:])==6: # cilia cell not on surface

                    # print("XYZ:",x,y,z)

                    direction = find_nearest_edge(body, x, y, z)
                    # print(direction)

                    i=1
                    nx,ny,nz = direction(x,y,z,d=i) # index of neighboring cell in the given direction

                    # Find the closest surface cell
                    dont_move = False
                    while nx < body.shape[0] and ny < body.shape[1] and nz < body.shape[2] and body[nx,ny,nz]!=0 and not dont_move:
                        if nx < 0 or ny < 0 or nz < 0: # if the indices move out of bounds don't move the cells
                            dont_move = True
                        i+=1
                        nx,ny,nz = direction(x,y,z,d=i)

                    if not dont_move:
                        i-=1 # step one cell backward to get surface cell
                        nx,ny,nz = direction(x,y,z,d=i)
                        # print(nx,ny,nz)
                        # print(body[nx,ny,nz])
                        # print(cilia[nx,ny,nz])

                        assert np.sum(neigh[nx,ny,nz,:])!=6 # make sure it is a surface cell

                        # make sure it is part of the body but not ciliated
                        # assert body[nx,ny,nz]==1

                        # move ciliated cell
                        body[nx,ny,nz]=2

    # print("done")
    
    return body

def remove_internal_cilia(arr):

    mask = arr>0 # ignore cell type

    neigh = get_neighbors(mask)
    n_neigh = np.sum(neigh, axis=3)
    assert np.min(n_neigh)==0
    assert np.max(n_neigh)==6

    # remove all voxels with all 6 neighbors
    arr[n_neigh==6]=1 

    return arr

def attach_floating_cilia(body):
    labeled, ncomponents = label(body)

    # find the label of the largest connected component
    largest_count = 0
    largest_label = 0
    for n in range(ncomponents+1):
        this_count = np.sum(labeled == n)
        vox_count = np.sum(body[labeled == n])
        if (this_count > largest_count) and (vox_count > 0):
            largest_label = n
            largest_count = this_count

    for l in range(1, ncomponents+1):
        if l != largest_label:
            if np.all(body[labeled == l]==2): # patch of disconnected cilia
                pass

def map_cilia_to_body(body, true_morph):
    # body = body with cilia
    # true_morph = body alone (true shape of the bot)

    # compute COM of true morphology, round to nearest int, and convert to int type
    body_com = [int(x) for x in np.around(center_of_mass(body))]

    cilia_vectors = []
    
    for x in range(body.shape[0]):
        for y in range(body.shape[1]):
            for z in range(body.shape[2]):
                if body[x,y,z]==2: # ciliated cell
                    cilia_vectors.append((x-body_com[0],y-body_com[1],z-body_com[2]))

    radius = int(np.round(np.cbrt(np.sum(true_morph)*3/(4*np.pi))))

    for vec in cilia_vectors:
        # get unit vector and multiply by correct magnitude
        mag = np.sqrt(vec[0]**2+vec[1]**2+vec[2]**2)
        proj = (vec[0]/mag*radius, vec[1]/mag*radius, vec[2]/mag*radius)
        proj = [int(np.round(x)) for x in proj]
        
        x = body_com[0]+proj[0]
        y = body_com[1]+proj[1] 
        z = body_com[2]+proj[2]

        placed = False

        search_dist = 0
        directions = [up,down,front,back,left,right]

        while not placed:
            for direction in directions:
                if not placed:
                    nx,ny,nz=direction(x,y,z,d=search_dist)
                    if nx < true_morph.shape[0] and ny < true_morph.shape[1] and nz < true_morph.shape[2]:
                        if true_morph[nx,ny,nz]==1:
                            true_morph[nx,ny,nz]=2
                            placed = True
            search_dist+=1
    
    return true_morph
