'''
opens a body and cilia binvox file
converts to both to binary numpy arrays
performs post processing steps on the arrays
returns post-processed binary body and cilia numpy arrays
'''

import numpy as np
import pickle
import matplotlib.pyplot as plt 
from matplotlib.colors import LightSource
from glob import glob

from scipy.ndimage.measurements import label
from scipy.ndimage.morphology import binary_fill_holes, binary_closing, binary_dilation, binary_opening
from scipy.ndimage import zoom
from skimage.morphology import convex_hull_image
from skimage import morphology

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bot_processing import binvox_rw


# def downscale(body,skip):
#     # reduces resolution by uniform sampling
#     return body[::skip,::skip,::skip]

# def smooth_body(body):
#     return binary_fill_holes(body) 

# def get_2d_neighbors(a):
#     b = np.pad(a, pad_width=1, mode='constant', constant_values=0)
#     neigh = np.concatenate((b[2:, 1:-1, None], b[:-2, 1:-1, None],
#         b[1:-1, 2:, None], b[1:-1, :-2, None]), axis=2)
#     return neigh

# def save_image_of_dorsal_view(bot_id, body):
#     print("Plotting...")

#     body = np.fliplr(body)
#     cell_count = np.sum(body>0)
    
#     colors = np.zeros((body.shape[0], body.shape[1], body.shape[2], 3))
#     colors[..., 0][body==2] = 1 #red
#     colors[..., 1][body==2] = 1 #green
#     colors[..., 2][body==1] = 1 #blue
#     colors[..., 2][body==2] = 0

#     ls = LightSource(0, 90)

#     fig = plt.figure(figsize=(7, 8))

#     ax = fig.gca(projection='3d')
#     # ax = fig.add_subplot(8, 7, 1, projection='3d') # for multiply in one iamge

#     ax.set_xlim([0, body.shape[0]])
#     ax.set_ylim([0, body.shape[0]])
#     ax.set_zlim([0, body.shape[0]])

#     ax.view_init(elev=90, azim=0)
#     ax.set_axis_off()

#     ax.voxels(body, facecolors=colors, linewidth=0.4, shade=True, lightsource=ls, alpha=0.9)
#     # fig.subplots_adjust(wspace=-0.05, hspace=-0.05)
#     # bbox = fig.get_window_extent().transformed(fig.dpi_scale_trans.inverted())

#     os.makedirs("images/{}".format(bot_id), exist_ok=True)
#     save_dir = os.path.abspath(__file__ + "/../../") + "/"
#     save_fn = "images/" + bot_id + '/{}_res{}_{}voxels.png'.format(bot_id, body.shape[0], cell_count)
#     path = save_dir+save_fn

#     plt.savefig(path, bbox_inches='tight', dpi=300, transparent=True)
#     # plt.show()

# def scale_test(BOT_ID, BODY_FILENAME, CILIA_FILENAME):
#     SCALES = [4, 8, 10, 12, 14, 16, 18]
#     # SCALES = [5,6,7]  
#     for SCALE in SCALES: 
#         body = load(BOT_ID, BODY_FILENAME, CILIA_FILENAME, SCALE, save_pickle=False)        
#         save_image_of_dorsal_view(BOT_ID, body)

# def fill_holes(body):
#     r_hole = 2

#     for z in range(body.shape[2]):
#         neigh = get_2d_neighbors(body[:,:,z])
#         for x in range(body.shape[0]):
#             for y in range(body.shape[1]):
#                 # Fill from top and bottom
#                 if z!=0 and z!=body.shape[2]-1: 
#                     if body[up(x,y,z)]>0 and body[down(x,y,z)]>0:
#                         body[x,y,z]=1
#                 if z>r_hole-1 and z<body.shape[2]-r_hole: 
#                     if body[up(x,y,z,r_hole)]>0 and body[down(x,y,z,r_hole)]>0:
#                         body[x,y,z]=1

#                 # Fill holes in plane
#                 if np.sum(neigh[x,y,:])==4:
#                     body[x,y,z]=1

#     return body

# def remove(body):
#     neigh = get_neighbors(body)
#     for x in range(body.shape[0]):
#         for y in range(body.shape[1]):
#             for z in range(body.shape[2]):
#                 if np.sum(neigh[x,y,z,:])<3: # fill hole
#                     body[x,y,z]=0
#     return body

# ---------- OLD VERSION ----------
# def move_cilia_to_surface(body):
#     binary_mask = body>0
#     neigh = get_neighbors(binary_mask)

#     new_body = body.copy()
#     search_radius = 5

#     for x in range(body.shape[0]):
#         for y in range(body.shape[1]):
#             for z in range (body.shape[2]):
#                 if body[x,y,z]==2 and np.sum(neigh[x,y,z,:])>=6: # cilia cell not on surface
#                     # check each neighbor to find one on the surface that is not ciliated
#                     for direction in [up, down, left, right, front, back]:
#                         nx,ny,nz = direction(x,y,z)
#                         if body[nx,ny,nz]==1 and np.sum(neigh[nx,ny,nz,:])<6: # replace surface cell with ciliated cell
#                             new_body[nx,ny,nz]=2
#                             new_body[x,y,z]=1
#                         else:
#                             moved = False
#                             for i in range(search_radius): # 5 is the max depth below surface for ciliated cell to be detected and moved to surface
#                                 for direction in [up, down, left, right, front, back]:
#                                     nx,ny,nz = direction(x,y,z,d=i)
#                                     if body[nx,ny,nz]==1 and np.sum(neigh[nx,ny,nz,:])<6: # replace surface cell with ciliated cell
#                                         new_body[nx,ny,nz]=2
#                                         new_body[x,y,z]=1
#                                         moved = True
#                                     break
#                                 if moved:
#                                     break

#                         break # only replace one surface voxel (conserve total number on ciliated cells)

#     return new_body


# ---------- NEW VERSION ----------
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

def find_nearest_edge(body_bin, x, y, z):
    # where body is a binary matrix defining the body of the bot 

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

def remove_small_objects(arr):
    ret = morphology.remove_small_objects(arr, MIN_SIZE)
    return ret

def remove_protruding_voxels(body):
    neigh = get_neighbors(body)
    for x in range(body.shape[0]):
        for y in range(body.shape[1]):
            for z in range(body.shape[2]):
                if np.sum(neigh[x,y,z,:])==1: # remove protrude voxel
                    body[x,y,z]=0
    return body

def orient(arr):
    ret = rotate(arr)
    ret = mirror(ret)
    return ret

def postprocess_body(body, scale):
    # body = zoom(body, 1/scale, order=0)
    # body = shave(body, layers=5)
    body = zoom(body, scale, order=0)
    body = make_one_shape_only(body)
    body = remove_protruding_voxels(body)
    body = orient(body)
    body = fill(body, n=4)
    return body

def postprocess_cilia(cilia, scale):
    cilia = remove_small_objects(cilia)
    # cilia = zoom(cilia, 1/scale, order=0)
    cilia = zoom(cilia, scale, order=0)
    cilia = orient(cilia)
    cilia = fill(cilia, n=3)
    return cilia

def load(bot_name, body_fn, cilia_fn, scale, save_pickle=False):
    '''
    Loads bot from cilia and body .binvox files and performs post-processing 
    Saves list [body, scale] as pickle file where body is 3d body array and 
    scale is the scale it was downscaled at
    '''
    
    DIR = os.path.abspath(__file__ + "/../../") + "/"

    with open(DIR+body_fn, 'rb') as f:
        model = binvox_rw.read_as_3d_array(f)
    body = postprocess_body(model.data, scale)

    original_res = model.data.shape[0] * 2 # *2 because exported at half resolution   

    with open(DIR+cilia_fn, 'rb') as f:
        model = binvox_rw.read_as_3d_array(f) 
    cilia = postprocess_cilia(model.data, scale)

    # Combine the body and cilia arrays into one  
    body = body.astype(int)
    
    body[cilia] = 2 # set the material id for the ciliated cells to 2

    # body = move_cilia_to_surface(body)

    body = shift_down(body)

    # Remove any cilia hanging off
    mask = make_one_shape_only(body>0)
    body[mask==0]=0

    print("Scale:", scale/2)
    # print("Cell count:", np.sum(body>0))

    new_res = body.shape[0]
    
    print("Original resolution:", original_res)
    print("New resolution:", new_res)

    print('Voxel count:', np.sum(body>0))

    if save_pickle:
        os.makedirs("pickle/{}".format(bot_name), exist_ok=True)
        save_dir = os.path.abspath(__file__ + "/../../") + "/"
        save_fn = "pickle/" + bot_name + '/{}_res{}.p'.format(bot_name, body.shape[0])
        path = save_dir+save_fn

        with open(path, 'wb') as f:
            pickle.dump([body,scale], f)
    
        print('Bot saved in {}'.format(save_fn))
        return body, save_fn

    return body

if __name__=="__main__":
    # original image reduced by factor of 1/(SCALE*2)

    global MIN_SIZE

    BOT_ID = "Run4group0subject2"
    BODY_FILENAME = "binvox/"+BOT_ID+"/body_tform_alpha12_pts450000.binvox"
    CILIA_FILENAME = "binvox/"+BOT_ID+"/cilia_tform.binvox"

    MIN_SIZE = 1100

    # To test initial structure after conversion at highest res
    SCALE = 0.25
    body, save_fn = load(BOT_ID, BODY_FILENAME, CILIA_FILENAME, SCALE, save_pickle=True)

    # To view bot in voxcraft-viz
    os.system("rm Run*.vxa")
    os.system("python3 viz_vxa_generator.py {}".format(save_fn))

    # SCALES = [0.065, 0.07, 0.075, 0.08, 0.085, 0.09, 0.095, 0.0975, 0.1, 0.15, 0.2]
    # for SCALE in SCALES:
    #     body, save_fn = load(BOT_ID, BODY_FILENAME, CILIA_FILENAME, SCALE, save_pickle=True)
