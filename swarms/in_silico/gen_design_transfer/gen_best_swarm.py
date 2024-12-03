import numpy as np
from vxa import VXA
from vxd import VXD
from bot import Bot
from datetime import datetime
import pickle 
from glob import glob 
import random 
import os 
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import math

def remove_empty_slices(cilia, body):
    '''Hard-coded to remove the empty slices from the body arrays which messes with the cilia rotation'''
    new_body = np.zeros(shape=(7,7,7))
    new_cilia = np.zeros(shape=(7,7,7,3))

    new_body[:,:,:] = body[1:,1:,1:]
    new_cilia[:,:,:,:] = cilia[1:,1:,1:,:]

    return new_cilia,new_body

def rotate_cilia(cilia, k):
    for i in range(0,k): 
        cilia = rotate_90(cilia)
    return cilia

def rotate_90(cilia):
    
    rotated_cilia = np.zeros(shape=cilia.shape)

    temp = np.rot90(cilia, axes=(1,0))

    rotated_cilia[:,:,:,0] = temp[:,:,:,1] * -1
    rotated_cilia[:,:,:,1] = temp[:,:,:,0] 

    return rotated_cilia

def flip(cilia):

    flipped = np.fliplr(cilia)

    flipped[:,:,:,0] = flipped[:,:,:,0] * -1 # rotate force vectors about the y axis

    return flipped
    
vxa = VXA(SimTime = 5, EnableCilia=1, EnableCollision=0, RecordVoxel=1)

mat1 = vxa.add_material(RGBA=(255, 0, 0, 255), E=9e4, Cilia=0.0006, LockZ = 1) # red
mat2 = vxa.add_material(RGBA=(0, 255, 0, 255), E=9e4, Cilia=0.0006, LockZ = 1) # green
mat3 = vxa.add_material(RGBA=(0, 0, 255, 255), E=9e4, Cilia=0.0006, LockZ = 1) # blue
mat4 = vxa.add_material(RGBA=(255, 0, 255, 255), E=9e4, Cilia=0.0006, LockZ = 1) # purple
mat5 = vxa.add_material(RGBA=(233, 233, 233, 255), E=9e4, Cilia=0.0006, LockZ = 1) # passive
random.seed(0)

color_bot_maps = {"red":1, "green":2, "blue":3, "purple":4}
swarm_6 = [
"pickles/bot_2022_10_07_09_21_08_PM.p","red",
"pickles/bot_2022_10_07_09_19_50_PM.p","green",
"pickles/bot_2022_10_07_08_18_13_PM.p","blue",
"pickles/bot_2022_10_07_09_16_39_PM.p","purple",
]

swarm_8 = [
"pickles/bot_2022_10_07_08_18_13_PM.p","red",
"pickles/bot_2022_10_07_09_19_50_PM.p","green",
"pickles/bot_2022_10_07_09_21_08_PM.p","blue",
"pickles/bot_2022_10_07_09_16_39_PM.p","purple",
]

swarm_2 = [
"pickles/bot_2022_10_07_08_18_13_PM.p","red",
"pickles/bot_2022_10_07_09_21_08_PM.p","green",
"pickles/bot_2022_10_07_09_19_50_PM.p","blue",
"pickles/bot_2022_10_07_09_16_39_PM.p","purple"
]

swarm_1 = [
"pickles/bot_2022_10_07_09_19_50_PM.p","red",
"pickles/bot_2022_10_07_08_18_13_PM.p","green",
"pickles/bot_2022_10_07_09_21_08_PM.p","blue",
"pickles/bot_2022_10_07_09_16_39_PM.p","purple"
]

#randoms
swarm_3 = ["pickles/bot_2022_10_07_09_16_53_PM.p","red",
"pickles/bot_2022_10_07_09_16_57_PM.p","green",
"pickles/bot_2022_10_07_09_23_09_PM.p","blue",
"pickles/bot_2022_10_07_09_18_59_PM.p","purple"
]

swarm_4 = ["pickles/bot_2022_10_07_09_22_26_PM.p","red",
"pickles/bot_2022_10_07_09_22_01_PM.p","green",
"pickles/bot_2022_10_07_08_18_26_PM.p","blue",
"pickles/bot_2022_10_07_09_22_39_PM.p","purple"
]

swarm_5 = ["pickles/bot_2022_10_07_09_18_55_PM.p","red",
"pickles/bot_2022_10_07_08_19_24_PM.p","green",
"pickles/bot_2022_10_07_09_22_34_PM.p","blue",
"pickles/bot_2022_10_07_09_22_59_PM.p","purple"
]

#randoms seed 1 from test_gen_random.ipynb generated 3/20 using seed 0 

swarm_7 = ['pickles/bot_2022_10_07_09_18_05_PM.p',"red",
            'pickles/bot_2022_10_07_09_20_52_PM.p',"green",
            'pickles/bot_2022_10_07_08_18_20_PM.p', "blue",
            'pickles/bot_2022_10_07_09_19_30_PM.p', "purple"
]
count = 0
count_2 = 0
start_pts =  [[100, 40, 0], [40, 100, 0],  [40, 40, 0],  [100, 100, 0]]
for swarm in [swarm_1, swarm_2, swarm_3, swarm_4, swarm_5, swarm_6, swarm_7, swarm_8]:
  
  count_2 = 0
  world = np.zeros((200, 200, 20), dtype = int)
  ciliaWorld = np.zeros((world.shape[0], world.shape[1], world.shape[2], 3))
  pat = "rand_swarm_1_18/swarm_{}".format(count)

  os.makedirs(pat, exist_ok=True)
  vxa.write(pat + "/base.vxa")
  for i in range(world.shape[0]):
    for j in range(world.shape[1]):
      if i == 0 and j <= 160:
        world[i,j,0] = 5
      if i == 160 and j <= 160:
        world[i, j, 0] = 5
      if j == 160 and i <= 160:
        world[i, j, 0] = 5
      if j == 0 and i <=160:
        world[i, j, 0] = 5
  count+=1
  for bot in swarm:
    if bot == "red" or bot =="green" or bot == "blue" or bot == "purple":
        continue
    

    point = start_pts[count_2 ]
    count_2+=1 
    with open(bot, 'rb') as f:
        cilia, body = pickle.load(f)
    cilia, body = remove_empty_slices(cilia, body)
    k =  random.randint(0, 3)
    cilia = rotate_cilia(cilia, k)

    print(count_2)
    body[body==1] = count_2
    x, y, z = point[0], point[1], point[2]

    world[x:x+body.shape[0], y:y+body.shape[1], z:z+body.shape[2]] = body
    
    ciliaWorld[x:x+body.shape[0], y:y+body.shape[1], z:z+body.shape[2],0] = cilia[:,:,:,0]
    ciliaWorld[x:x+body.shape[0], y:y+body.shape[1], z:z+body.shape[2],1] = cilia[:,:,:,1]
    ciliaWorld[:,:,:,2] = 0 # no cilia forces in the z direction


  vxd = VXD()
  vxd.set_vxd_tags(body = world, cilia = ciliaWorld, record_voxels=1, record_history=1, RecordCoMTraceOfEachVoxelGroupfOfThisMaterial=1)
  vxd.set_data(world, ciliaWorld)
  vxd.write(pat +"/swarm{}.vxd".format(count))

