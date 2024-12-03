from bot import Bot
from utils.vxd import VXD
import numpy as np
import random
import sys
import os, shutil
import pickle
import matplotlib.pyplot as plt

# Get inputs
RUN = int(sys.argv[1])
BOT_ID = sys.argv[2]
RECORD_VOXELS = True if sys.argv[3]=="1" else False

# Set random seed
np.random.seed(RUN)
random.seed(RUN)

# Experimental parameters
RANDOMIZE_CILIA_DISTRIBUTION = True if sys.argv[4]=="1" else False  # control 1
SPHERE = True if sys.argv[5] == "1" else False # control 2
RESTRICT = True if sys.argv[6] == '1' else False # treatment

TAG = sys.argv[7]

BOT_TYPE = sys.argv[8]

# Read bot from pickle file
folder = BOT_ID.split('_')[0]
bot = Bot('pickle/'+BOT_TYPE+'/'+ folder + '/' + BOT_ID +'.p', make_sphere=SPHERE)
# Set cilia forces
bot.set_cilia_forces(randomize_distribution=RANDOMIZE_CILIA_DISTRIBUTION, restricted=RESTRICT)

# Save out cilia to pickle file 
os.makedirs("cilia/{}".format(BOT_ID), exist_ok=True)

RUN_ID="{}_{}_run{}".format(BOT_ID, TAG, RUN)

with open("cilia/"+BOT_ID+"/"+RUN_ID+"_cilia.p", 'wb') as f:
    pickle.dump(bot.cilia,f)

# Write bot to a vxd file and save to data folder
vxd = VXD()
vxd.set_vxd_tags(bot.body, bot.cilia, record_history=True, record_voxels=RECORD_VOXELS)

os.makedirs("data/{}".format(RUN_ID), exist_ok=True)
shutil.copyfile("data/base.vxa", "data/{}/base.vxa".format(RUN_ID))
vxd.write("{}/body_{}.vxd".format(RUN_ID, RUN_ID))
