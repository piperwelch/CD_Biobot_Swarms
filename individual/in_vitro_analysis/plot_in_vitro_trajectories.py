import numpy as np
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
import pickle
import os
from matplotlib import cm
from matplotlib.colors import Normalize
import math

def shift_trajectory(trajectory):
    x_start2 = trajectory["x"][0]
    y_start2 = trajectory["y"][0]

    # shift all trajectories to (0,0)
    shiftx = 0-x_start2
    shifty = 0-y_start2
    
    trajectory['x_shift'] = trajectory['x']+shiftx
    trajectory['y_shift'] = trajectory['y']+shifty

    return trajectory

CHUNK_LENGTH = 30 # in seconds

bot_folders = glob("in_vitro_analysis/split_data/*{}sec_chunks".format(CHUNK_LENGTH))

os.makedirs('in_vitro_analysis/plots/trajectories/',exist_ok=True)

# TRAJECTORY_COLOR = (0,0,1,0.6)
hsv_cmap = cm.get_cmap('hsv')

for bot_folder in bot_folders:

    BOT_ID = bot_folder.split('/')[-1].split('_')[0]

    os.makedirs('in_vitro_analysis/plots/trajectories/{}'.format(BOT_ID), exist_ok=True)

    print(BOT_ID)

    fig, ax = plt.subplots(constrained_layout=True)

    for i,bot_csv in enumerate(glob(bot_folder+'/*.csv')):

        df = pd.read_csv(bot_csv)

        trajectory = shift_trajectory(df)

        # compute heading
        # Heading is the vector between the first and second points of the trajectory
        x1 = trajectory['x_shift'][0] 
        y1 = trajectory ['y_shift'][0]

        x2 = trajectory['x_shift'][1]
        y2 = trajectory ['y_shift'][1]

        # heading vector is an angle in radians in the range [-  pi,pi]
        init_heading = [y2-y1, x2-x1]
        angle_pos_degrees = (math.atan2(init_heading[0], init_heading[1]) * 180 / math.pi - 90) % 360

        norm = Normalize(vmin=0, vmax=360)
        normalized_heading = norm(angle_pos_degrees)

        ax.plot(trajectory["x_shift"][0], trajectory["y_shift"][0], '*r')
        ax.plot(trajectory["x_shift"], trajectory["y_shift"], color=hsv_cmap(normalized_heading))
        
        # fig.tight_layout()
        plt.axis('equal')


    save_path = 'in_vitro_analysis/plots/trajectories/{}/{}_{}sec_chunks_{}_headingcmap.png'.format(BOT_ID, BOT_ID, CHUNK_LENGTH, i)
    
    # plt.title('{} second chunks'.format(CHUNK_LENGTH))
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
    # plt.savefig('../../../Desktop/in_vitro.png', bbox_inches='tight', dpi=500)
    plt.close()
    # plt.show()
    # exit()


    # # Read in CSV and clean
    # df_raw = pd.read_csv(infile)

    # if 'subject' in BOT_ID:
    #     ID_NBR = BOT_ID.split('subject')[-1]
    # else:
    #     ID_NBR = None

    # if "ignore" in df_raw.columns:
    #     df = clean(df_raw, ID_NBR)
    # else:
    #     df = df_raw

    # # Split into 30 sec chunks
    # # Frame rate = 40 fps
    # trajectories = split(df, frames_per_chunk)

    # # fig, ax = plt.subplots(1,1, figsize=(7,7))

    # for i,trajectory in enumerate(trajectories):

    #     # trajectory = trajectories[run]

    #     fig, ax = plt.subplots(1,1, figsize=(7,7))

    #     ax.plot(trajectory["x"][0], trajectory["y"][0], '*r')
    #     ax.plot(trajectory["x"], trajectory["y"], color=TRAJECTORY_COLOR)
        
    #     fig.tight_layout()
    #     plt.axis('equal')


    #     save_path = 'in_vitro_analysis/plots/trajectories_{}secsplt/{}/{}_{}.png'.format(splt,BOT_ID,BOT_ID, i)
        
    #     plt.savefig(save_path)
    #     plt.close()
    #     # plt.show()


        
    
