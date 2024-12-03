import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import os
import re
import pylab
import pickle
import sys
import pandas as pd
from matplotlib import cm
from matplotlib.colors import Normalize
import math

def shift_trajectory(trajectory, shiftx, shifty):
    # shifted_df = pd.DataFrame(columns=['x','y'])

    # for i,row in trajectory.iterrows():
    #     newx = row['x'] + shiftx
    #     newy = row['y'] + shifty
    #     shifted_df = shifted_df.append({'x':newx, 'y':newy}, ignore_index=True)

    trajectory['x_shift'] = trajectory['x']+shiftx
    trajectory['y_shift'] = trajectory['y']+shifty

    return trajectory

def rotate_trajectory(trajectory, rotate_angle_degrees):

    # angle in clockwise rotation (subtract from 360 for anti-clockwise rotation)
    rotate_angle_degrees = 360 - rotate_angle_degrees

    # convert degrees to radians
    rotate_angle_rad = (np.pi/180)*rotate_angle_degrees

    # Rotate the points in the trajectory by the rotate angle anti-clockwise
    # https://matthew-brett.github.io/teaching/rotation_2d.html

    new_x = np.zeros(len(trajectory['x_shift']))
    new_y = np.zeros(len(trajectory['y_shift']))

    new_x[0]=0
    new_y[0]=0

    for i in range(1,len(trajectory['x_shift'])): # skip first point

        new_x[i] = np.cos(rotate_angle_rad) * trajectory['x_shift'][i] - np.sin(rotate_angle_rad)*trajectory['y_shift'][i]
        new_y[i] = np.sin(rotate_angle_rad) * trajectory['x_shift'][i] + np.cos(rotate_angle_rad)*trajectory['y_shift'][i]

    trajectory['x_rotate'] = new_x
    trajectory['y_rotate'] = new_y

    return trajectory

################# NORMAL RUNS #################
BOT_ID = 'Run5group7subject1'
zoomed_out = False #False
ROTATE = True # rotate trajectories to match initial oriention of in vitro bot
WITH_AVG_VECTOR = False #True
path_to_rotation_csv = 'in_vitro_analysis/degrees_to_rotate_in_silico_bots.csv'
rotation_df = pd.read_csv(path_to_rotation_csv)

# BOT_IDs = ['Run4group0subject0', 'Run4group0subject2', 'Run4group0subject3', 'Run4group0subject4', 'Run4group0subject5',
#             'Run4group3subject1', 'Run4group3subject2', 'Run4group5subject3', 'Run5group7subject1', 'Run6group5subject1', \
#             'Run6group5subject2', 'Run6group5subject3', 'Run6group5subject4', 'Run6group5subject5', 'Run8bot1', 'Run8bot2', \
#             'Run8bot4', 'Run8bot5', 'Run8bot6', 'Run8bot7', 'Run8bot9',  'Run8bot10', 'Run8bot12', 'Run8bot13']

# for BOT_ID in BOT_IDs:
RESULTS_PATH = glob('results/{}/{}*_results_report.p'.format(BOT_ID, BOT_ID))

SAVE_PATH = '../../../Desktop'

# os.makedirs('plots/{}'.format(BOT_ID), exist_ok=True)
# if zoomed_out:
#     SAVE_PATH = 'plots/{}/trajectories_zoomed_out/'.format(BOT_ID)
# elif ROTATE and not WITH_AVG_VECTOR:
#     SAVE_PATH = 'plots/{}/trajectories_rotated/'.format(BOT_ID)
# elif ROTATE and WITH_AVG_VECTOR:
#     SAVE_PATH = 'plots/{}/trajectories_rotated_with_vectors/'.format(BOT_ID)
#     avg_heading_df = pd.read_csv('heading_data/in_silico/{}.csv'.format(BOT_ID))
# else:
#     SAVE_PATH = 'plots/{}/trajectories/'.format(BOT_ID)
# os.makedirs(SAVE_PATH, exist_ok=True)
###################################################

# # ################# TESTING #################
# BOT_ID = 'test_sphere_right'
# RESULTS_PATH = glob('testing/test_ctrl/spheres/results_fix/{}*_results_report.p'.format(BOT_ID, BOT_ID))

# os.makedirs('testing/test_ctrl/spheres/plots_fix/', exist_ok=True)
# SAVE_PATH = 'testing/test_ctrl/spheres/plots_fix/trajectories/'
# os.makedirs(SAVE_PATH, exist_ok=True)
# # ###################################################

for filename in RESULTS_PATH:
    TX = filename.split('/')[-1].split('_results_report')[-2]
    print(TX)

    with open(filename, 'rb') as f:
        trajectories = pickle.load(f)

    print(f"Results for {len(trajectories)} simulations")

    fig, ax = plt.subplots(1,1, figsize=(7,7))

    # color = (0,0,1,0.6) # blue 
    hsv_cmap = cm.get_cmap('hsv')
    # NUM_COLORS = len(trajectories)
    # cm = pylab.get_cmap('hsv')
    # colors = []

    # for i in range(NUM_COLORS):
    #     color = cm(1.*i/NUM_COLORS)  # color will now be an RGBA tuple
    #     colors.append((color[0],color[1],color[2], 0.6))

    for i,run in enumerate(trajectories):
        trajectory = trajectories[run]

        # Shift trajectory to start at (0,0)
        x_start = trajectory["x"][0]
        y_start = trajectory["y"][0]

        shiftx = 0-x_start
        shifty = 0-y_start

        trajectory = shift_trajectory(trajectory, shiftx,shifty)
        # ax.plot(trajectory["x"], trajectory["y"], color=colors[i])

        if ROTATE:
            row = rotation_df[rotation_df['bot']==BOT_ID]
            rotate_angle_in_degrees = int(row['degrees'].values[0])

            trajectory = rotate_trajectory(trajectory, rotate_angle_in_degrees)
            
            # paint trajectory based on heading
            heading_vec = [trajectory["x_rotate"][11] - trajectory["x_rotate"][0], trajectory["y_rotate"][11] - trajectory["y_rotate"][0]]
            heading_vec_unit = heading_vec / np.linalg.norm(heading_vec) # make a unit vector specifying the initial direction of movement
            angle_pos_degrees = (math.atan2(heading_vec_unit[0], heading_vec_unit[1]) * 180 / math.pi - 90) % 360

            norm = Normalize(vmin=0, vmax=360)
            normalized_heading = norm(angle_pos_degrees)

            ax.plot(trajectory["x_rotate"], trajectory["y_rotate"], color=hsv_cmap(normalized_heading),zorder=1)
            ax.plot(trajectory["x_rotate"][0], trajectory["y_rotate"][0], 'k', marker='*', zorder=2)

            # if WITH_AVG_VECTOR:
            #     row = avg_heading_df[avg_heading_df['treatment']==TX]
            #     x = float(row['x'].values[0]) 
            #     y = float(row['y'].values[0]) 
            #     ax.quiver(0,0, x, y, color='fuchsia', zorder=3)
        else:

            # paint trajectory based on heading
            heading_vec = [trajectory["x_shift"][11] - trajectory["x_shift"][0], trajectory["y_shift"][11] - trajectory["y_shift"][0]]
            heading_vec_unit = heading_vec / np.linalg.norm(heading_vec) # make a unit vector specifying the initial direction of movement
            angle_pos_degrees = (math.atan2(heading_vec_unit[0], heading_vec_unit[1]) * 180 / math.pi - 90) % 360

            norm = Normalize(vmin=0, vmax=360)
            normalized_heading = norm(angle_pos_degrees)

            ax.plot(trajectory["x_shift"], trajectory["y_shift"], color=hsv_cmap(normalized_heading))
            ax.plot(trajectory["x_shift"][0], trajectory["y_shift"][0], '*o')

    
    # plt.xlim([-0.5,0.8])

    fig.tight_layout()

    if zoomed_out:
        ax.set_xlim([-0.75, 0.25])
        ax.set_ylim([-0.4, 0.6])
        # ax.set_xlim([-0.5, 0.4])
        # ax.set_ylim([-0.8, 0.2])
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
    else:
        plt.axis('equal')

        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)

    # SAVE_PATH = "plots/{}/trajectories/".format(BOT_ID)
        
    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=hsv_cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm)
    cbar.set_label('Initial Heading (degrees)', fontsize=25) 
    cbar.ax.tick_params(labelsize=20)  # Adjust the fontsize here
        
    if zoomed_out:
        plt.savefig("{}/{}_trajectories.png".format(SAVE_PATH, TX), dpi=500, bbox_inches='tight')
    else:
        plt.savefig("{}/{}_trajectories_zoomed.png".format(SAVE_PATH, TX), dpi=500, bbox_inches='tight')
    plt.close()
    # plt.savefig("{}/trajectories.png".format(SAVE_PATH), dpi=500)
    # plt.show()
    exit()
    