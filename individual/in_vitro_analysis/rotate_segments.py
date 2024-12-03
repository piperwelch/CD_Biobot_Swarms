import numpy as np
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
import pickle
import os
import csv
from matplotlib import cm
from matplotlib.colors import Normalize
import math


'''
Rotates the 30 second segments based on heading of the bot from timelapse videos
'''

def shift_trajectory(trajectory):
    x_start2 = trajectory["x"][0]
    y_start2 = trajectory["y"][0]

    # shift all trajectories to (0,0)
    shiftx = 0-x_start2
    shifty = 0-y_start2
    
    trajectory['x_shift'] = trajectory['x']+shiftx
    trajectory['y_shift'] = trajectory['y']+shifty

    return trajectory

def rotate_trajectory(trajectory, init_heading):

    # find heading of the trajectory
    x1 = trajectory['x_shift'][0] 
    y1 = trajectory ['y_shift'][0]

    assert x1==0
    assert y1==0

    x2 = trajectory['x_shift'][1]
    y2 = trajectory ['y_shift'][1]

    curr_heading = np.arctan2(y2, x2) 

    # find the angle between the initial heading vector and the current heading vector 
    rotate_angle = init_heading - curr_heading

    # print(init_heading, curr_heading)
    # print(rotate_angle)

    # Rotate the points in the trajectory by the rotate angle
    # https://matthew-brett.github.io/teaching/rotation_2d.html

    # first point in next trajectory... 
    x_prime = np.cos(rotate_angle) * x2 - np.sin(rotate_angle) * y2
    y_prime = np.sin(rotate_angle) * x2 + np.cos(rotate_angle) * y2

    new_x = np.zeros(len(trajectory['x_shift']))
    new_y = np.zeros(len(trajectory['y_shift']))

    new_x[0]=0
    new_y[0]=0
    new_x[1]=x_prime
    new_y[1]=y_prime


    for i in range(2,len(trajectory['x_shift'])): # skip first two points (origin and second point rotated about origin)
        
        # new_x[i] = new_x[i-1] + (xs[i]-xs[i-1])
        # new_y[i] = new_y[i-1] + (ys[i]-ys[i-1])

        new_x[i] = np.cos(rotate_angle) * trajectory['x_shift'][i] - np.sin(rotate_angle)*trajectory['y_shift'][i]
        new_y[i] = np.sin(rotate_angle) * trajectory['x_shift'][i] + np.cos(rotate_angle)*trajectory['y_shift'][i]

    # trajectory['x_rotate'] = np.cos(rotate_angle) * trajectory['x_shift'] - np.sin(rotate_angle) * trajectory['y_shift']
    # trajectory['y_rotate'] = np.sin(rotate_angle) * trajectory['x_shift'] - np.cos(rotate_angle) * trajectory['y_shift']

    trajectory['x_rotate'] = new_x
    trajectory['y_rotate'] = new_y

    return trajectory


CHUNK_LENGTH = 30 # in seconds

bot_folders = glob("in_vitro_analysis/split_data/*{}sec_chunks".format(CHUNK_LENGTH))

os.makedirs('in_vitro_analysis/plots/trajectories/',exist_ok=True)

# Path to save headings in
os.makedirs('heading_data/in_vitro/',exist_ok=True)

# TRAJECTORY_COLOR = (0,0,1,0.6)
hsv_cmap = cm.get_cmap('hsv')

# Establish a new csv for each bot with columns (run, curvature)
heading_csv_save_path = 'heading_data/in_vitro/headings.csv'
f_write = open(heading_csv_save_path, 'w')
writer = csv.writer(f_write)

# Write csv header
writer.writerow(['bot_id','x','y'])

for bot_folder in bot_folders:

    BOT_ID = bot_folder.split('/')[-1].split('_')[0]

    os.makedirs('in_vitro_analysis/plots/trajectories/{}'.format(BOT_ID), exist_ok=True)

    print(BOT_ID)

    fig, ax = plt.subplots(constrained_layout=True)

    num_trajectories = len(glob(bot_folder+'/*.csv'))

    for run in range(num_trajectories):
        bot_csv = bot_folder + '/' + BOT_ID + '_run' + str(run) + '.csv'

        df = pd.read_csv(bot_csv)

        trajectory = shift_trajectory(df)

        # Get the heading of the first trajectory and store it
        if run==0: # First trajectory

            # Heading is the vector between the first and second points of the trajectory
            x1 = trajectory['x_shift'][0] 
            y1 = trajectory ['y_shift'][0]

            x2 = trajectory['x_shift'][1]
            y2 = trajectory ['y_shift'][1]

            # heading vector is an angle in radians in the range [-  pi,pi]
            init_heading = np.arctan2(y2-y1, x2-x1) 

            v = [x2-x1,y2-y1]
            v_unit = v / np.linalg.norm(v)


        else: # rotate all trajectories except the first 
            trajectory = rotate_trajectory(df, init_heading)

        angle_pos_degrees = (init_heading * 180 / math.pi - 90) % 360

        norm = Normalize(vmin=0, vmax=360)
        normalized_heading = norm(angle_pos_degrees)

        if run==0:
            ax.plot(trajectory["x_shift"], trajectory["y_shift"], color=hsv_cmap(normalized_heading), zorder=1)
            # ax.plot(trajectory["x_shift"][0], trajectory["y_shift"][0], color='k', marker='*', zorder=2)
            # ax.quiver(0,0, v_unit[0], v_unit[1], color='fuchsia', zorder=3)

        else:
            # ax.plot(trajectory["x_shift"], trajectory["y_shift"], color='cyan')
            ax.plot(trajectory["x_rotate"], trajectory["y_rotate"], color=hsv_cmap(normalized_heading), zorder=1)
            # ax.quiver(0,0, v_unit[0], v_unit[1], color='fuchsia', zorder=3)
        ax.plot(trajectory["x_shift"][0], trajectory["y_shift"][0], color='k', marker='*', zorder=2)
        plt.axis('equal')    
            # plt.show()
            # exit()
        
        # fig.tight_layout()

    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    row = [BOT_ID, str(v_unit[0]), str(v_unit[1])]
    writer.writerow(row)

    save_path = 'in_vitro_analysis/plots/trajectories/{}/{}_{}sec_chunks_{}_rotated_headingcmap.png'.format(BOT_ID, BOT_ID, CHUNK_LENGTH, run)
    
    plt.title('{} second chunks, rotated'.format(CHUNK_LENGTH))
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

f_write.close()

        
    
