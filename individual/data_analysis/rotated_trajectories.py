import numpy as np
from glob import glob
import os
import pickle
import sys
import pandas as pd
import csv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import constants

def shift_trajectory(trajectory):
    # Shift trajectory to start at (0,0)

     # Shift trajectory to start at (0,0)
    x_start = trajectory["x"][0]
    y_start = trajectory["y"][0]

    shiftx = 0-x_start
    shifty = 0-y_start

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

# BOT_ID = 'Run8bot10'
path_to_rotation_csv = 'in_vitro_analysis/degrees_to_rotate_in_silico_bots.csv'
rotation_df = pd.read_csv(path_to_rotation_csv)

for BOT_ID in constants.BOT_ID:
    print(BOT_ID)
    RESULTS_PATH = glob('results/{}/{}*_results_report.p'.format(BOT_ID, BOT_ID)) # path to raw trajectories

    os.makedirs('rotated_trajectories/', exist_ok=True)
    SAVE_PATH = 'rotated_trajectories/' +BOT_ID + '/'
    os.makedirs(SAVE_PATH, exist_ok=True)

    for filename in RESULTS_PATH:

        TX = filename.split('/')[-1].split('_results_report')[-2].split('_')[-1]
        print(TX)

        with open(filename, 'rb') as f:
            trajectories = pickle.load(f)

        print(f"Results for {len(trajectories)} simulations")

        for i,run in enumerate(trajectories):
            trajectory = trajectories[run]

            trajectory = shift_trajectory(trajectory)
            
            row = rotation_df[rotation_df['bot']==BOT_ID]
            rotate_angle_in_degrees = int(row['degrees'].values[0])

            trajectory = rotate_trajectory(trajectory, rotate_angle_in_degrees)

            print(trajectory.head())

            exit()


        # Save out trajectories
        save_fn = filename.split('/')[-1].split('_results_report')[0]+'_trajectories.p'
        
        with open(SAVE_PATH+save_fn, 'wb') as f:
            pickle.dump(trajectories, f)