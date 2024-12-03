import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import os
import pickle
import sys
import pandas as pd
import csv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import constants

def compute_angle(v1,v2):
    # Returns angle between two vectors ranging from 0-180
    # v2 is always the in vitro heading

    angle_v1 = np.arctan2(v1[1], v1[0]) *180 /np.pi # arctan(y,x) gives angle from positive x-axis in radians
    angle_v2 = np.arctan2(v2[1], v2[0]) *180 /np.pi

    # print("V1:", v1, angle_v1)
    # print("V2:", v2, angle_v2)
    
    theta_temp = np.abs(angle_v2 - angle_v1) 
    theta = np.min([theta_temp, 360-theta_temp]) # find the smaller angle around the unit circle
    # print("ERROR:", theta)
    # print()
    
    return theta

def normalize_angle(theta):
    # min = 0, max = 180
    return theta/180

############### Extract in vitro heading data ###############
path_to_in_vitro_headings = 'heading_data/in_vitro/headings.csv'

in_vitro_headings_df = pd.read_csv(path_to_in_vitro_headings)
in_vitro_headings = {}

for row in in_vitro_headings_df.iterrows():
    bot_id = row[1]['bot_id']
    x = float(row[1]['x'])
    y = float(row[1]['y'])

    in_vitro_headings[bot_id] = (x,y)

################# In Silico Data #################
# BOT_ID = 'Run4group0subject4'

for BOT_ID in constants.BOT_IDs:

    # Path to trajectories
    RESULTS_PATH = glob('rotated_trajectories/{}/{}*_trajectories.p'.format(BOT_ID, BOT_ID))

    # Path to save csvs in 
    SAVE_PATH = 'heading_errors/'
    os.makedirs(SAVE_PATH, exist_ok=True)

    # Establish a new csv for each bot with columns (tx, run, error, normalized_error)
    csv_save_path = SAVE_PATH + '{}.csv'.format(BOT_ID)
    f_write = open(csv_save_path, 'w')
    writer = csv.writer(f_write)

    # Write csv header
    writer.writerow(['tx', 'run', 'heading_x', 'heading_y', 'error','normalized_error'])

    print(BOT_ID)
    for filename in RESULTS_PATH: # iterating through each treatment

        TX = filename.split('/')[-1].split('_trajectories')[-2].split('_')[-1]
        print(TX)

        with open(filename, 'rb') as f:
            trajectories = pickle.load(f)

        print(f"Results for {len(trajectories)} simulations")

        for i,run in enumerate(trajectories): # iterating through each run for the given treatment
            trajectory = trajectories[run]

            run_nbr = run.split('.')[0].split('run')[-1]

            # Compute heading vector - use first and 11th point to get good estimate of heading (this is used when computing rotational direction)
            # (sampling rate is too high, first two points are too close together)
            
            heading_vec = [trajectory["x_rotate"][11] - trajectory["x_rotate"][0], trajectory["y_rotate"][11] - trajectory["y_rotate"][0]]

            if np.linalg.norm(heading_vec)==0: # bot doesn't move for at least the first 11 time steps 

                # plt.plot(trajectory['x_rotate'], trajectory['y_rotate'])
                # plt.show()

                error = 'NaN'
                error_norm = 'NaN'

            else:
                heading_vec_unit = heading_vec / np.linalg.norm(heading_vec) # make a unit vector specifying the initial direction of movement

                # Compute error between in silico and in vitro headings

                assert np.isclose(np.linalg.norm(in_vitro_headings[BOT_ID]),1.0)

                error = compute_angle(heading_vec_unit, in_vitro_headings[BOT_ID])

                # Compute the normalized error
                assert error<=180
                assert error>=0

                error_norm = normalize_angle(error)

                assert error_norm<=1
                assert error_norm>=0
        
            # # Write line to csv 
            row = [TX, run_nbr, heading_vec_unit[0], heading_vec_unit[1], error, error_norm]
            writer.writerow(row)

    # Save out csv 
    f_write.close()
