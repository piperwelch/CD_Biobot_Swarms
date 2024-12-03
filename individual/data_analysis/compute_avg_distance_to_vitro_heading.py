import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import os
import re
import pylab
import pickle
import sys
import pandas as pd
import csv

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

def compute_angle(v1,v2):
    # v2 should always be the in vitro heading
    angle_v1 = np.arctan2(v1[1], v1[0]) *180 /np.pi # arctan(y,x) gives angle from positive x-axis in radians
    # if angle_v1<0: 
    #     angle_v1 = 360+angle_v1 # convert to positive angle 
    angle_v2 = np.arctan2(v2[1], v2[0]) *180 /np.pi
    # if angle_v2<0: 
    #     angle_v2 = 360+angle_v2 # convert to positive angle 

    # https://www.omnicalculator.com/math/angle-between-two-vectors#angle-between-two-vectors-formulas
    angle = np.arccos((v2[0] * v1[0] + v2[1] * v1[1]) / (np.sqrt(v2[0]**2 + v2[1]**2) * np.sqrt(v1[0]**2 + v1[1]**2)))
    angle_deg = angle*180 /np.pi

    # print("ANGLES:",angle_v1,angle_v2)
    # print("DIFFERENCE:",np.abs(angle_v2 - angle_v1))

    # print("DIFFERENCE ANGLE:", angle_deg)
    # exit()

    # return np.abs(angle_v2 - angle_v1)
    return angle_deg


############### Extract in vitro heading data ###############
path_to_in_vitro_headings = 'heading_data/in_vitro/headings.csv'

in_vitro_headings_df = pd.read_csv(path_to_in_vitro_headings)
in_vitro_headings = {}

for row in in_vitro_headings_df.iterrows():
    bot_id = row[1]['bot_id']
    x = float(row[1]['x'])
    y = float(row[1]['y'])

    in_vitro_headings[bot_id] = (x,y)

################# NORMAL RUNS #################
# BOT_ID = 'Run6group5subject3'
path_to_rotation_csv = 'in_vitro_analysis/degrees_to_rotate_in_silico_bots.csv'
rotation_df = pd.read_csv(path_to_rotation_csv)

BOT_IDs = ['Run4group0subject0', 'Run4group0subject2', 'Run4group0subject3', 'Run4group0subject4', 'Run4group0subject5',
            'Run4group3subject1', 'Run4group3subject2', 'Run4group5subject3', 'Run5group7subject1', 'Run6group5subject1', \
            'Run6group5subject2', 'Run6group5subject3', 'Run6group5subject4', 'Run6group5subject5', 'Run8bot1', 'Run8bot2', \
            'Run8bot4', 'Run8bot5', 'Run8bot6', 'Run8bot7', 'Run8bot9', 'Run8bot10', 'Run8bot12', 'Run8bot13']

for BOT_ID in BOT_IDs:
    RESULTS_PATH = glob('results/{}/{}*_results_report.p'.format(BOT_ID, BOT_ID))

    # os.makedirs('heading_errors/', exist_ok=True)

    SAVE_PATH = 'heading_errors'
    os.makedirs(SAVE_PATH, exist_ok=True)

    # Establish a new csv for each bot with columns (run, curvature)
    heading_csv_save_path = 'heading_errors/{}.csv'.format(BOT_ID)
    f_write = open(heading_csv_save_path, 'w')
    writer = csv.writer(f_write)

    # Write csv header
    writer.writerow(['treatment','error_angle','std'])

    for filename in RESULTS_PATH:
        TX = filename.split('/')[-1].split('_results_report')[-2]
        print(TX)

        with open(filename, 'rb') as f:
            trajectories = pickle.load(f)

        print(f"Results for {len(trajectories)} simulations")

        err_in_headings = []

        for i,run in enumerate(trajectories):
            trajectory = trajectories[run]

            # Shift trajectory to start at (0,0)
            x_start = trajectory["x"][0]
            y_start = trajectory["y"][0]

            shiftx = 0-x_start
            shifty = 0-y_start

            trajectory = shift_trajectory(trajectory, shiftx,shifty)
            
            row = rotation_df[rotation_df['bot']==BOT_ID]
            rotate_angle_in_degrees = int(row['degrees'].values[0])

            trajectory = rotate_trajectory(trajectory, rotate_angle_in_degrees)

            # compute heading vector and add to a list
            # use first and 11th point (this is used what computing rotational direction)
            # sampling rate is too high, first two points are too close together
            v = [trajectory["x_rotate"][11] - trajectory["x_rotate"][0], trajectory["y_rotate"][11] - trajectory["y_rotate"][0]]
            v_unit = v / np.linalg.norm(v)
            err_in_headings.append(compute_angle(v_unit, in_vitro_headings[BOT_ID]))

        # Average all heading vectors
        mean_err = np.mean(np.asarray(err_in_headings), axis=0)

        std_err = np.std(np.asarray(err_in_headings), axis=0)

        # print()
        # print()

        print(mean_err, std_err)

        # Write line to csv 
        row = [TX, str(mean_err), str(std_err)]
        writer.writerow(row)

    # Save out csv 
    f_write.close()
