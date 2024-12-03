'''
Prepares results for analysis

For a collection of runs on a single bot:
Reads fitness/time data from result .xml files to a dictionary results_dict
Reads trajectories to a dictionary trajectory_dict
'''

import numpy as np
from lxml import etree
from glob import glob
import os
import pickle
import pandas as pd
import sys

# def read_results_from_xml(DIR):
        
#     # PATH = os.path.abspath(__file__ + "/../../") + '/' + DIR 

#     filenames = glob(DIR+"/*.xml")

#     results_dict = {}

#     for i,fn in enumerate(filenames):

#         # preprocess files to remove any extra text at the bottom of the xml
#         with open(fn,'r') as infile:
#             temp = infile.read()
#             end = temp.find('</report>')+9
#             temp2 = temp[:end]

#         if not temp2:
#             return {}
            
#         # save back to file
#         with open(fn, 'w') as outfile:
#             outfile.write(temp2)
        
#         with open(fn, 'r') as f:
#             report = etree.parse(f)

#         robots = report.xpath("//detail/*")
#         for j,robot_report in enumerate(robots):

#             results_dict[i] = {}
#             results_dict[i]["ID"] = robot_report.tag
#             distance = round(float(robot_report.xpath("fitness_score")[-1].text), ndigits=6)
#             time = round(float(robot_report.xpath("currentTime")[-1].text), ndigits=6)

#             results_dict[i]["distance"] = np.sqrt(distance)
#             results_dict[i]["time"] = time
#     return results_dict

def read_trajectories(DIR):
    '''
    Reads trajectories into pandas dataframes 
    Stores all trajectories for a set of simulations in a dictionary
        {filename : trajectory}
    '''

    # PATH = os.path.abspath(__file__ + "/../../") + '/' + DIR 

    history_files = glob(DIR+"/*.history")

    trajectories = {}

    for i, filename in enumerate(history_files):
        f = open(filename, "r")
        # print(filename)
        
        line = f.readline()

        while line and "real_stepsize" not in line:
            line = f.readline()

        # beginning of trajectory data
        line = f.readline()

        timesteps = []
        x_coords = []
        y_coords = []

        while line and "Simulation" not in line and "Stopping" not in line:
            coords = line.split(',')
            t = int(coords[0].split('}')[0].split("{")[-1])
            x = float(coords[0].split('}')[-1])
            y = float(coords[1])

            timesteps.append(t)
            x_coords.append(x)
            y_coords.append(y)

            line = f.readline()
        
        f.close()

        if len(timesteps)!=0:
            data = np.zeros((len(x_coords), 3))
            data[:,0] = timesteps
            data[:,1] = x_coords
            data[:,2] = y_coords
            
            df = pd.DataFrame(data,columns=['t','x','y'])
        
            trajectories[filename.split('/')[-1]]=df
        else:
            print("no data:", filename)
    print("Found data for",len(trajectories),"runs")
    return trajectories

if __name__ == "__main__":

    # Example: OUTPUT_PATH = "output_data/Run4group0subject2/Run4group0subject2_res16_ctrl"

    # ################# TESTING #################
    # ID = 'test_sphere_right'
    # OUTPUT_PATH = "testing/test_ctrl/spheres/output_data_fix/{}".format(ID)

    # trajectory_dict = read_trajectories(OUTPUT_PATH)

    # SAVE_DIR = "testing/test_ctrl/spheres/results_fix/"
    # os.makedirs(SAVE_DIR, exist_ok=True)

    # with open("{}/{}_results_report.p".format(SAVE_DIR, ID), 'wb') as f:
    #     pickle.dump(trajectory_dict, f)
    # ###################################################

    ################# NORMAL RUNS #################    
    OUTPUT_PATH = sys.argv[1]    
    RUN_ID = OUTPUT_PATH.split('/')[-1]
    BOT_ID = OUTPUT_PATH.split('/')[-2]
    SAVE_DIR = "results/" + BOT_ID 

    trajectory_dict = read_trajectories(OUTPUT_PATH)

    os.makedirs(SAVE_DIR, exist_ok=True)

    with open("{}/{}_results_report.p".format(SAVE_DIR, RUN_ID), 'wb') as f:
        pickle.dump(trajectory_dict, f)
    ###################################################
