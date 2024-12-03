from glob import glob
import pickle
import csv
import pandas as pd
import os
import numpy as np
from pkg_resources import run_script

# BOT_ID = 'Run4group0subject2'

#####################################################
# Cut this section out and reduce indent if only trying to make csvs for a single bot

results_dirs = glob('results/*/')

for result_dir in results_dirs:
    BOT_ID = result_dir.split('/')[1]

    os.makedirs('CSV/'+BOT_ID, exist_ok=True)

    #################################################

    results_reports = glob('results/'+BOT_ID+'/*.p')

    for report in results_reports:

        if '_ctrl_' in report:
            CSV_SAVE_DIR = 'CSV/'+BOT_ID+'/ctrl'
            os.makedirs(CSV_SAVE_DIR, exist_ok=True)
        if '_ctrl2_' in report:
            CSV_SAVE_DIR = 'CSV/'+BOT_ID+'/ctrl2'
            os.makedirs(CSV_SAVE_DIR, exist_ok=True)
        if '_ctrl3_' in report:
            CSV_SAVE_DIR = 'CSV/'+BOT_ID+'/ctrl3'
            os.makedirs(CSV_SAVE_DIR, exist_ok=True)
        if '_tx_' in report:
            CSV_SAVE_DIR = 'CSV/'+BOT_ID+'/tx'
            os.makedirs(CSV_SAVE_DIR, exist_ok=True)

        with open(report,'rb') as f:
            all_runs = pickle.load(f)

        for run in all_runs:

            trajectory = all_runs[run]

            # Edit data frame to add additional columns for R analysis
            # Add 'id' column : 1 for all rows
            trajectory['id'] = np.ones(trajectory.shape[0], dtype=int)

            # Add 'frame' column : 1-len(dataframe) counting up
            trajectory['frame'] = np.arange(1,trajectory.shape[0]+1)

            # Add 'track_fixed' : column (1 for all rows)
            trajectory['track_fixed'] = np.ones(trajectory.shape[0], dtype=int)

            # Add 'ignore' column : FALSE for all rows 
            trajectory['ignore']=('FALSE',)*trajectory.shape[0]

            # CSV_SAVE_FN = CSV_SAVE_DIR + '/' + run.split('.')[0] + '.csv'
            # print(CSV_SAVE_FN)

            # Save CSV in the correct bot/treatment folder with just the run number as the filename
            CSV_SAVE_FN = CSV_SAVE_DIR + '/' + run.split('run')[-1].split('.')[0] +'.csv'
            # bot_id = run.split('Run')[1].split('group')[0] + run.split('group')[1].split('subject')[0]+run.split('subject')[1].split('_')[0]
            trajectory.to_csv(CSV_SAVE_FN)

        # # Make an info.xlsx for each folder/tx each

        # info = pd.DataFrame()
        # n_rows = len(all_runs)
        # info['replicate'] = np.arange(n_rows)
        # info['condition']=('stuff',)*n_rows
        # info['age'] = (6,)*n_rows
        # info['scale'] = (0.001,)*n_rows
        # info['top_left']=('(434, 376)',)*n_rows
        # info['top_right']=('(10000, 10000)',)*n_rows
        # INFO_SAVE_FN = CSV_SAVE_DIR + '/info.csv'
        # info.to_csv(INFO_SAVE_FN)



