import numpy as np
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
import pylab
import pickle
import os
from matplotlib import cm
from matplotlib.colors import Normalize
import math

def clean(df, ID_NBR=None):
    # filters out ignore=True rows
    # track fixed column is the BOT ID 
    clean_df = df[df['ignore']==False].reset_index(drop=True)

    if ID_NBR is not None:
        clean_df = clean_df[clean_df['track_fixed']==int(ID_NBR)].reset_index(drop=True)
    return clean_df

def split(df, frames_per_chunk):
    # split dataframe into X sec chunks
    # returns list of dataframes

    if len(df) == 0:
        return None
    
    number_of_frames = list(df['frame'])[-1]

    curr_frames_left = number_of_frames

    if number_of_frames<=frames_per_chunk:
        return [df]

    chunks = []
    while curr_frames_left>0:
        start = number_of_frames-curr_frames_left+1
        stop = start+frames_per_chunk
        
        if stop>number_of_frames: # fewer than 1200 frames
            stop=start+curr_frames_left

        # print(start, stop)
        # print(df.loc[(df['frame']>=start) & (df['frame']<=stop)])

        chunk = df.loc[(df['frame']>=start) & (df['frame']<=stop)].reset_index(drop=True)
        # print(chunk)
        # exit()
        chunks.append(chunk)

        # print(chunk)

        curr_frames_left-=(stop-start)
    
    return chunks

def shift_trajectory(trajectory, shiftx, shifty):
    shifted_df = pd.DataFrame(columns=['x','y'])

    for i,row in trajectory.iterrows():
        newx = row['x'] + shiftx
        newy = row['y'] + shifty
        shifted_df = shifted_df.append({'x':newx, 'y':newy}, ignore_index=True)

    return shifted_df
 
frame_rate = 2.5 #fps
splt = 2
frames_per_chunk = frame_rate*splt # 5 sec chunks

# os.makedirs('in_vitro_analysis/overlayed_trajectories_{}sec_split/'.format(splt),exist_ok=True)

BOT_ID = 'Run5group7subject1'

# os.makedirs('in_vitro_analysis/{}/'.format(BOT_ID),exist_ok=True)

# TRAJECTORY_COLOR = (0,0,1,0.6)
hsv_cmap = cm.get_cmap('hsv')

# NUM_COLORS = 100
# cm = pylab.get_cmap('hsv')
# colors = []

# for i in range(NUM_COLORS):
#     color = cm(1.*i/NUM_COLORS)  # color will now be an RGBA tuple
#     colors.append((color[0],color[1],color[2], 0.6))

#############################################################
# # Remove this chunk and decrease indent to process a single bot
# BOT_IDs=[]
# for bot_csv in glob('in_vitro_analysis/in_vitro_behavior_data/*.csv'):
#     strp = bot_csv.split('/')[-1]
#     if '_' in strp:
#         BOT_ID = strp.split('_')[0]
#     else:
#         BOT_ID = strp.split('.')[0]
    
#     if BOT_ID not in BOT_IDs:
#         BOT_IDs.append(BOT_ID)

# for BOT_ID in BOT_IDs:
#############################################################

fig, ax = plt.subplots(1,1, figsize=(7,7))
fig.tight_layout()
plt.axis('equal')

for i,bot_csv in enumerate(glob('in_vitro_analysis/in_vitro_behavior_data/{}*.csv'.format(BOT_ID))):

    print(bot_csv)

    # Read in CSV and clean
    df_raw = pd.read_csv(bot_csv)

    if 'subject' in BOT_ID:
        ID_NBR = BOT_ID.split('subject')[-1]
    else:
        ID_NBR = None

    if "ignore" in df_raw.columns:
        df = clean(df_raw, ID_NBR)
    else:
        df = df_raw

    trajectories = split(df, frames_per_chunk)

    if trajectories is None:
        continue

    # if i==0:
    #     x_start = trajectories[0]["x"][0]
    #     y_start = trajectories[0]["y"][0]

    for j,trajectory in enumerate(trajectories):
        if len(trajectory)==0:
            continue
        # if i==0 and j==0: # plot trajectory as is
        #     ax.plot(trajectory["x"][0], trajectory["y"][0], '*r')
        #     ax.plot(trajectory["x"], trajectory["y"], color=colors[np.random.choice(len(colors))])
        # else:
        # shift trajectory to start at (x_start,y_start)
        x_start2 = trajectory["x"][0]
        y_start2 = trajectory["y"][0]

        # shift all trajectories to (0,0)
        shiftx = 0-x_start2
        shifty = 0-y_start2

        shifted_tajectory = shift_trajectory(trajectory, shiftx,shifty)

        # heading vector is an angle in radians in the range [-  pi,pi]
        x1 = shifted_tajectory['x'][0] 
        y1 = shifted_tajectory ['y'][0]

        x2 = shifted_tajectory['x'][1]
        y2 = shifted_tajectory ['y'][1]

        init_heading = [y2-y1, x2-x1]
        angle_pos_degrees = (math.atan2(init_heading[0], init_heading[1]) * 180 / math.pi - 90) % 360

        norm = Normalize(vmin=0, vmax=360)
        normalized_heading = norm(angle_pos_degrees)
    
        # ax.plot(shifted_tajectory["x"][0], shifted_tajectory["y"][0], '*r')
        ax.plot(shifted_tajectory["x"][0], shifted_tajectory["y"][0], 'k', marker='*', zorder=2)
        ax.plot(shifted_tajectory["x"], shifted_tajectory["y"], color=hsv_cmap(normalized_heading))
        # ax.plot(shifted_tajectory["x"], shifted_tajectory["y"], color=colors[np.random.choice(len(colors))])
            
    
# plt.show()
# plt.savefig('in_vitro_analysis/{}/{}sec_trajectory.png'.format(BOT_ID, splt), dpi=300, bbox_inches='tight')
# plt.savefig('in_vitro_analysis/overlayed_trajectories_{}sec_split/{}_trajectories.png'.format(splt,BOT_ID), dpi=300, bbox_inches='tight')

save_path = 'in_vitro_analysis/plots/trajectories/{}/{}_headingcmap.png'.format(BOT_ID, BOT_ID)
plt.savefig(save_path, dpi=300, bbox_inches='tight')

plt.close()


            
        
