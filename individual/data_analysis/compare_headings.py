import pandas as pd
import numpy as np
from glob import glob
import matplotlib.pyplot as plt

def compute_vector_difference(v1, v2):
    v1 = np.asarray(v1)
    v2 = np.asarray(v2)
    return np.linalg.norm(v1-v2)


############### Extract in vitro heading data ###############
path_to_in_vitro_headings = 'heading_data/in_vitro/headings.csv'

in_vitro_headings_df = pd.read_csv(path_to_in_vitro_headings)
in_vitro_headings = {}

for row in in_vitro_headings_df.iterrows():
    bot_id = row[1]['bot_id']
    x = float(row[1]['x'])
    y = float(row[1]['y'])

    in_vitro_headings[bot_id] = (x,y)

############### Extract in silico heading data ###############
in_silico_headings_files = glob('heading_data/in_silico/*.csv')

ctrl_headings = {}
ctrl2_headings = {}
ctrl3_headings = {}
tx_headings = {}

ctrl_headings_std = {}
ctrl2_headings_std = {}
ctrl3_headings_std = {}
tx_headings_std = {}

for filename in in_silico_headings_files:
    BOT_ID = filename.split('/')[-1].split('.')[0]

    df = pd.read_csv(filename)

    for row in df.iterrows():
        TX = row[1]['treatment'].split('_')[-1]

        x = float(row[1]['x'])
        y = float(row[1]['y'])

        std_x = float(row[1]['std_x'])
        std_y = float(row[1]['std_y'])

        if TX == "ctrl":
            ctrl_headings[BOT_ID] = (x,y)
            ctrl_headings_std[BOT_ID] = (std_x,std_y)
        elif TX == "ctrl2":
            ctrl2_headings[BOT_ID] = (x,y)
            ctrl2_headings_std[BOT_ID] = (std_x,std_y)
        elif TX == "ctrl3":
            ctrl3_headings[BOT_ID] = (x,y)
            ctrl3_headings_std[BOT_ID] = (std_x,std_y)
        elif TX == "tx":
            tx_headings[BOT_ID] = (x,y)
            tx_headings_std[BOT_ID] = (std_x,std_y)

# ############### Individual: Plot vectors and std bars ###############
# colors = {'vitro': 'fuchsia', 'tx':'blue', 'ctrl':'limegreen', 'ctrl2':'darkgreen', 'ctrl3': 'deepskyblue'}
# for bot in ctrl_headings:

#     print(bot) 

#     ctrl_x, ctrl_y = ctrl_headings[bot]
#     ctrl2_x, ctrl2_y = ctrl2_headings[bot]
#     ctrl3_x, ctrl3_y = ctrl3_headings[bot]
#     tx_x, tx_y = tx_headings[bot]

#     # print(np.linalg.norm(ctrl_headings[bot]/np.linalg.norm(ctrl_headings[bot])))

#     vitro_x, vitro_y = in_vitro_headings[bot]
    
#     fig, ax = plt.subplots(1,1, figsize=(7,7))
#     ax.quiver(0,0, ctrl_x, ctrl_y, color='limegreen', scale=30, units='xy')
#     ax.quiver(0,0, ctrl2_x, ctrl2_y, color='darkgreen', scale=30, units='xy')
#     ax.quiver(0,0, ctrl3_x, ctrl3_y, color='deepskyblue', scale=30, units='xy')
#     ax.quiver(0,0, tx_x, tx_y, color='blue', scale=30, units='xy')
#     ax.quiver(0,0, vitro_x, vitro_y, color='fuchsia', scale=30, units='xy')
#     plt.legend(['ctrl','ctrl2','ctrl3','tx','vitro'])
#     # fig.tight_layout()
#     plt.axis('off')
#     plt.show()
#     # plt.savefig('plots/{}/heading_vectors.png'.format(bot), bbox_inches='tight', dpi=300)
#     # plt.close()
#     exit()  


# ############### Aggregate: Compute difference in heading btwn in silico and in vitro for each treatment ###############

# Difference = euclidean distance (l2 norm of the vector between the in vitro and treatment vectors)

# ctrl_diffs = []
# ctrl2_diffs = []
# ctrl3_diffs = []
# tx_diffs = []

# for bot in ctrl_headings:
#     ctrl_diffs.append(compute_vector_difference(ctrl_headings[bot], in_vitro_headings[bot]))
#     ctrl2_diffs.append(compute_vector_difference(ctrl2_headings[bot], in_vitro_headings[bot]))
#     ctrl3_diffs.append(compute_vector_difference(ctrl3_headings[bot], in_vitro_headings[bot]))
#     tx_diffs.append(compute_vector_difference(tx_headings[bot], in_vitro_headings[bot]))


# ctrl_mean_diff = np.nanmean(ctrl_diffs)
# ctrl_std_diff = np.nanstd(ctrl_diffs)

# ctrl2_mean_diff = np.nanmean(ctrl2_diffs)
# ctrl2_std_diff = np.nanstd(ctrl2_diffs)

# ctrl3_mean_diff = np.nanmean(ctrl3_diffs)
# ctrl3_std_diff = np.nanmean(ctrl3_diffs)

# tx_mean_diff = np.nanmean(tx_diffs)
# tx_std_diff = np.nanmean(tx_diffs)

# plt.scatter([1,2,3,4], [ctrl_mean_diff, ctrl2_mean_diff, ctrl3_mean_diff, tx_mean_diff])
# plt.show()


############### Aggregate: Compute difference in heading btwn in silico and in vitro for each treatment ###############

# Difference = angle between the in vitro and treatment vectors

def compute_angle(v1,v2):
    # v2 should always be the in vitro heading
    angle_v1 = np.arctan2(v1[1], v1[0]) *180 /np.pi # arctan(y,x) gives angle from positive x-axis in radians
    angle_v2 = np.arctan2(v2[1], v2[0]) *180 /np.pi

    return np.abs(angle_v2 - angle_v1)

## Compute the average difference and the average weight for each treatment 
# Weight quantifies uncertainty in statistical test - low weight corresponds to high uncertainty
# Weight here is 1/||v1||_2 where v1 is the average vector for the given treatment
# If variance was 0 (all runs produce the same vector) then the magnitude would be 1 and the weight would be 1
# If the variance is high, the magnitude of the average will be smaller (vectors will cancel), low weight correpsonds to high uncertainty

ctrl_diffs_str = []
ctrl2_diffs_str = []
ctrl3_diffs_str = []
tx_diffs_str = []

ctrl_weight_str = []
ctrl2_weight_str = []
ctrl3_weight_str = []
tx_weight_str = []

ctrl_diffs_circ = []
ctrl2_diffs_circ = []
ctrl3_diffs_circ = []
tx_diffs_circ = []

ctrl_weight_circ = []
ctrl2_weight_circ = []
ctrl3_weight_circ = []
tx_weight_circ = []

for bot in ctrl_headings:
    print(bot)

    if "bot" in bot: # Straight bot
        ctrl_diffs_str.append(compute_angle(ctrl_headings[bot], in_vitro_headings[bot]))
        ctrl2_diffs_str.append(compute_angle(ctrl2_headings[bot], in_vitro_headings[bot]))
        ctrl3_diffs_str.append(compute_angle(ctrl3_headings[bot], in_vitro_headings[bot]))
        tx_diffs_str.append(compute_angle(tx_headings[bot], in_vitro_headings[bot]))

        ctrl_weight_str.append(np.linalg.norm(ctrl_headings[bot]))
        ctrl2_weight_str.append(np.linalg.norm(ctrl2_headings[bot]))
        ctrl3_weight_str.append(np.linalg.norm(ctrl3_headings[bot]))
        tx_weight_str.append(np.linalg.norm(tx_headings[bot]))

    else:
        ctrl_diffs_circ.append(compute_angle(ctrl_headings[bot], in_vitro_headings[bot]))
        ctrl2_diffs_circ.append(compute_angle(ctrl2_headings[bot], in_vitro_headings[bot]))
        ctrl3_diffs_circ.append(compute_angle(ctrl3_headings[bot], in_vitro_headings[bot]))
        tx_diffs_circ.append(compute_angle(tx_headings[bot], in_vitro_headings[bot]))

        ctrl_weight_circ.append(np.linalg.norm(ctrl_headings[bot]))
        ctrl2_weight_circ.append(np.linalg.norm(ctrl2_headings[bot]))
        ctrl3_weight_circ.append(np.linalg.norm(ctrl3_headings[bot]))
        tx_weight_circ.append(np.linalg.norm(tx_headings[bot]))

    # print(compute_angle(ctrl_headings[bot], in_vitro_headings[bot]))
    # print(compute_angle(ctrl2_headings[bot], in_vitro_headings[bot]))
    # print(compute_angle(ctrl3_headings[bot], in_vitro_headings[bot]))
    # print(compute_angle(tx_headings[bot], in_vitro_headings[bot]))

    # print(np.linalg.norm(ctrl_headings[bot]))
    # print(np.linalg.norm(ctrl2_headings[bot]))
    # print(np.linalg.norm(ctrl3_headings[bot]))
    # print(np.linalg.norm(tx_headings[bot]))

# Compute means

# Straight only 
ctrl_mean_diff_str = np.nanmean(ctrl_diffs_str)
ctrl_std_diff_str = np.nanstd(ctrl_diffs_str)
ctrl_mean_weight_str = np.nanmean(ctrl_weight_str)

ctrl2_mean_diff_str = np.nanmean(ctrl2_diffs_str)
ctrl2_std_diff_str = np.nanstd(ctrl2_diffs_str)
ctrl2_mean_weight_str = np.nanmean(ctrl_weight_str)

ctrl3_mean_diff_str = np.nanmean(ctrl3_diffs_str)
ctrl3_std_diff_str = np.nanmean(ctrl3_diffs_str)
ctrl3_mean_weight_str = np.nanmean(ctrl3_weight_str)

tx_mean_diff_str = np.nanmean(tx_diffs_str)
tx_std_diff_str = np.nanmean(tx_diffs_str)
tx_mean_weight_str = np.nanmean(tx_weight_str)

# Circ only
ctrl_mean_diff_circ = np.nanmean(ctrl_diffs_circ)
ctrl_std_diff_circ = np.nanstd(ctrl_diffs_circ)
ctrl_mean_weight_circ = np.nanmean(ctrl_weight_circ)

ctrl2_mean_diff_circ = np.nanmean(ctrl2_diffs_circ)
ctrl2_std_diff_circ = np.nanstd(ctrl2_diffs_circ)
ctrl2_mean_weight_circ = np.nanmean(ctrl2_weight_circ)

ctrl3_mean_diff_circ = np.nanmean(ctrl3_diffs_circ)
ctrl3_std_diff_circ = np.nanmean(ctrl3_diffs_circ)
ctrl3_mean_weight_circ = np.nanmean(ctrl3_weight_circ)

tx_mean_diff_circ = np.nanmean(tx_diffs_circ)
tx_std_diff_circ = np.nanmean(tx_diffs_circ)
tx_mean_weight_circ = np.nanmean(tx_weight_circ)

# All
ctrl_mean_diff = np.nanmean(ctrl_diffs_circ+ctrl_diffs_str)
ctrl_std_diff = np.nanstd(ctrl_diffs_circ + ctrl_diffs_str)
ctrl_mean_weight = np.nanmean(ctrl_weight_circ+ctrl_weight_str)

ctrl2_mean_diff = np.nanmean(ctrl2_diffs_circ+ctrl2_diffs_str)
ctrl2_std_diff = np.nanstd(ctrl2_diffs_circ+ctrl2_diffs_str)
ctrl2_mean_weight = np.nanmean(ctrl2_weight_circ+ctrl2_weight_str)

ctrl3_mean_diff = np.nanmean(ctrl3_diffs_circ+ctrl3_diffs_str)
ctrl3_std_diff = np.nanmean(ctrl3_diffs_circ+ctrl3_diffs_str)
ctrl3_mean_weight = np.nanmean(ctrl3_weight_circ+ctrl3_weight_str)

tx_mean_diff = np.nanmean(tx_diffs_circ+tx_diffs_str)
tx_std_diff = np.nanmean(tx_diffs_circ+tx_diffs_str)
tx_mean_weight = np.nanmean(tx_weight_circ+tx_weight_str)


plt.scatter([1,2,3,4], [ctrl_mean_diff_circ, ctrl2_mean_diff_circ, ctrl3_mean_diff_circ, tx_mean_diff_circ])
plt.show()

plt.scatter([1,2,3,4], [ctrl_mean_weight_circ, ctrl2_mean_weight_circ, ctrl3_mean_weight_circ, tx_mean_weight_circ])
plt.show()
