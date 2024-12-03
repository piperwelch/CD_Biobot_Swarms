'''
8/14/24


'''
from scipy.stats import mannwhitneyu
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
import numpy as np
from frechetdist import frdist
import itertools
from scipy.stats import circvar
import matplotlib.cm as cm

# data written out using data_analysis/plot_trajectories.py and in_vitro_analysis/rotate_segments.py

def plot_in_vitro_trajectories(bot_id):
    filenames = glob(f'in_vitro/in_vitro/{bot_id}/run*.csv')
    strness = compute_straightness_index(bot_id, group='in_vitro', all_vals=True)
    norm = plt.Normalize(0, 1)    
    for idx, filename in enumerate(filenames):
        df = pd.read_csv(filename)
        value_cmap = strness[filename]
        color = cm.jet(norm(value_cmap))
        plt.plot(df['x_rotate'], df['y_rotate'], color=color,linestyle='-') 
        # plt.savefig(f"{bot_id}_idx{idx}.png")
        # plt.close()


    plt.xlabel('x')
    plt.ylabel('y')
    plt.gca().set_aspect('equal', adjustable='box')
    # plt.xlim([-0.8,0.2])
    # plt.ylim([-20,20])
    plt.savefig(f"{bot_id}.png")
    plt.close()

def plot_in_silico_trajectories(bot_id, tx='tx'):
    filenames = glob(f'in_silico/in_silico/{bot_id}/{tx}/run*.csv')
    strness = compute_straightness_index(bot_id, tx, all_vals=True)
    norm = plt.Normalize(0, 1)          
    for filename in filenames:
        df = pd.read_csv(filename)
        value_cmap = strness[filename]
        color = cm.jet(norm(value_cmap))
        plt.plot(df['x_rotate'], df['y_rotate'], linestyle='-', color=color)  # 'o' for points, '-' for line
    plt.xlabel('x')
    plt.ylabel('y')
    plt.gca().set_aspect('equal', adjustable='box')
    # plt.xlim([-0.8,0.2])
    # plt.ylim([-0.4,0.6])
    plt.savefig(f"{bot_id}_{tx}.png")
    plt.close()

def get_frdist(bot_id, tx='tx', folder='in_vitro'):
    for folder in ['in_vitro', 'in_silico']:
        if folder == "in_vitro":
            filenames = glob(f'{folder}/{folder}/{bot_id}/run*.csv')
        else:
            filenames = glob(f'{folder}/{folder}/{bot_id}/{tx}/run*.csv')

        filenames = filenames[:16]
        pairs = list(itertools.combinations(filenames, 2))
        print(len(pairs))
        frdists = []
        for pair in pairs:
            df1 = pd.read_csv(pair[0])
            df2 = pd.read_csv(pair[1])
            
            trajectory1 = np.column_stack((df1['x_rotate'], df1['y_rotate']))
            trajectory2 = np.column_stack((df2['x_rotate'], df2['y_rotate']))
            max_length = min([trajectory1.shape[0], trajectory2.shape[0]])

            trajectory1 = trajectory1[:max_length]
            trajectory2 = trajectory2[:max_length]

            trajectory1 /= np.linalg.norm(trajectory1[-1])
            trajectory2 /= np.linalg.norm(trajectory2[-1])

            distance = frdist(trajectory1, trajectory2)
            frdists.append(distance)
            # if distance > 20: 
            #     plt.plot(df1['x_rotate'], df1['y_rotate'], marker='o', linestyle='-', color='k')  # 'o' for points, '-' for line
            #     plt.plot(df2['x_rotate'], df2['y_rotate'], marker='o', linestyle='-', color='k')  # 'o' for points, '-' for line
            #     plt.show()
            # print(pair)
            # print(f"The Fréchet distance between the trajectories is: {distance}")
        print(f'done {folder}')
        plt.hist(frdists, bins='auto', alpha =0.5, label=folder)
    plt.ylabel('Count')
    plt.legend()
    plt.xlabel('Fréchet distance')
    plt.show()


def compute_mean_curvature(bot_id, tx='tx', group='in_silico', normalize='False'):
    curvatures = []
    filenames = glob(f'in_silico/in_silico/{bot_id}/{tx}/run*.csv')
    if group == "in_vitro":
        filenames = glob(f'in_vitro/in_vitro/{bot_id}/run*.csv')
        
    for filename in filenames:
        df = pd.read_csv(filename)
        # print(filename)
    
        # https://www.delftstack.com/howto/numpy/curvature-formula-numpy/
        x_coordinates = df['x_rotate']
        if len(x_coordinates) < 3: 
            continue 
        y_coordinates = df['y_rotate']
        if normalize:
            x_min = np.min(x_coordinates)
            x_max = np.max(x_coordinates)
            x_normalized = (x_coordinates - x_min) / (x_max - x_min)

            # Normalize the y coordinates
            y_min = np.min(y_coordinates)
            y_max = np.max(y_coordinates)
            y_normalized = (y_coordinates - y_min) / (y_max - y_min)

            x_coordinates,y_coordinates = discard_duplicate_points(x_normalized,y_normalized)
        else: 
            x_coordinates,y_coordinates = discard_duplicate_points(x_normalized,y_normalized)

        # compute gradient at each point
        x_t = np.gradient(x_coordinates) # gradient in x direction
        y_t = np.gradient(y_coordinates) # gradient in y direction

        xx_t = np.gradient(x_t)
        yy_t = np.gradient(y_t)

        curvature_val = np.abs(xx_t * y_t - x_t * yy_t) / (x_t * x_t + y_t * y_t)**1.5
        curvatures.append(np.mean(curvature_val))

    return np.mean(curvatures)


def discard_duplicate_points(x_coords, y_coords):
    prev_x = -10000
    prev_y = -10000

    unique_x = []
    unique_y = []

    for i in range(1,len(x_coords)):
        
        if x_coords[i] == prev_x and y_coords[i] == prev_y:
            pass
        else:
            unique_x.append(x_coords[i])
            unique_y.append(y_coords[i])

            prev_x = x_coords[i]
            prev_y = y_coords[i]
    
    new_coordinates = np.concatenate((np.asarray(unique_x).reshape(-1,1), np.asarray(unique_y).reshape(-1,1)), axis=1)
    
    return new_coordinates[:,0], new_coordinates[:,1]


def compute_heading(x_pts, y_pts):
    # computes headings from x and y coordinates of the trajectory
    # heading is computed between pairs of points and therefore cannot be compute for the first and last points 
    headings = []
    for i in range(1,len(x_pts)):
        x_diff = x_pts[i]-x_pts[i-1]
        y_diff = y_pts[i]-y_pts[i-1]
        heading = np.arctan2(y_diff,x_diff)
        headings.append(heading)
    
    return headings

def compute_straightness_index(bot_id, tx='tx', group='in_silico', all_vals=False):
    # Computes straightness index from headings (circular variance of headings)
    filenames = glob(f'in_silico/in_silico/{bot_id}/{tx}/run*.csv')
    if group == "in_vitro":
        filenames = glob(f'in_vitro/in_vitro/{bot_id}/run*.csv')
    strsness = []
    if all_vals: strsness_dict = {}

    for filename in filenames:
        df = pd.read_csv(filename)
        # https://www.delftstack.com/howto/numpy/curvature-formula-numpy/
        x_coordinates = df['x_rotate']
        if len(x_coordinates) < 3: 
            continue 
        y_coordinates = df['y_rotate']

        headings = compute_heading(x_coordinates, y_coordinates)

        # compute circular variance of the headings (between 0-2pi)
        circva = circvar(headings,low=-np.pi, high=np.pi)
        if all_vals: strsness_dict[filename] = 1 - circva
        else: strsness.append(1 - circva)
    if all_vals: return strsness_dict
    else: return np.mean(strsness)


if __name__=='__main__':
    bot_id = "Run5group7subject1"
    plot_in_silico_trajectories(bot_id, tx='ctrl3')
    plot_in_silico_trajectories(bot_id, tx='ctrl')
    plot_in_silico_trajectories(bot_id, tx='ctrl2')
    plot_in_silico_trajectories(bot_id, tx='tx')
    plot_in_vitro_trajectories(bot_id)
    quit()
    curvatures = {}
    straightnesses = {}
    file = open("straightness_curvature_normalized1.csv", "w")
    file.write("bot_id,straightness,curvature,treatment\n")

    folders = ['ctrl', 'ctrl2', 'ctrl3', 'tx', 'in_vitro']
    for folder in folders:
        curvatures[folder] = []
        straightnesses[folder] = []


    in_silico_fldrs = [f.split("\\")[-1] for f in glob("in_silico\\in_silico\\Run*")]
    ignore_folders = []

    for fldr in glob("in_vitro\\in_vitro\\Run*"):
        if fldr.split("\\")[-1] not in in_silico_fldrs:
            ignore_folders.append((fldr.split("\\")[-1]))

    for folder in ['ctrl', 'ctrl2', 'ctrl3', 'tx']:
        print(folder)
        for bot_folder in glob('in_silico\\in_silico\\*'):
            BOT_ID = bot_folder.split("in_silico\\")[-1]
            curvature = compute_mean_curvature(BOT_ID, folder)
            straightness = compute_straightness_index(BOT_ID, folder)
            curvatures[folder].append(curvature)
            straightnesses[folder].append(straightness)
            file.write(f"{BOT_ID},{straightness},{curvature},{folder}\n")

        
    for bot_folder in glob('in_vitro\\in_vitro\\*'):
        BOT_ID = bot_folder.split("in_vitro\\")[-1]
        if BOT_ID == "Run4group3subject2":
            plot_in_vitro_trajectories(BOT_ID)

        if BOT_ID in ignore_folders:
            print(BOT_ID)
            continue
        curvature = compute_mean_curvature(BOT_ID, group='in_vitro')
        straightness = compute_straightness_index(BOT_ID, group='in_vitro')
        curvatures['in_vitro'].append(curvature)
        straightnesses['in_vitro'].append(straightness)
        file.write(f"{BOT_ID},{straightness},{curvature},{'in_vitro'}\n")
    file.close()
