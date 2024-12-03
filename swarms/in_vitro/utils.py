from scipy.stats import circvar
import numpy as np 
import math 
from collections import Counter
import constants
import pandas as pd 
import os 
from shapely.geometry import Polygon

def compute_entropy(xs):

    px = np.asarray(list(dict(Counter(xs)).values()), dtype=int)/len(xs)

    Hx = 0
    for i in range(len(px)):
        Hx+=px[i]*math.log(1/px[i], 2)
    return Hx


def compute_heading(x_pts, y_pts):

    headings = []
    for i in range(1,len(x_pts)):
        x_diff = x_pts[i]-x_pts[i-1]
        y_diff = y_pts[i]-y_pts[i-1]
        heading = np.arctan2(y_diff,x_diff)
        headings.append(heading)
    
    return headings


def compute_straightness_index(x_pts, y_pts):
    # Computes straightness index from headings (circular variance of headings)
    headings = compute_heading(x_pts, y_pts)
    # compute circular variance of the headings (between 0-2pi)
    circvar_raw = circvar(headings)
    # rescale circualr variance to be between 0-1
    circvar_normalized = circvar_raw / (2*np.pi)

    return 1 - circvar_normalized


def compute_heading_deltas(headings):
    deltas = np.diff(headings)
    norm_deltas = (deltas - (-2*np.pi))/(2*np.pi - (-2*np.pi))
    binned_norm_deltas = np.asarray(norm_deltas*100,dtype=int)
    return binned_norm_deltas


def compute_mean_curvature(x_coordinates, y_coordinates):
    "input is the bot's trajectory - a 2D array of x,y coordinates"
    
    # https://www.delftstack.com/howto/numpy/curvature-formula-numpy/

    # Calculate velocity of curve

    # compute gradient at each point
    x_t = np.gradient(x_coordinates) # gradient in x direction
    y_t = np.gradient(y_coordinates) # gradient in y direction

    #if both gradients are 0 then the equation gets upset 

    vel = np.array([[x_t[i], y_t[i]] for i in range(x_t.size)])

    speed = np.sqrt(x_t * x_t + y_t * y_t)

    tangent = np.array([1/speed] * 2).transpose() * vel

    ss_t = np.gradient(speed)
    xx_t = np.gradient(x_t)
    yy_t = np.gradient(y_t)

    curvature_val = np.abs(xx_t * y_t - x_t * yy_t) / (x_t * x_t + y_t * y_t)**1.5
    #curvature_val = curvature_val[~np.isnan(curvature_val)]

    return np.mean(curvature_val)

def parse_silico_single_bot(file_name):
    #the info in a history file is...
    #x * 1/0.001, y * 1/0.001, z * 1/0.001
    #4 values for information about angles

    f  = open("results/{}".format(file_name), "r")
    str_file = f.read()
    f.close()

    lst_file = str_file.split(">>>")
    lst_file = lst_file[1:]
    parsed = []

    for line in lst_file:
        p_line = line.split(";")
        if "Time" not in p_line[0]:
            parsed.append(line.split(";"))

    x_list, y_list = [], []

    for list in parsed: 
        for entry in list:
            points = entry.split(",")
            if len(points) != 1: #get xy coords
                x_list.append(float(points[0]))
                y_list.append(float(points[1]))
    return x_list, y_list

def parse_vitro_single_bot(csv_file, trim_frame=10000):
    file = open(csv_file, "r")
    file.readline()

    xs, ys = [],[]
    for line in file:
        if line == "\n":
            continue
        if "TRUE"  in line:
            continue
        data = line.split(",")
        if int(data[0]) > trim_frame:
            break

        xs.append(float(data[2]))
        ys.append(float(data[3]))
    

    return xs, ys

def check_motility(all_bots_points, threshold, min_frame):
    bot_motility_dict = {}
    min_dist = threshold

    for bot_id, v in all_bots_points.items():

        start_point = v[0]

        if bot_id not in bot_motility_dict:
            bot_motility_dict[bot_id] = [0]*(min_frame+1)
        
        for i in range(len(v)):
            point = v[i]
            lengths = math.sqrt(((start_point[0]-point[0])**2)+((start_point[1]-point[1])**2))
            start_point = point
            if lengths > min_dist:
                bot_motility_dict[bot_id][i] = 1
            else:
                print(lengths)
                bot_motility_dict[bot_id][i] = 0
    total_motile = 0 
    for k,v in bot_motility_dict.items():
        total_motile += sum(v)

    return total_motile


def check_speed(all_bots_points, frame_coordmap, min_frame):
    bot_motility_dict = {}

    for bot_id, v in all_bots_points.items():

        start_point = v[0]

        if bot_id not in bot_motility_dict:
            bot_motility_dict[bot_id] = [-1]*(min_frame+1)
        
        for i in range(len(v)):
            point = v[i]
            lengths = math.sqrt(((start_point[0]-point[0])**2)+((start_point[1]-point[1])**2))
            start_point = point
  
            bot_motility_dict[bot_id][frame_coordmap[(start_point[0], start_point[1])]-1] = lengths


    return bot_motility_dict

def check_collision(all_bots_points, bot_sizes, min_frame):

    bot_collision_dict = {}
    n = len(all_bots_points.keys())
    
    for i in range(1, n+1):
        bot_1_coords = all_bots_points[i]
        bot_1_sizes = bot_sizes[i]

        for j in range(i + 1, n+1):
            if i not in bot_collision_dict:
                bot_collision_dict[i] = [-1]*(min_frame+1)
            if j not in bot_collision_dict:
                bot_collision_dict[j] = [-1]*(min_frame+1)

            bot_2_coords = all_bots_points[j]
            bot_2_sizes = bot_sizes[j]
            # Calculate the half-lengths in x and y directions

            for pt_index in range(min(len(all_bots_points[i]), len(all_bots_points[j])) - 1):
                #need to add code for when bot is all by itself in arena 
                if -1 in bot_1_coords[pt_index] or -1 in bot_2_coords[pt_index]:
                    continue
                half_length_x1 = bot_1_sizes[pt_index][0]/2
                half_length_y1 = bot_1_sizes[pt_index][1]/2
                half_length_x2 = bot_2_sizes[pt_index][0]/2
                half_length_y2 = bot_2_sizes[pt_index][1]/2

                dx = abs(bot_1_coords[pt_index][0] - bot_2_coords[pt_index][0])
                dy = abs(bot_1_coords[pt_index][1] - bot_2_coords[pt_index][1])

                if dx <= half_length_x1 + half_length_x2 + 4 and dy <= half_length_y1 + half_length_y2 + 4: #collision has happened 
                    bot_collision_dict[i][pt_index] = 1
                    bot_collision_dict[j][pt_index] = 1
                else: #collision has not happened 
                    
                    if bot_collision_dict[i][pt_index] != 1: #don't over ride a previous collisions for this bot because there is not one here 
                        bot_collision_dict[i][pt_index] = 0
                    
                    if bot_collision_dict[j][pt_index] != 1:
                        bot_collision_dict[j][pt_index] = 0
    return bot_collision_dict


def get_entropy(traj1, traj2):
    # Convert the trajectories into probability distributions
    unique_values_1, counts_1 = np.unique(traj1, return_counts=True)
    unique_values_2, counts_2 = np.unique(traj2, return_counts=True)

    # Calculate probabilities for each unique value in the trajectories
    prob_distribution_1 = counts_1 / len(traj1)
    prob_distribution_2 = counts_2 / len(traj2)

    # Compute KL divergence between the two distributions
    kl_divergence = entropy(prob_distribution_1, prob_distribution_2)

    return kl_divergence


def get_indices(bot):
    csv_file = open("misc_files/10_7_swarm_data.csv", "r")
    file_contents = csv_file.readlines()

    bot = bot.split("/")[1].split(".")[0]

    for line in file_contents:
        if bot in line:
            return float(line.split(",")[1]), float(line.split(",")[2])
        
def evaluate_fitness(points):

    # Compute the center of the starting points

    # Get initial starting points of all bots
    x_starts = []
    y_starts = []
    all_coords_define = False
    for i in points:
        trajectory = points[i]
        x_starts.append(trajectory[0,0])
        y_starts.append(trajectory[0,1])

        # Also create array of all coordinates for use in computing the bounding box around the trajectories later
        if all_coords_define == False:
            all_coords = points[i]
            all_coords_define = True
        else:
            all_coords = np.concatenate((all_coords,points[i]))

    center_x = np.mean(x_starts)
    center_y = np.mean(y_starts)
    print(all_coords.shape)
    # Create a boundary of 20*constants.BOT_LENGTH
    min_x = 0
    max_x = 912
    min_y = 0
    max_y = 736

    # Count # unique points
    unique_points = np.unique(all_coords,axis=0)

    # Only keep unique points within the boundary

    # true if in bounds
    x_min_mask = np.reshape(unique_points[:,0]>min_x,newshape=(-1,1))
    x_max_mask = np.reshape(unique_points[:,0]<max_x,newshape=(-1,1))
    y_min_mask = np.reshape(unique_points[:,1]>min_y,newshape=(-1,1))
    y_max_mask = np.reshape(unique_points[:,1]<max_y,newshape=(-1,1))

    mask = np.all(np.concatenate((x_min_mask, x_max_mask, y_min_mask, y_max_mask),axis=1),axis=1)

    unique_points_in_bounds = unique_points[mask]

    return fractal_box_count(unique_points_in_bounds,(min_x,min_y,max_x,max_y))



def fractal_box_count(points, boundary):
    # https://francescoturci.net/2016/03/31/box-counting-in-numpy/

    min_x,min_y,max_x,max_y = boundary # unpack tuple

    Ns=[]

    # scales = np.arange(start=2, stop=int(constants.BOUNDARY_LENGTH/constants.MIN_GRID_DIM)) #start with quadrents and go to resolution of voxcraft float
    levels = np.arange(start=1, stop=constants.MAX_LEVEL)

    for level in levels: 

        scale = 2**level

        cell_width = 912/scale
        cell_height = 736/scale

        # H, edges=np.histogramdd(points, bins=(np.linspace(min_x,max_x,constants.BOUNDARY_LENGTH/scale),np.linspace(min_y,max_y,constants.BOUNDARY_LENGTH/scale)))
        H, edges=np.histogramdd(points, bins=(np.linspace(min_x,max_x,num=scale+1),np.linspace(min_y,max_y,num=scale+1)))

        weight = (cell_width*cell_height)/(912*736) # David scaling
        Ns.append(np.sum(H>0)*weight)

    # Divide by # of levels to get a value between 0-1
    scaled_box_count = np.sum(Ns)/len(levels) # David scaling

    return scaled_box_count

def parse_com_csv(csv_file, trim_frame):
    """Parses csv. 

    Args:
        history_file (str): Path to .csv file to be parsed.

    Returns:
        dict: Dictionary of trajectories. Key is the bot number and value is an array of size (n_timesteps x 2) where the columns 
                are the x,y coordinates of the CoM of that bot.
    """    
    file = open(csv_file, "r")
    trajectories = {}
    frame_number_coord_map = {}
    file.readline()
    
    for line in file:
        if line == "\n":
            continue
        data = line.split(",")
        # print(data)
        if int(data[0]) > trim_frame:
            break
        if int(float(data[-1])) > 4:
            
            bot_id = int(data[1])
        else: 
            bot_id = int(data[-1])
        if bot_id not in trajectories:
            trajectories[bot_id] = [] 


        trajectories[bot_id].append((float(data[2]), float(data[3])))
        frame_number_coord_map[(float(data[2]), float(data[3]))] = int(data[0])
        
    # reformat trajectory data

    trajectories_arr = {}

    for i in trajectories:
        points_touples = trajectories[i]

        points_arr = np.reshape(points_touples, newshape=(len(points_touples),2))
    
        trajectories_arr[i] = points_arr

    return trajectories_arr



def concatenate_bot_points(df, swarm_id, trial_id, max_frames=129):
    # Filter the dataframe for the specific swarm and trial
    df_filtered = df[(df['swarm_id'] == swarm_id) & (df['trial_id'] == trial_id)]
    
    # Initialize an empty list to store the points
    points_list = []

    # Loop through each bot_id in the filtered dataframe
    for bot_id in df_filtered['bot_id'].unique():
        bot_data = df_filtered[df_filtered['bot_id'] == bot_id].sort_values(by='frame')
        bot_data = bot_data[bot_data['frame'] <= max_frames]  # Filter to include only frames <= 129
        x = bot_data['x'].values
        y = bot_data['y'].values
        
        # Combine x and y into an n by 2 array
        bot_points = np.column_stack((x, y))
        points_list.append(bot_points)
    
    # Concatenate all bot points into a single array
    all_points = np.vstack(points_list)
    
    return all_points

def rotate_point(x, y, cx, cy, angle):
    """Rotate a point around a center with a given angle."""
    angle = np.deg2rad(angle)
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    nx = cos_angle * (x - cx) - sin_angle * (y - cy) + cx
    ny = sin_angle * (x - cx) + cos_angle * (y - cy) + cy
    return nx, ny

def get_corners(x, y, width, height, angle):
    """Get the corners of a rotated rectangle."""
    cx, cy = x, y
    hw, hh = width / 2, height / 2
    corners = [
        (cx - hw, cy - hh),
        (cx + hw, cy - hh),
        (cx + hw, cy + hh),
        (cx - hw, cy + hh)
    ]
    rotated_corners = [rotate_point(c[0], c[1], cx, cy, angle) for c in corners]
    return rotated_corners

def intersection_area(poly1, poly2):
    """Calculate the intersection area of two polygons."""
    poly1 = Polygon(poly1)
    poly2 = Polygon(poly2)
    if poly1.intersects(poly2):
        intersection = poly1.intersection(poly2)
        return intersection.area
    return 0.0

def is_collision(row, df, threshold=100.0):
    if pd.isna(row['x']) or pd.isna(row['y']) or pd.isna(row['width']) or pd.isna(row['height']):
        return np.nan
    corners1 = get_corners(row['x'], row['y'], row['width'], row['height'], row['angle'])
    other_bots = df[(df['swarm_id'] == row['swarm_id']) & 
                    (df['trial_id'] == row['trial_id']) & 
                    (df['frame'] == row['frame']) & 
                    (df['bot_id'] != row['bot_id'])]
    for _, other in other_bots.iterrows():
        if pd.isna(other['x']) or pd.isna(other['y']) or pd.isna(other['width']) or pd.isna(other['height']):
            continue
        corners2 = get_corners(other['x'], other['y'], other['width'], other['height'], other['angle'])
        # print(intersection_area(corners1, corners2))
        # print(intersection_area(corners1, corners2))
        if intersection_area(corners1, corners2) > threshold:
            return True
    return False

def interpolate_points(x, y, mean_dist):
    interpolated_x = []
    interpolated_y = []

    for i in range(len(x) - 1):
        # Get the current and next points
        x0, y0 = x[i], y[i]
        x1, y1 = x[i + 1], y[i + 1]
        distance = np.sqrt((x0-x1)**2 + (y0-y1)**2)
    
        if np.isnan(distance): #speed if nan if the coordinates are not in the tracking area
            interpolated_y.append(y0)
            interpolated_x.append(x0)
            continue
        
        #insert a number of points to make the curr distance traveled approx the mean_dist
        num_points = int(distance/mean_dist)
        # print(num_points)
        if num_points == 0: 
            interpolated_y.append(y0)
            interpolated_x.append(x0)
        else:
            interpolated_x.append(x0)
            interpolated_y.append(y0)
            for t in np.linspace(0, 1, num_points + 2)[1:-1]:
                xt = (1 - t) * x0 + t * x1
                yt = (1 - t) * y0 + t * y1
                interpolated_x.append(xt)
                interpolated_y.append(yt)
            # interpolated_x.append(x1)
            # interpolated_y.append(y1)

    return np.array(interpolated_x), np.array(interpolated_y)