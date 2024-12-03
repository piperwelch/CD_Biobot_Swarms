from glob import glob
import numpy as np
import csv
import os
import pickle

def compute_mean_curvature(coordinates):
    "input is the bot's trajectory - a 2D array of x,y coordinates"
    assert(len(coordinates[:,0])==len(coordinates[:,1]))
    
    # https://www.delftstack.com/howto/numpy/curvature-formula-numpy/
    x_coordinates = coordinates[:,0]
    y_coordinates = coordinates[:,1]

    # Calculate velocity of curve

    # compute gradient at each point
    x_t = np.gradient(x_coordinates) # gradient in x direction
    y_t = np.gradient(y_coordinates) # gradient in y direction

    # Zipping the x,y gradients back together
    vel = np.array([ [x_t[i], y_t[i]] for i in range(x_t.size)])

    speed = np.sqrt(x_t * x_t + y_t * y_t)

    tangent = np.array([1/speed] * 2).transpose() * vel

    ss_t = np.gradient(speed)
    xx_t = np.gradient(x_t)
    yy_t = np.gradient(y_t)

    curvature_val = np.abs(xx_t * y_t - x_t * yy_t) / (x_t * x_t + y_t * y_t)**1.5

    return np.mean(curvature_val)

def discard_duplicate_points(coordinates):
    prev_x = -10000
    prev_y = -10000

    unique_x = []
    unique_y = []

    for i in range(1,len(coordinates)):
        
        if coordinates[i,0] == prev_x and coordinates[i,1] == prev_y:
            pass
        else:
            unique_x.append(coordinates[i,0])
            unique_y.append(coordinates[i,1])

            prev_x = coordinates[i,0]
            prev_y = coordinates[i,1]
    
    new_coordinates = np.concatenate((np.asarray(unique_x).reshape(-1,1), np.asarray(unique_y).reshape(-1,1)), axis=1)
    
    return new_coordinates
