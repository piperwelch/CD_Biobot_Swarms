from glob import glob
import numpy as np
import csv
import pandas as pd
import os
import pickle
from scipy.stats import circvar

def compute_heading(trajectory):
    # computes headings from x and y coordinates of the trajectory
    # heading is computed between pairs of points and therefore cannot be compute for the first and last points 
    x_pts = np.asarray(trajectory["x"])
    y_pts = np.asarray(trajectory["y"])

    headings = []
    for i in range(1,len(x_pts)):
        x_diff = x_pts[i]-x_pts[i-1]
        y_diff = y_pts[i]-y_pts[i-1]
        heading = np.arctan2(y_diff,x_diff)
        headings.append(heading)
    
    return headings

def compute_straightness_index(trajectory):
    # Computes straightness index from headings (circular variance of headings)

    headings = compute_heading(trajectory)
    
    # compute circular variance of the headings (between 0-2pi)
    circvar = circvar(headings,low=-np.pi, high=np.pi)

    return 1 - circvar

