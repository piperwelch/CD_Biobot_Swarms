#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 09:59:14 2024

@author: thosvarley
"""

import os
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean, pdist, squareform
from scipy.stats import zscore
from copy import deepcopy 
np.random.seed(0)

#%% LIBRARY

# UNIVARIATE 

def get_accelerations(df):

    xy = df[["x","y"]].values
    # print(xy[0])
    # quit()
    velocities = [euclidean(xy[i], xy[i+1]) for i in range(xy.shape[0]-1)]
    return np.diff(velocities)


def get_angles(df):
    xy = df[["x","y"]].values * 1
    
    angles = np.zeros(xy.shape[0]-2)
    for i in range(xy.shape[0]-2):
        
        past = xy[i]
        present = xy[i+1]
        future = xy[i+2]
        
        # law of cosines
        
        a = euclidean(past, present)
        b = euclidean(present, future)
        c = euclidean(past, future)

        angle = np.pi - np.arccos(((a**2) + (b**2) - (c**2)) / (2*a*b))
        angles[i] = angle
        
    return angles


def get_tortuosity(df):
    
    xy = df[["x","y"]].values 

    integral = np.trapz([euclidean(xy[i,:], xy[i+1,:]) for i in range(xy.shape[0]-1)])
    final_distance = euclidean(xy[0,:], xy[-1,:])
    
    return integral / final_distance

    
# MULTIVARIATE 


def get_edge_series(X, accels, norm=False):
    
    edges = []
    yticks = []
    
    for i in range(len(accels)):
        for j in range(i):
            if norm == False:    
                edges.append(X[i]*X[j])
            else:
                edges.append(
                    zscore(X[i]) * zscore(X[j])
                    )
            yticks.append("({0}, {1})".format(i+1, j+1))
            
    return np.array(edges), yticks 

    

def total_correlation(X):    
    cov = np.cov(
        zscore(X, axis=-1), 
        ddof=0.0)
    tc = -np.linalg.slogdet(cov)[1]/2
    
    return tc


# NULL

def circular_shift_null(X):
    """
    Produces a null dataset that preserves first-level autocorrelation by 
    by applying an independent circular shift to each channel. 
    
    Call this function many times to build a null distrbution for 
    NHSTing. 
    """
    X_copy = deepcopy(X)
    
    for i in range(len(X)):
        shift = np.random.randint(-X_copy[i].shape[0]//2,
                                  X_copy[i].shape[0]//2)
        X_copy[i] = np.roll(X[i], 
                            shift)
    
    return X_copy


#%% EXAMPLE SCRIPT 

in_dir = "swarm_data\Swarm_C\\"

listdir = os.listdir(in_dir)

trial_1 = pd.read_csv(in_dir + 'SwarmC_Trial1\\SwarmC_Trial1_correct_order.csv')

tracks = [
    trial_1[trial_1["track"] == i] for i in range(1,5)
    ]

for track in tracks:
    track.set_index("frame", inplace=True)

idxs = [set(track.index) for track in tracks]
intersection = list(set.intersection(*idxs))


accels = [get_accelerations(df.loc[intersection]) for df in tracks]
angles = [get_angles(df.loc[intersection]) for df in tracks]

tc_angles = total_correlation(angles)

null = []

for i in range(10_000):
    null.append(total_correlation(circular_shift_null(angles)))
# print(sum(null)/len(null))
# P-value of 0.028. Significant (barely)


#%%