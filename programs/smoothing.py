#!/usr/bin/env python

""" Smoothing of trajectories """

import numpy as np
import pandas as pd


def moving_average(curve, radius): 
    """ calculate moving average to smoothen curve """
    
    window_size = 2 * radius + 1
    
    # Define the filter 
    f = np.ones(window_size)/window_size 
    
    # Add padding to the boundaries 
    curve_pad = np.lib.pad(curve, (radius, radius), 'edge')
    
    # Apply convolution 
    curve_smoothed = np.convolve(curve_pad, f, mode='same') 
    
    # Remove padding 
    curve_smoothed = curve_smoothed[radius:-radius]
    
    # return smoothed curve
    return curve_smoothed


def interpolate_outliers(curve, avg_body_length, radius, remove=False):

    # make new object
    curve = np.array(curve)
    
    # smooth based on moving average
    smoothed_curve = moving_average(curve, radius)

    # identify outliers
    outliers = np.where(abs(curve - smoothed_curve) > 2*avg_body_length)[0]

    # interpolate values
    if remove:
        curve[outliers] = np.nan
    else:
        curve[outliers] = smoothed_curve[outliers]
    
    return curve


def smooth(labels, bodyparts, method='soft', cutoff=None):
    """ Smoothing of trajectories,
    method = 'hard': interpolates points that are beyond a cut-off of a smoothened trajectory
    method = 'soft': moving average of the trajectory
    """
    
    # initialise a new dataframe
    transformed_df = pd.DataFrame().reindex_like(labels)

    # here we go
    for bodypart in bodyparts:
        
        # read in x, y coordinates
        x = labels[bodypart]['x']
        y = labels[bodypart]['y']
        
        if method == 'hard' or method == 'both':
            
            # interpolate
            x = interpolate_outliers(x, cutoff, 25)
            y = interpolate_outliers(y, cutoff, 25)
        
        if method == 'soft' or method == 'both':
        
            # smoothen
            x = moving_average(x, 3)
            y = moving_average(y, 3)
        
        # write to dataframe
        new_labels = pd.DataFrame([x, y], index=['x', 'y']).T
        new_labels['likelihood'] = labels[bodypart]['likelihood']
        
        transformed_df[bodypart] = new_labels

    return transformed_df
