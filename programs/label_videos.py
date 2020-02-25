#!/usr/bin/env python

""" Pose estimation using DeepLabCut """

import os
import pandas as pd
import deeplabcut


def find_labels(video_path):
    """ Finds labels in CSV format based on the filename of the video 
    Warning: assumes there are no other CSV files in the directory """
    
    # get upper directory and video file
    upper_dir, video = video_path.rsplit('/', 1)
    
    # remove extension (.mp4)
    filename = video[:-4]
    
    # find CSV file that matches the filename
    csv = [upper_dir + '/' + file for file in os.listdir(upper_dir) if file.startswith(filename) and file.endswith('.csv')][0]
    
    return csv
 

def label_videos(metadata_path, config, gputouse=0):
	"""Extracts labels in CSV format and generates labelled videos using DeepLabCut 
	
	Arguments
	---------
	metadata (str): Format on metadata can be found in README
	config (str): Filename should be 'config.yaml' and points to the path to the trained model (check 'project_path')

	"""

	# save path
	cur_dir = os.getcwd()
	
	# import data
	metadata = pd.read_csv(metadata_path)

	videos = list(metadata.video_path)

	# analyse videos to extract labels
	deeplabcut.analyze_videos(config, videos, gputouse = gputouse, shuffle = 1, save_as_csv = True)

	# to check correct labelling
	deeplabcut.create_labeled_video(config, videos)

	# go back to saved path
	os.chdir(cur_dir)

	# add path to labels to metadata
	metadata['labels_path'] = metadata['video_path'].apply(lambda x: find_labels(x))
	metadata.to_csv(metadata_path, index=False)

