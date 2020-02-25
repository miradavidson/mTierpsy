#!/usr/bin/env python

""" Functions for box labelling """

import os
import cv2
import pandas as pd


def extract_images(metadata, output_dir):
	""" Extracts the first image of all videos in the metadata """

	# read in metadata
	metadata = pd.read_csv(metadata)

	# Make output directory
	if not os.path.exists(output_dir):
		os.mkdir(output_dir)

	for index, exp in metadata.iterrows():
		
		# find video
		video = exp['video_path']

		# use openCV functions to read
		cap = cv2.VideoCapture(video)
		ret, frame = cap.read() 
		cv2.imwrite(os.path.join(output_dir,'%s.jpg') % index, frame)
		cap.release()


def order_corners(file):
	""" Orders the corners based on X and Y coordinates of annotated file in ImageJ """
	
	# read in file
	measurements = pd.read_csv(file)

	# remove unnecessary columns
	corners = measurements[['X', 'Y']]
	corners.columns = map(str.lower, corners.columns)
	
	# on the left
	left = corners.sort_values('x')[:2]
	top_left = left.loc[left['y'].idxmin()]  # lowest Y
	bottom_left = left.loc[left['y'].idxmax()]  # highest Y

	# on the right
	right = corners.sort_values('x')[2:]
	top_right = right.loc[right['y'].idxmin()]
	bottom_right = right.loc[right['y'].idxmax()]
	
	# order and return as new dataframe
	ordered_corners = pd.concat([top_left,top_right,bottom_left,bottom_right],axis=1)
	ordered_corners.columns = ['top_left','top_right','bottom_left','bottom_right']
	ordered_corners = ordered_corners.unstack().swaplevel(1).sort_index()
	ordered_corners = pd.DataFrame(ordered_corners).T
	ordered_corners.index.name = 'frame'
	
	# write to file
	ordered_corners.to_csv(file)

	return ordered_corners


def add_box_to_metadata(metadata_path, input_dir):
	""" Order corners and add to metadata """

	# read in metadata
	metadata = pd.read_csv(metadata_path)

	# order corners
	corners = []
	for i in range(len(metadata)):
		path = os.path.join(input_dir, '%s.csv' % i)
		order_corners(path)
		corners.append(os.path.join(os.getcwd(), path))

	# add to metadata
	metadata['box_labels_path'] = corners
	metadata.to_csv(metadata_path, index=False)


def box_to_df(box_labels, labels):
    """ Makes box into dataframe per frame for transformation """

    # repeat for each frame
    box_labels = box_labels.loc[box_labels.index.repeat(len(labels))]
    box_labels = box_labels.reset_index(drop=True)

    return box_labels
