#!/usr/bin/env python

""" Calibration of pixels and frames """

import pandas as pd
import numpy as np
import cv2
import math
import argparse
import sys


def calculate_calibration(entry, box_size):
	""" Calculates the calibration based on the box size and fps of video"""

	# calculate calibration
	calibration = {
		'fps': calc_fps(entry.video_path),
		'pxmm': calc_pxmm(entry.box_labels_path, box_size)
	}

	return calibration


def calc_fps(video):
	""" Calc fps using openCV """

	video = cv2.VideoCapture(video)
	fps = round(video.get(cv2.CAP_PROP_FPS))

	return fps


def calc_pxmm(box, box_size):
	""" Calc pxmm by measuring lengths of sides of box """

	corners = pd.read_csv(box, header=[0,1], index_col=0)
	len_sides = sides_from_corners(corners)

	# average distance between 4 sides
	avg_distance = np.mean(len_sides)
	pxmm = avg_distance/box_size

	return pxmm


def calc_distance_between_points(p1, p2):
	""" Calculate distance between corners """

	dx = abs(p2.x - p1.x)
	dy = abs(p2.y - p1.y)
	dz = math.sqrt(dx**2 + dy**2)

	return dz


def sides_from_corners(corners):
	""" Extracts sides based on distances between corners """

	len_sides = [
		calc_distance_between_points(corners.top_right, corners.top_left),
		calc_distance_between_points(corners.top_right, corners.bottom_right),
		calc_distance_between_points(corners.bottom_right, corners.bottom_left),
		calc_distance_between_points(corners.bottom_left, corners.top_left)
	]

	return len_sides


def calibrate(metadata, box_size, output):
	"""
	Calculates average calibration across experiment
	"""

	# read in metadata
	metadata = pd.read_csv(metadata)

	experiments = {}

	for exp in metadata.experiment.unique():

		entries = metadata.loc[metadata.experiment == exp]

		calibrations = [calculate_calibration(row, box_size) for index, row in entries.iterrows()]

		calibrations = pd.DataFrame(calibrations)
	
		experiments[exp] = round(calibrations.mean(), 2)

	experiments = pd.DataFrame(experiments)
	experiments.to_csv(output)


if __name__ == '__main__':
	
	# Command-line interface
	parser = argparse.ArgumentParser(description='Calculates calibration per experiment.')
	parser.add_argument('metadata', type=str, action='store',
						help='Path to metadata file.')
	parser.add_argument('box_size', type=int, action='store',
						help='Size of box in mm.')
	parser.add_argument('-o', '--output', type=str, default=sys.stdout,
						help='Path to output file.')
	args = parser.parse_args()

	# Run calibration
	calibration(args.metadata, args.box_size, args.output)
