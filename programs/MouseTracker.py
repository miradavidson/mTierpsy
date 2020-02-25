#!/usr/bin/env python

"""
Pose estimation and feature extraction for analysis of mouse behaviour
"""

import os
from shutil import copyfile

from mousetracker.programs.box_utils import extract_images, add_box_to_metadata
from mousetracker.programs.label_videos import label_videos
from mousetracker.programs.calibration import calibrate
from mousetracker.programs.video_stabilisation import stabilise_videos, transform_labels
from mousetracker.programs.feature_extraction import feature_extraction


class MouseTracker():

	def __init__(self, path):
		self.path = path
		self.metadata = os.path.join(path, 'metadata.csv')
		self.box = os.path.join(path, 'box')
		self.calibration = os.path.join(path, 'calibration.csv')
		self.features = os.path.join(path, 'features.csv')


	def create_new_project(self, metadata):
		""" Create a new project and copy over metadata

		Params
		------
		metadata (str) Path to metadata, will be copied to the path
		"""

		# check if metadata exists
		if not os.path.exists(metadata):
			raise OSError('Metadata file cannot be found.')

		if not os.path.exists(self.path):
			os.makedirs(self.box)
			copyfile(metadata, self.metadata)
			print('Project created in: %s' % os.path.join(os.getcwd(), self.path))
		else:
			raise OSError('Project directory already exists.')


	def extract_images(self):
		""" Extract images from videos for labelling of box """
		extract_images(self.metadata, output_dir=self.box)


	def add_box_to_metadata(self):
		""" Add path to labelled corners to metadata """
		add_box_to_metadata(self.metadata, input_dir=self.box)


	def calibrate(self, box_size, output=None):
		""" Create calibration file based on box size and videos """
		if not output: output = self.calibration 
		calibrate(self.metadata, box_size, output)


	def label_videos(self, config, **kwargs):
		""" Label videos using DeepLabCut """
		label_videos(self.metadata, config, **kwargs)


	def extract_features(self, calibration=None, output=None, **kwargs):
		""" Feature extraction """
		if not calibration: calibration = self.calibration
		output = self.features if not output else os.path.join(self.path, output)
		feature_extraction(self.metadata, calibration, output, **kwargs)


	def stabilise_videos(self, mask=None):
		""" Run the interactive video stabilisation module
		mask (Numpy array): Cover the first frame with a mask
		to prevent tracking of moving objects
		"""
		stabilise_videos(self.metadata, mask=mask)


	def transform_labels(self):
		""" Apply transformation to labels and update metadata """
		transform_labels(self.metadata)
