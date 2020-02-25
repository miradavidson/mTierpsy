#!/usr/bin/env python

""" Feature extraction based on metadata (verbose) """

import pandas as pd
from tqdm import tqdm

from mousetracker.programs.data_extraction import extract_data
from mousetracker.programs.calibration import calculate_calibration
from mousetracker.programs.feature_computation import Mouse


def feature_extraction(metadata, calibration, output, start=0, end=None):
	""" Runs the feature extraction based on pose estimation of mouse """

	metadata = pd.read_csv(metadata)
	calibration = pd.read_csv(calibration, index_col=0).to_dict()

	results = []

	# run each experiment separately
	for exp in metadata.experiment.unique():

		# extract calibration for this experiment
		cal = calibration[exp]

		# subdivide entries
		entries = metadata.loc[metadata.experiment == exp]
		print('\n%s (%s entries)' % (exp, len(entries)))

		# read data
		labels, box_labels = extract_data(entries, start, end)

		# feature extraction
		print('\nFeature extraction . . .')
		for id, mouse, box in tqdm(zip(entries['file_id'], labels, box_labels), total=len(entries)):
			result = Mouse(id, mouse, box, cal).run()
			results.append(result)

	# write all to one file
	results = pd.concat(results).sort_index()
	results.index.name = 'file_id'
	results.to_csv(output)

	print('\nResults saved to %s' % output)
