#!/usr/bin/env python

""" Data extraction """

import pandas as pd
from tqdm import tqdm
from mousetracker.programs.box_utils import box_to_df

def extract_data(metadata, start=0, end=None):
    """ Read in datasets """

    labels = []
    box_labels = []

    print('\nReading labels . . .')
    for i, exp in tqdm(metadata.iterrows(), total=len(metadata)):
        
        # read in labels
        mouse = pd.read_csv(exp.labels_path, header=[1,2])
        box = pd.read_csv(exp.box_labels_path, header=[0,1], index_col=0)
        
        # unncessary if stabilisation has happened
        # need same shape for matrix calculations
        if len(mouse) != len(box):
            box = box_to_df(box, mouse)

        # set to range of frames
        mouse = mouse[start:end].reset_index(drop=True)
        box = box[start:end].reset_index(drop=True)

        labels.append(mouse)
        box_labels.append(box)

    return labels, box_labels
