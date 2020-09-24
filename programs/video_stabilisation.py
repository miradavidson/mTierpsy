#!/usr/bin/env python

""" Video stabilisation using OpenCV """

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from mTierpsy.programs.box_utils import box_to_df
from mTierpsy.programs.data_extraction import extract_data
from mTierpsy.programs.smoothing import moving_average


def make_mask(video, mask=None, show=True):
    """ Returns image and mask based on video """
    
    cap, n_frames, prev_gray = read_video(video)

    # make mask if not defined and return so user can play with it
    if mask is None:
        mask = np.zeros(prev_gray.shape, np.uint8)

    # show mask
    if show:
        plt.imshow(prev_gray)
        plt.imshow(mask, alpha=0.5)
        plt.show()
    
    return prev_gray, mask


def read_video(video):

    # Read input video
    cap = cv2.VideoCapture(video)

    # Get frame count
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Read first frame
    _, prev = cap.read() 

    # Convert frame to grayscale
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY) 

    return cap, n_frames, prev_gray


def find_points(cap, n_frames, prev_gray, mask=None, maxCorners=200, qualityLevel=0.1, minDistance=20, show=False):
    
    # Detect feature points in previous frame
    prev_pts = cv2.goodFeaturesToTrack(prev_gray,
                                     maxCorners=maxCorners,
                                     qualityLevel=qualityLevel,
                                     minDistance=minDistance,
                                     mask=mask)
    corners = np.int0(prev_pts)
    demo = prev_gray.copy()

    for i in corners:
        x,y = i.ravel()
        circ = cv2.circle(demo,(x,y),10,0,-1)

    if show: plt.imshow(demo),plt.show()
    
    return prev_pts


def calc_transform(cap, n_frames, prev_gray, prev_pts):
    print('Calculating transformations . . .')

    # Pre-define transformation-store array
    transforms = np.zeros((n_frames, 3), np.float32) 

    for i in tqdm(range(1,n_frames+1)):

        # Read next frame
        success, curr = cap.read()
        if not success:
            break

        # Convert to grayscale
        curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY) 

        # Calculate optical flow (i.e. track feature points)
        curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)

        # Sanity check
        assert prev_pts.shape == curr_pts.shape 

        # Filter only valid points
        idx = np.where(status==1)[0]
        prev_pts = prev_pts[idx]
        curr_pts = curr_pts[idx]

        #Find transformation matrix
        m = cv2.estimateAffinePartial2D(prev_pts, curr_pts)[0]

        # Extract translation and rotation angle
        dx = m[0,2]
        dy = m[1,2]
        da = np.arctan2(m[1,0], m[0,0])

        # Store transformation
        transforms[i] = [dx,dy,da]

    return transforms


def smoothen_transforms(transforms, radius=100):
    # smooth transforms based on moving average
    smooth_transforms = np.empty_like(transforms)
    
    smooth_transforms[:,0] = moving_average(transforms[:,0], radius)
    smooth_transforms[:,1] = moving_average(transforms[:,1], radius)
    smooth_transforms[:,2] = moving_average(transforms[:,2], radius)
    
    return smooth_transforms


def check_stabilisation(transforms):
    
    # plot translations and rotations
    plt.plot(transforms[:,0])
    plt.plot(transforms[:,1])
    plt.plot(transforms[:,2])
    plt.legend(['dx', 'dy', 'da'])
    plt.show()


def before_and_after(cap, n_frames, transforms):
    # Get width and height of video stream
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) 
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Reset stream to first frame 
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0) 

    # Write n_frames-1 transformed frames
    for i in tqdm(range(0,n_frames+1)):

        # Read next frame
        success, frame = cap.read()
        if not success:
            break
            
        # Extract transformations from the new transformation array
        m = make_transformation_matrix(transforms[i,:])

        # Apply affine wrapping to the given frame
        frame_stabilized = cv2.warpAffine(frame, m, (w,h))

        # Show video
        frame_out = cv2.hconcat([frame, frame_stabilized])
        cv2.imshow("Before and After", frame_out)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            return


def make_transformation_matrix(dxya):

    # Extract transformations from the transformation array
    dx=-dxya[0]
    dy=-dxya[1]
    da=-dxya[2]

    # Reconstruct transformation matrix accordingly to new values
    m = np.zeros((2,3), np.float32)
    m[0,0] = np.cos(da)
    m[0,1] = -np.sin(da)
    m[1,0] = np.sin(da)
    m[1,1] = np.cos(da)
    m[0,2] = dx
    m[1,2] = dy
    
    return m


def stabilise_videos(metadata, mask=None, checkpoints=True):
    """ Calculates the transformation at every frame for all videos in metadata """

    # read in video paths
    metadata = pd.read_csv(metadata)
    paths = metadata.original_video_path.unique()
    
    # store failed videos
    failed = []

    for video in paths:

        try:

            cap, n_frames, prev_gray = read_video(video)
            prev_pts = find_points(cap, n_frames, prev_gray, mask=mask, show=checkpoints)
            transforms = calc_transform(cap, n_frames, prev_gray, prev_pts)
            transforms = smoothen_transforms(transforms)

            if checkpoints:

                check_stabilisation(transforms)

                # show video
                show_video = input('Check before and after? (y/n) ')
                if show_video == 'y': before_and_after(cap, n_frames, transforms)

                # ask to save transformation
                save_transformation = input('Save transformation? (y/n) ')

            else:  # always save the transformation
                save_transformation = 'y'

            if save_transformation == 'y':
                output = video[:-4] + '_transformation.csv'
                transforms = pd.DataFrame(transforms, columns=['dx', 'dy', 'da'])
                transforms.to_csv(output, index=False)
            else:
                failed.append(video)
        except Exception as e:
            print(e) 
            failed.append(video)

    # return failed videos
    if len(failed) > 0:
        print('%s video(s) failed:\n' % len(failed))
        for video in failed:
            print(video+'\n')


def stabilise(entry, transforms):
    """
    Stabilisation of labels based on transformation dataframe
    TODO: speed up
    """

    # we're storing the results here
    transformed_df = pd.DataFrame().reindex_like(entry)

    # make numpy compatible
    transforms = transforms.values

    # read in columns with coordinates
    columns = list(entry.columns.get_level_values(0).unique())
    xy_columns = [c for c in columns if 'x' in entry[c].columns and 'y' in entry[c].columns]
    other_columns = [c for c in columns if c not in xy_columns]

    # here we go
    for column in xy_columns:

        # read in x, y coordinates
        xy = entry[column][['x', 'y']]

        # based on T = M * [x, y, 1].T
        # see https://docs.opencv.org/2.4/doc/tutorials/imgproc/imgtrans/warp_affine/warp_affine.html
        xy = np.c_[xy, np.ones(xy.shape[0])]

        # store transformed xy coordinates here
        xy_t = []

        # for each frame calculate the transformation matrix
        for j, frame in enumerate(xy):

            m = make_transformation_matrix(transforms[j,:])
            
            # apply transformation
            frame_t = np.dot(frame, m.T)
            xy_t.append(frame_t)

        # write new dataframe and add as column
        xy_t = pd.DataFrame(xy_t, columns=['x', 'y'])
        
        # add other subcolumns if exist
        subcolumns = [sub for sub in entry[column].columns if (sub != 'x') and (sub != 'y')]
        for subcolumn in subcolumns:
            xy_t[subcolumn] = entry[column][subcolumn]

        transformed_df[column] = xy_t

    # add other columns
    for column in other_columns:
        transformed_df[column] = entry[column]

    return transformed_df


def transform_labels(metadata_path):
    """ Read labels, transform, and update metadata """

    # read labels
    metadata = pd.read_csv(metadata_path)

    # set filename of transformed labels
    change_filename = lambda x: x[:-4] + '_transformed.csv'

    print('\nTransforming labels and adding to metadata . . .')
    for i, entry in tqdm(metadata.iterrows(), total=len(metadata)):

        # read labels
        labels = pd.read_csv(entry.labels_path, header=[0,1,2])
        header = labels.columns.get_level_values(0) # we'll need this later
        labels.columns = labels.columns.droplevel()
        box_labels = pd.read_csv(entry.box_labels_path, header=[0,1], index_col=0)
        box_labels = box_to_df(box_labels, labels)
        
        # read in transformation matrix
        transforms_path = entry.original_video_path[:-4] + '_transformation.csv'
        transforms_df = pd.read_csv(transforms_path)[['dx', 'dy','da']]

        # stabilise labels
        assert len(labels) == len(transforms_df)  # sanity check
        transformed_labels = stabilise(labels, transforms_df)
        transformed_box_labels = stabilise(box_labels, transforms_df)

        # write to file
        c1, c2 = zip(*transformed_labels.columns)  # extract second and third columns
        transformed_labels.columns = pd.MultiIndex.from_tuples(zip(header, c1, c2))
        transformed_labels.to_csv(change_filename(entry.labels_path), index=None)
        transformed_box_labels.to_csv(change_filename(entry.box_labels_path))

    # update metadata
    metadata['labels_path'] = metadata['labels_path'].apply(change_filename)
    metadata['box_labels_path'] = metadata['box_labels_path'].apply(change_filename)
    metadata.to_csv(metadata_path, index=None)
