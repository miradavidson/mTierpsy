#!/usr/bin/env python

""" Feature computation and transformation """

import pandas as pd 
import numpy as np

from mTierpsy.programs.smoothing import smooth, moving_average


class Mouse:
    """ A mouse object with all feature computations """

    bodyparts = ['Center', 'Nose', 'Tailbase', 'Tailtip']

    thresholds = {
        'center/corner': 60, # (mm)
        'motion': (1, 3),  # stopping, walking, running (mm/s)
        #'rearing': (60, 40),  # distance, length (mm, mm)
    }

    # smoothing labels, options='soft'/'hard'/'both'/False
    smoothing_factor = 'both'


    def __init__(self, id, labels, box, calibration):

        self.id = id
        self.labels = labels
        self.box = box
        self.calibration = calibration

        # we store the final features here
        self.primary_features = {}
        self.event_features = {}
        self.expanded_features = {}
        self.features = {}

    ##########################
    ### curvature measures ###
    ##########################


    def calculate_angles(self, bodyparts):
        """ Calculates the angle between all bodyparts in labels """

        # initalise dictionary to store all angle combinations
        angles = {}
        
        # iterate over all bodyparts
        for b1 in bodyparts:
            
            # save all angles associated to this bodypart
            bodypart_angles = {}
            
            # for every combination
            for b2 in bodyparts:
                # calculate angle
                angle = np.arctan2(self.labels[b2].y-self.labels[b1].y, self.labels[b2].x-self.labels[b1].x)
                # if bigger than full circle
                angle = angle - 2*np.pi*np.floor((angle+np.pi)/(2*np.pi))
                bodypart_angles[b2] = angle
            
            # and save
            angles[b1] = bodypart_angles
        
        # reformat for multiindex dataframe
        angles = {(outerKey, innerKey): values
                  for outerKey, innerDict in angles.items()
                  for innerKey, values in innerDict.items()}
        
        # save as dataframe
        angles = pd.DataFrame(angles)
        
        return angles


    def calculate_body_angle(self):
        """ Calculates body angle based on angle difference of
        center to tail and nose to center """
        
        angles = self.calculate_angles(['Nose', 'Center', 'Tailbase'])
        phi = angles.Center.Tailbase - angles.Nose.Center
        return phi


    def calculate_tail_angle(self):
        """ Calculates tail angle based on angle difference of
        nose to tailbase and tailbase to tailtip """
        
        angles = self.calculate_angles(['Nose', 'Tailbase', 'Tailtip'])
        phi = angles.Tailbase.Tailtip - angles.Nose.Tailbase
        return phi


    def calculate_kappa(self, labels, frame):
        """ Calculate phi from three points """
        
        # based on theta = tan(dy/dx)
        dy1 = labels['y%s' % int(frame/2)]-labels['y0']
        dx1 = labels['x%s' % int(frame/2)]-labels['x0']
        theta1 = np.arctan2(dy1, dx1)
        
        # need this to calculate angle
        dy2 = labels['y%s' % frame]-labels['y%s' % int(frame/2)]
        dx2 = labels['x%s' % frame]-labels['x%s' % int(frame/2)]
        theta2 = np.arctan2(dy2, dx2)
        
        angle = theta2 - theta1
        # if bigger than full circle
        angle = angle - 2*np.pi*np.floor((angle+np.pi)/(2*np.pi))   
        
        # k = dT/ds
        # integrate over all distances to get ds
        ds = 0
        for i in range(frame):
            # distance between 2 points
            d = np.sqrt((labels['y%s'%(i+1)]-labels['y%s'%i])**2 +(labels['x%s'%(i+1)]-labels['x%s'%i])**2)
            ds+=d
        
        kappa = angle/ds
        
        return kappa


    def calculate_curvature(self, labels, frame=2):
        """ 
        Calculates path curvature over time
        frame: frame to which you want to compare the current angle (2 for next angle)
        """
        
        # make dataframes to compare two angles between three points
        labels = labels[['x', 'y']]
        
        # all different time frames
        timeseries = []
        
        for i in range(frame+1):
            end = i-frame if i != frame else len(labels)
            timeseries.append(labels[i:end].reset_index(drop=True))
       
        # concat time frames for matrix functions
        p = pd.concat(timeseries, axis=1)
        
        # rename columns
        p.columns = ['%s%s' % (j, i) for i in range(frame+1) for j in ['x', 'y']]

        # calculate kappa
        kappa = self.calculate_kappa(p, frame)

        # reset index based on frame
        kappa.index += frame/2
        
        return kappa


    def calculate_angular_velocity(self, labels, per_frames=5):
        
        # calculate curvature
        k = np.array(self.calculate_curvature(labels))
        
        # calculate angular velocity
        av = np.gradient(k, per_frames)
        
        return av


    def calculate_relative_angular_velocity_body(self, per_frames=5):
        
        # calculate angle
        a = self.calculate_body_angle()

        # calculate angular velocity
        av = np.gradient(a, per_frames)
        
        return av


    def calculate_relative_angular_velocity_tail(self, per_frames=5):
        
        # calculate angle
        a = self.calculate_tail_angle()
        
        # calculate angular velocity
        av = np.gradient(a, per_frames)
        
        return av


    ##########################
    ### distance measures ####
    ##########################


    def calculate_jerk(self, labels, per_frames = 5, smooth=3):
        """Calculates the third derivative of d over t"""
        
        # calculate acceleration
        a = self.calculate_acceleration(labels, smooth=smooth)
        
        # calculate jerk
        j = np.gradient(a, per_frames)
        
        if smooth:
            j = moving_average(j, smooth)
        
        return j


    def calculate_acceleration(self, labels, per_frames = 5, smooth=3):
        """Calculates the second derivative of d over t"""
        
        # calculate speed
        v = self.calculate_speed(labels, smooth=smooth)

        # calculate acceleration
        a = np.gradient(v, per_frames)
        
        if smooth:
            a = moving_average(a, smooth)

        return a
        

    def calculate_speed(self, labels, per_frames = 5, smooth=3):
        """Calculates the first derivative of d over t"""

        # calculate distance travelled
        d = self.calculate_distance(labels)

        # calculate speed
        v = abs(np.gradient(d, per_frames))  # we don't take into account backward motion
 
        # if smoothed
        if smooth:
            v = moving_average(v, smooth)

        return v


    def calculate_distance(self, labels, per_frames = 1):
        """Calculates distance over t"""

        # calculate distance travelled at each time point
        dx = np.array(labels.x[1::per_frames]) - np.array(labels.x[:-1:per_frames])
        dy = np.array(labels.y[1::per_frames]) - np.array(labels.y[:-1:per_frames])
        dz = np.sqrt((dx**2 + dy ** 2))

        return dz


    def calculate_total_distance(self, labels, per_frames = 1):
        """Calculates the total distance travelled over t"""
        
        # calculate distance
        dz = self.calculate_distance(labels, per_frames)
        
        # calculate total distance travelled
        total_dz = np.nansum(dz)

        return total_dz


    ############################
    ### relationship to wall ###
    ############################


    def calculate_distance_point_to_line(self, p0, p1, p2):
        """Calculates distance of a point, p0, to a line between p1 and p2 """

        # based on h = 2A/b
        double_area = abs((p2.y-p1.y)*p0.x - (p2.x-p1.x)*p0.y + p2.x*p1.y - p2.y*p1.x)
        base = np.sqrt((p1.y-p2.y)**2+(p1.x-p2.x)**2)
        height = double_area/base
        
        return height
        

    def calculate_distance_to_walls(self, labels):
        """Reads in box size and calculates distance to each wall"""

        # calculate distance to each wall
        top = self.calculate_distance_point_to_line(labels, self.box['top_left'], self.box['top_right'])
        bottom = self.calculate_distance_point_to_line(labels, self.box['bottom_left'], self.box['bottom_right'])
        left = self.calculate_distance_point_to_line(labels, self.box['top_left'], self.box['bottom_left'])
        right = self.calculate_distance_point_to_line(labels, self.box['top_right'], self.box['bottom_right'])

        # add to single dataframe
        distances = pd.concat([top, bottom, left, right], axis=1)

        distances.columns = ['top', 'bottom', 'left', 'right']

        return distances


    def calculate_distance_to_closest_wall(self, labels):
        """Calculates distance to the closest wall"""

        # calculate distances to each wall
        distances = self.calculate_distance_to_walls(labels)

        # take distance to closest wall
        closest = distances.min(axis=1)
        
        # tracking errors that go beyond wall
        closest[closest < 0] = 0

        return closest


    def calculate_distance_to_corner(self, labels):
        """Calculates distance to closest corner
        TODO: accidentally made this heavier than it needs to  be,
        could just directly compare to each corner
        """
        
        # calculate distances to each wall
        distances = self.calculate_distance_to_walls(labels)

        # looks heavy but faster
        a = distances.values
        smallest = a[np.arange(len(distances))[:,None],np.argpartition(-a,np.arange(2),axis=1)[:,-2:]]

        # distance to corner
        distance_to_corner = np.sqrt(smallest[:,0]**2 + smallest[:,1]**2)

        return distance_to_corner


    def calculate_distance_from_center(labels, box):
        """ Computes center of box and finds distance to this at each t """
        
        # find center of box
        centroid = (sum(box.loc['X']) / len(box), sum(box.loc['Y']) / len(box))

        # calculate distance between each point and the center
        dx = centroid[0] - np.array(labels.x)
        dy = centroid[1] - np.array(labels.y)
        dz = np.sqrt((dx**2 + dy ** 2))
        
        return dz
        


    ###########################
    ### multiple body parts ###
    ###########################

    def calculate_body_length(self, bodyparts=['Nose','Center','Tailbase']):
        """ Calculates the length of the body by two points """

        # initialise total length
        length = 0
        
        # calculate distance between two bodyparts
        for i in range(len(bodyparts)-1):
            dx = np.array(self.labels[bodyparts[i]].x) - np.array(self.labels[bodyparts[i+1]].x)
            dy = np.array(self.labels[bodyparts[i]].y) - np.array(self.labels[bodyparts[i+1]].y)
            dz = np.sqrt((dx**2 + dy ** 2))
            length += dz
        
        return length


    ######################
    ### grid measures ####
    ######################

    def make_grid(self, points):
        
        # create grid
        
        # make square from box
        width = self.box.loc['X'].max() - self.box.loc['X'].min()
        height = self.box.loc['Y'].max() - self.box.loc['Y'].min()
        
        # how big each grid point is going to be
        point_width = width/np.sqrt(points)
        point_height = height/np.sqrt(points)

        # where we store all grid points
        grid_points = []
        
        # iterate over width and height
        w = 0
        h = 0
        
        while w < width:
        
            curr_w = (w,w + point_width)
            w += point_width
            
            while h < height:
                
                curr_h = (h, h + point_height)
                h += point_height
                grid_points.append((curr_w, curr_h))
            
            h = 0
          
        return grid_points
      
        
    def find_gridpoint(self, coordinate, grid):
        """
        Goes over each grid point and checks whether point in that grid point
        TODO: speed this up
        """

        for i, point in enumerate(grid):
            x = point[0]
            y = point[1]
            if coordinate.x > x[0] and coordinate.x < x[1]:
                if coordinate.y > y[0] and coordinate.y < y[1]:
                    return i


    def grid_over_time(self, labels, points):
        """ Returns a time series of occupied grid point number """

        # make the grid
        grid = self.make_grid(points)

        # find gridpoint for each timepoint
        occupancy = labels.apply(lambda x: self.find_gridpoint(x, grid), axis=1)

        return occupancy


    def calculate_dwelling(self, labels, points=25):
        """ Calculates average time spent on each point """
        
        # get time series
        time_series = self.grid_over_time(labels, points)
        
        times = []
        # iteration starts here
        seen = time_series[0]
        count = 1
        for i in time_series[1:]:
            if i == seen:
                count +=1
            else:
                times.append(count)
                seen = i
                count = 1
        
        return sum(times)/len(times)
        
    #####################
    ### motion states ###
    #####################


    def calculate_motion_states(self):
        

        # motion states
        stopping = self.primary_features['speed_Center'] < self.thresholds['motion'][0]
        walking = (self.primary_features['speed_Center'] > self.thresholds['motion'][0]) & \
                  (self.primary_features['speed_Center'] < self.thresholds['motion'][1])
        running = self.primary_features['speed_Center'] > self.thresholds['motion'][1]

        return stopping, walking, running
        

    #############################
    ######## other states #######
    #############################


    def calculate_rearing(self):
        """ Calculates rearing frequency based on the assumption that
        a mouse is close to the wall and points are close to each other"""
        
        # get thresholds
        d_threshold, l_threshold = self.thresholds['rearing']

        if_close = self.primary_features['distance_to_wall_Center'] < d_threshold
        if_small = self.primary_features['length'] < l_threshold
        rearing = if_close | if_small
       
        return rearing


    def calculate_in_corner(self):
        """Calculates time spent in corner based on a distance threshold"""
        
        # calculate distances to each wall
        distances = self.calculate_distance_to_walls(self.labels.Center).apply(self.calibrate('distance'))
        
        # find instances where close to a wall
        close = distances < self.thresholds['center/corner']
        time_close = close.sum(axis=1)
        in_corner = time_close == 2  # 2 is close to two walls aka a corner
        
        return in_corner


    def calculate_in_center(self):
        """Calculates time spent in corner based on a distance threshold"""
        
        # calculate distances to each wall
        distances = self.calculate_distance_to_walls(self.labels.Center).apply(self.calibrate('distance'))
        
        # find instances where far from a wall
        close = distances < self.thresholds['center/corner']
        time_close = close.sum(axis=1)
        in_center = time_close == 0  # not close to any wall
        
        return in_center
     
        
    def calculate_small_body(self):
        """ Calculates when body length smaller than half the body length """

        if_small = self.length < np.nanpercentile(self.length, 90)/2  # 90 to avoid outliers

        return if_small


    def calculate_large_body(self):
        """ Calculates when body length larger than half the body length """

        if_large = self.length > np.nanpercentile(self.length, 90)/2

        return if_large


        
    ##############
    ### events ###
    ##############


    def calculate_frequency(self, labels):
        """ Calculate how often event occurs """
        
        # initialisation
        n_events = 0
        started = False
        
        for i in labels:
            if i:
                if not started:
                    started = True
                    n_events += 1
            else:
                started = False
        
        # normalise for video length
        frequency = n_events / len(labels)
                    
        return frequency


    def calculate_fraction(self, labels):
        """ Calculate fraction of time spent in event """
        return sum(labels) / len(labels)


    def calculate_mean_duration(self, labels):
        """ Calculate mean duration of event """
        
        # initialisation
        durations = []
        count = 0
        
        for i in labels:
            if i:
                count += 1
            else:
                durations.append(count)
                count = 0
        
        # remove when event is not happening
        durations = [x for x in durations if x != 0]

        # return the mean
        if len(durations) != 0:
            return np.mean(durations) / self.calibration['fps']  # convert to seconds
        else:
            return None
 

    ##########################
    ######## results #########
    ##########################


    def calculate_primary_features(self):
        """ Calculates primary "time series" features """

        # body part features
        for b in self.bodyparts:
            self.primary_features['speed_' + b] = pd.Series(self.calculate_speed(self.labels[b])).apply(self.calibrate('speed'))
            self.primary_features['acceleration_' + b] = pd.Series(self.calculate_acceleration(self.labels[b])).apply(self.calibrate('acceleration'))
            self.primary_features['jerk_' + b] = pd.Series(self.calculate_jerk(self.labels[b])).apply(self.calibrate('jerk'))
            self.primary_features['distance_to_wall_' + b] = pd.Series(self.calculate_distance_to_closest_wall(self.labels[b])).apply(self.calibrate('distance'))
            self.primary_features['distance_to_corner_' + b] = pd.Series(self.calculate_distance_to_corner(self.labels[b])).apply(self.calibrate('distance'))
            self.primary_features['curvature5_' + b] = pd.Series(self.calculate_curvature(self.labels[b], frame=10)).apply(self.calibrate('curvature'))
            self.primary_features['curvature25_' + b] = pd.Series(self.calculate_curvature(self.labels[b], frame=50)).apply(self.calibrate('curvature'))
            self.primary_features['angular_velocity_' + b] = pd.Series(self.calculate_angular_velocity(self.labels[b])).apply(self.calibrate('angular_velocity'))

        # overall features
        self.primary_features['body_angle'] = pd.Series(self.calculate_body_angle())   
        self.primary_features['tail_angle'] = pd.Series(self.calculate_tail_angle())
        self.primary_features['relative_angular_velocity_body'] = pd.Series(self.calculate_relative_angular_velocity_body()).apply(self.calibrate('relative_angular_velocity'))
        self.primary_features['relative_angular_velocity_tail'] = pd.Series(self.calculate_relative_angular_velocity_tail()).apply(self.calibrate('relative_angular_velocity'))
        
        # not used for primary features anymore but to calculate small/large body
        self.length = pd.Series(self.calculate_body_length())


    def calculate_signed_features(self):
        """ Further subdivides features based on sign i.e. positive/negative/absolute """

        # relative angular velocity automatically gets added by angular_velocity
        measures = ['curvature5', 'curvature25', 'angular_velocity', 'body_angle', 'tail_angle']

        # extract all the features associated with these measures e.g. 'curvature5_Center'
        features =  [feat for feat in self.expanded_features.keys() for m in measures if m in feat]

        # calculate signed values
        for feat in features:
            
            positive, negative, absolute = self.calculate_sign(self.expanded_features[feat])

            self.expanded_features[feat + '_positive'] = positive
            self.expanded_features[feat + '_negative'] = negative
            self.expanded_features[feat + '_absolute'] = absolute


    def calculate_event_features(self):
        """ Calculates event features (fraction, frequency, mean duration) """

        # calculate motion states
        self.stopping, self.walking, self.running = self.calculate_motion_states()

        # all events
        events = {
            'in_center': self.calculate_in_center(),
            'in_corner': self.calculate_in_corner(),
            'small_body': self.calculate_small_body(),
            'large_body': self.calculate_large_body(),
            'stopping': self.stopping,
            'walking': self.walking,
            'running': self.running
        }

        # calculate frequency, fraction, mean duration
        for e in events:
            self.event_features[e + '_frequency'] = self.calculate_frequency(events[e])
            self.event_features[e + '_fraction'] = self.calculate_fraction(events[e])
            self.event_features[e + '_mean_duration'] = self.calculate_mean_duration(events[e])


    def feature_expansion(self):
        """ Final transformations to primary features """

        # initialise expanded features
        self.expanded_features.update(self.primary_features)

        # divide by motion
        for feature in self.primary_features:

            # concat feature with conditional dataframes
            with_motions = pd.concat([self.primary_features[feature], self.stopping, self.walking, self.running], axis=1, join='inner')
            with_motions.columns = [feature, 'stopping', 'walking', 'running']
            
            self.expanded_features[feature + '_stopping'] = with_motions[with_motions['stopping']][feature]
            self.expanded_features[feature + '_walking'] = with_motions[with_motions['walking']][feature]
            self.expanded_features[feature + '_running'] = with_motions[with_motions['running']][feature]

        # sign features
        self.calculate_signed_features()

        # expand by quantile
        for feature in self.expanded_features:

            # split by body part
            q10, q50, q90, IQR = self.calculate_quantiles(self.expanded_features[feature])

            self.features[feature + '_q10'] = q10
            self.features[feature + '_q50'] = q50
            self.features[feature + '_q90'] = q90
            self.features[feature + '_IQR'] = IQR

        # add event features
        self.features.update(self.event_features)

        # other features:
        # total distance
        for b in self.bodyparts:
            d = self.calculate_total_distance((self.labels[b]))
            self.features['total_distance_' + b] = self.calibrate('distance')(d)
        
        # dwelling
        # TODO: too long computation time at the moment
        # dwelling = {'dwelling_' + b: self.calculate_dwelling(self.labels[b]) for b in self.bodyparts}
        # self.features.update(dwelling)


    def calculate_quantiles(self, labels):
        """ Calculates the 10, 50, and 90th quantiles and interquantile range """

        q10 = np.nanpercentile(labels, 10)
        q50 = np.nanpercentile(labels, 50)
        q90 = np.nanpercentile(labels, 90)
        IQR = np.nanpercentile(labels, 75) - np.nanpercentile(labels, 25)

        return q10, q50, q90, IQR


    def calculate_sign(self, labels):
        """ Calculates positive, negative, and absolute values """

        positive = labels[labels > 0]
        negative = labels[labels < 0]
        absolute = abs(labels)

        return positive, negative, absolute


    def calibrate(self, feature):

        # read in the calibration
        fps = self.calibration['fps']
        pxmm = self.calibration['pxmm']

        calibrations = {

            'speed': lambda x: x / pxmm * fps,
            'acceleration': lambda x: x / pxmm * fps**2,
            'jerk': lambda x: x / pxmm * fps**3,
            'distance': lambda x: x / pxmm,
            'curvature': lambda x: x * pxmm,
            'angular_velocity': lambda x: x * pxmm * fps,
            'relative_angular_velocity': lambda x: x * fps

        }

        return calibrations[feature]


    def smooth_labels(self, cutoff='auto'):
        """ Smooth labels based on cutoff
        If cutoff is 'auto': cutoff will be average body length
        """
        
        # define cutoff
        if cutoff == 'auto':
            cutoff = np.mean(self.calculate_body_length())

        # smoothing
        self.labels = smooth(self.labels, self.bodyparts, self.smoothing_factor, cutoff)


    def run(self):
        """ Main function that runs feature extraction and returns all features """

        # smooth data
        if self.smoothing_factor: self.smooth_labels()

        # calculate features
        self.calculate_primary_features()
        self.calculate_event_features()
        self.feature_expansion()

        # extract features
        feature_df = pd.DataFrame(self.features, index=[self.id])

        return feature_df
       