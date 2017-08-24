#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Locally sparse noise algorithm


Write down a list `L` of integers which will correspond in a 1-1 manner to a grid of
pixels. Now pick a random point in the plane p_0 by randomly selecting one of 
the integers from the . Now continue to select pixels from the 
remaining a pixel p_1 such that 
    ||p_0 - p_1|| >= d.
Continue by finding a point p_2 such that
    ||p_2 - p_0|| >= d and ||p_2 - p_1|| >= d.
Keep going until 








@author : johnyearsley
"""
import numpy as np
from scipy import spatial
import matplotlib.pyplot as plt
import matplotlib.animation as animation
    
def exclude_region(av,(y,x),exclusion=0):

    X,Y = np.meshgrid(np.arange(av.shape[1]), np.arange(av.shape[0]))
    circle_mask = ((X-x)**2 + (Y-y)**2) <= exclusion**2
    av[circle_mask] = False
    
    return av

def exclude_threshold(av,pixel_count_grid, repeats):
    
    count_mask = pixel_count_grid == repeats
    av[count_mask] = False
    
    return av


    
Y=16
X=28
exclusion=5
pixel_repeats = 100
num_frames= 10000
buffer_x=6
buffer_y=6



# generate a pixel grid space that is larger than the monitor so that
# we can use a circle mask around each point that is on.
X_extended = X+2*buffer_x
Y_extended = Y+2*buffer_y

extended_pixel_grid = [Y_extended, X_extended]

pixel_count_grid = np.zeros(extended_pixel_grid, dtype=np.uint8)
target_pixel_count = pixel_repeats*np.ones(extended_pixel_grid,dtype=np.uint8)


# 127 is mean luminance value
sn = 127*np.ones([num_frames,Y_extended,X_extended],dtype=np.uint8)

for frame in range(num_frames):
    available_pixels = np.ones(extended_pixel_grid, dtype=np.bool)
    
    
    available_pixels = exclude_threshold(available_pixels, 
                                       pixel_count_grid, 
                                       pixel_repeats)
        
    while np.any(available_pixels):
        y_available, x_available = np.where(available_pixels)
        pairs = zip(y_available,x_available)
        pair_index = np.random.choice(range(len(pairs)))
        y,x = pairs[pair_index]
        
        pixel_count_grid[y,x] += 1
        
        p = np.random.random()

        if p < 0.5:
            sn[frame,y,x] = 255
        else:
            sn[frame,y,x] = 0

        available_pixels = exclude_region(available_pixels,(y,x),exclusion=exclusion)
            
if np.array_equal(pixel_count_grid, target_pixel_count):
    print 'target pixel sample rate achieved'
    
else:
    print 'target pixel sample rate not achieved. Increase number of frames'

sn = sn[:,buffer_y:(Y+buffer_y), buffer_x:(X+buffer_x)]


class LocallySparseNoise(object):
    
    
    def __init__(self,
                 monitor,
                 indicator,
                 background=0.,
                 coordinate='degree',
                 grid_space=(10.,10.),
                 probe_size=(10.,10.),
                 probe_orientation=0.,
                 probe_frame_num=6,
                 subregion=None,
                 sign='ON-OFF',
                 iteration=1,
                 pregap_dur=2.,
                 postgap_dur=3.,
                 is_include_edge=True):
        
        super(SparseNoise,self).__init__(monitor=monitor,
                                         indicator=indicator,
                                         background=background,
                                         coordinate=coordinate,
                                         pregap_dur=pregap_dur,
                                         postgap_dur=postgap_dur)
        """    
        Initialize locally sparse noise object, inherits Parameters from Stim object
        """

        self.stim_name = 'LocallySparseNoise'
        self.grid_space = grid_space
        self.probe_size = probe_size
        self.probe_orientation = probe_orientation
        self.probe_frame_num = probe_frame_num
        self.is_include_edge = is_include_edge
        self.frame_config = ('is_display', '(azimuth, altitude)',
                             'polarity', 'indicator_color')

        if subregion is None:
            if self.coordinate == 'degree':
                self.subregion = [np.amin(self.monitor.deg_coord_y),
                                  np.amax(self.monitor.deg_coord_y),
                                  np.amin(self.monitor.deg_coord_x),
                                  np.amax(self.monitor.deg_coord_x)]
            if self.coordinate == 'linear':
                self.subregion = [np.amin(self.monitor.lin_coord_y),
                                  np.amax(self.monitor.lin_coord_y),
                                  np.amin(self.monitor.lin_coord_x),
                                  np.amax(self.monitor.lin_coord_x)]
        else:
            self.subregion = subregion

        self.sign = sign
        self.iteration = iteration


        

        
        self.clear()
        
    def _getgrid_points(self, is_plot=False):
        """
        generate all the grid points in display area (covered by both subregion and
        monitor span)

        Returns
        -------
        grid_points : n x 2 array, 
            refined [azi, alt] pairs of probe centers going to be displayed
        """

        rows = np.arange(self.subregion[0], 
                         self.subregion[1] + self.grid_space[0], 
                         self.grid_space[0])
        columns = np.arange(self.subregion[2], 
                            self.subregion[3] + self.grid_space[1], 
                            self.grid_space[1])

        xx, yy = np.meshgrid(columns, rows)

        gridPoints = np.transpose(np.array([xx.flatten(), yy.flatten()]))

        # get all the visual points for each pixels on monitor
        if self.coordinate == 'degree':
            monitor_x = self.monitor.deg_coord_x
            monitor_y = self.monitor.deg_coord_y
        elif self.coordinate == 'linear':
            monitor_x = self.monitor.lin_coord_x
            monitor_y = self.monitor.lin_coord_y
        else:
            raise ValueError('Do not understand coordinate system: {}. Should be either "linear" or "degree".'.
                             format(self.coordinate))

        left_alt = monitor_y[:, 0]
        right_alt = monitor_y[:, -1]
        top_azi = monitor_x[0, :]
        bottom_azi = monitor_x[-1, :]

        left_azi = monitor_x[:, 0]
        right_azi = monitor_x[:, -1]
        top_alt = monitor_y[0, :]
        bottom_alt = monitor_y[-1, :]

        left_azi_e = left_azi - self.grid_space[1]
        right_azi_e = right_azi + self.grid_space[1]
        top_alt_e = top_alt + self.grid_space[0]
        bottom_alt_e = bottom_alt - self.grid_space[0]

        all_alt = np.concatenate((left_alt, right_alt, top_alt, bottom_alt))
        all_azi = np.concatenate((left_azi, right_azi, top_azi, bottom_azi))

        all_alt_e = np.concatenate((left_alt, right_alt, top_alt_e, bottom_alt_e))
        all_azi_e = np.concatenate((left_azi_e, right_azi_e, top_azi, bottom_azi))

        monitorPoints = np.array([all_azi, all_alt]).transpose()
        monitorPoints_e = np.array([all_azi_e, all_alt_e]).transpose()

        # get the grid points within the coverage of monitor
        if self.is_include_edge:
            gridPoints = gridPoints[in_hull(gridPoints, monitorPoints_e)]
        else:
            gridPoints = gridPoints[in_hull(gridPoints, monitorPoints)]

        if is_plot:
            f = plt.figure()
            ax = f.add_subplot(111)
            ax.plot(monitorPoints[:, 0], monitorPoints[:, 1], '.r', label='monitor')
            ax.plot(monitorPoints_e[:, 0], monitorPoints_e[:, 1], '.g', label='monitor_e')
            ax.plot(gridPoints[:, 0], gridPoints[:, 1], '.b', label='grid')
            ax.legend()
            plt.show()

        return gridPoints
        
        
    @staticmethod
    def exclude_region(av,(x,y),exclusion=0):

        X,Y = np.meshgrid(np.arange(av.shape[0]), np.arange(av.shape[1]))

        mask = ((X-x)**2 + (Y-y)**2) <= exclusion**2
        av[mask] = False
     
    @staticmethod    
    def exclude_threshold(av,pixel_off_count_grid, pixel_on_count_grid, repeats):
    
        count_mask_off = pixel_off_count_grid == repeats
        av[count_mask_off] = False
          
        count_mask_on = pixel_on_count_grid == repeats
        av[count_mask_on] = False
    
    
    @classmethod
    def create_sparse_noise_matrix(cls,X=16,Y=28,exclusion=5,T=9000, 
                                   buffer_x=6, buffer_y=6, pixel_repeats=10):

        X_extended = X+2*buffer_x
        Y_extended = Y+2*buffer_y
        
        
        extended_pixel_grid = [X_extended, Y_extended]
        pixel_off_count_grid = np.zeros(extended_pixel_grid, dtype=np.uint8)
        pixel_on_count_grid = np.zeros(extended_pixel_grid, dtype=np.uint8)
        target_pixel_count = pixel_repeats*np.ones(extended_pixel_grid,dtype=np.uint8)

        # 127 is mean luminance value
        sn = 127*np.ones([T,X_extended,Y_extended],dtype=np.uint8)

        for t in range(T):
            available = np.ones(extended_pixel_grid).astype(np.bool)
            
            cls.exclude_threshold(available, pixel_off_count_grid, 
                                  pixel_on_count_grid, pixel_repeats)

            while np.any(available):
                x_available, y_available = np.where(available)

                pairs = zip(x_available,y_available)
                pair_index = np.random.choice(range(len(pairs)))
                x,y = pairs[pair_index]

                p = np.random.random()
                if p < 0.5:
                    sn[t,x,y] = 255
                    pixel_on_count_grid[x,y] += 1
                else:
                    sn[t,x,y] = 0
                    pixel_off_count_grid[x,y] += 1
                      

                cls.exclude_region(available,(x,y),exclusion=exclusion)
                
        if (np.array_equal(pixel_off_count_grid, target_pixel_count) and 
            np.array_equal(pixel_on_count_grid, target_pixel_count)):

            print 'target pixel sample rate achieved'
        else:
            print 'target pixel sample rate not achieved. Increase number of frames'

        return sn[:, buffer_x:(X+buffer_x),buffer_y:(Y+buffer_y)]



