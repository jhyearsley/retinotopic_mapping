#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Locally sparse noise algorithm


Write down a list L of integer tuples which will correspond in a 1-1 manner to a grid of
pixels. Now pick a random value p_0 = (m,n) from the list of coordinates values. Next,
remove all tuples p_i from the list that do not satisfy the relation
    ||p_0 - p_i|| >= d.

Now we have a new list L' such that each point is at distance at least d away from 
p_0. Then pick a point p_1 from L' and repeat the process until no points remain.


@author : johnyearsley
"""
import numpy as np

    
def exclude_region(av,(y,x),exclusion=0):
    """ used to filter points that are within distance `exclusion` """
    X,Y = np.meshgrid(np.arange(av.shape[1]), np.arange(av.shape[0]))
    circle_mask = ((X-x)**2 + (Y-y)**2) <= exclusion**2
    av[circle_mask] = False
    
    return av

def exclude_threshold(av,pixel_count_grid, repeats):
    """ filter points that have already been chosen `repeats` times 
    
    Parameters
    ----------
    av : ndarray of booleans
        the matrix of available pixels to choose from
    pixel_count_grid : ndarray of integers
        

    """
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
# we can use a circle mask around each point that is on (or off respectively).
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


