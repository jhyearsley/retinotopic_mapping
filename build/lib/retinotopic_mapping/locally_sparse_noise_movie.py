#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 29 16:00:38 2017

@author: johnyearsley
"""

import os, random

from psychopy import core,visual

time_for_image = 1
time_for_gap = 0


images = []

path = os.path.join(os.getcwd(),'lsn_test')

for image in os.listdir(path):
    images.append(os.path.join(path, image))
    
my_win = visual.Window(size=(600,600), color=[0.,0.,0.])

gray_back = visual.ImageStim(my_win, color=[0.,0.,0.])

# Shuffle images
random.shuffle(images)

for img in images:
    stim = visual.ImageStim(my_win, image=img)
    stim.draw()
    
    core.wait(time_for_image)
    my_win.flip()
    
    gray_back.draw()
    core.wait(time_for_gap)
    
    my_win.flip()
    
    
    
my_win.close()
