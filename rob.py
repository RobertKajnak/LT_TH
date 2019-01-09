#!/usr/bin/env python2
from __future__ import print_function
import time
import robobo
import cv2
import numpy as np
import vision
import motion

if __name__ == "__main__":
    rob = robobo.SimulationRobobo().connect(address='192.168.178.10', port=19999)
    #tilt the camera to be horizontal
    rob.set_phone_tilt(0.4, 10)
    move = motion.Motion(rob)

    
    sense = []
    last_direction=0
    while not np.any(sense):
        sense = rob.read_irs()
        
        im = rob.get_image_front()
        sim = vision.color_per_area(im)
        
        #R and B are flipped for some reason, even though it worked correctly,
        #when I loaded the png separately
        direction = [s[2] for s in sim].count('B') - \
                    [s[0] for s in sim].count('B')
        # the second part prevents oscillation, when the target is close
        if direction<0 and last_direction>=0:
            move.left()
            rob.talk('Spotted something on the left side')
        elif direction>0 and last_direction>=0:
            move.right()
            rob.talk('Spotted something on the right side')
        else:
            if [s[1] for s in sim].count('B')==0:
                move.right()    
                rob.talk('Can\'t see anything. Searching...')
            else:
                move.forward();
                rob.talk('Moving towards target')
        last_direction = direction
        
        rob.talk('\n{}'.format(sim))
    
    if [s[1] for s in sim].count('B')>0:
        rob.talk('Found some food')
        rob.set_emotion('Intriqued')
        
    
    #image = rob.get_image_front()
    
    #cv2.imwrite("test_pictures_0.png",image)
    
    #print(sense)
    
    