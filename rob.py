#!/usr/bin/env python2
from __future__ import print_function
import time
import robobo
import cv2
import numpy as np
import vision
import motion
import irsensors
import select
import sys

def heardEnter():
    i,o,e = select.select([sys.stdin],[],[],0.0001)
    for s in i:
        if s == sys.stdin:
            return True
    return False

if __name__ == "__main__":
    #rob = robobo.SimulationRobobo().connect(address='192.168.178.10', port=19999)
    #rob = robobo.SimulationRobobo().connect(address='196.168.137.1', port=19997)
    rob = robobo.SimulationRobobo().connect(address='192.168.1.6', port=19997)
    #rob = robobo.HardwareRobobo(camera=True).connect(address="192.168.1.7")
    
    #tilt the camera to be horizontal
    #rob.play_simulation()
    rob.set_phone_tilt(0.4, 10)
    move = motion.Motion(rob,True,speed=50,time=500)
   
    sense = []
    last_direction=0
    
    sens = irsensors.Sensors(rob,True)
    

    time.sleep(3)
    for i in range (150):
        if not np.any(sense):
            move.move(-1)
        move.move(-1)
        time.sleep(0.5)
        sense = sens.discrete()
        print(sense)
        print(sens.continuous())
        if heardEnter():
            break
#
#    move.forward()        
#    time.sleep(1)
#    move.right()
#    time.sleep(1)
#    move.left()
#    time.sleep(1)
#    move.turn_right()
#    time.sleep(1)
#    move.turn_left()
#    time.sleep(1)
#    move.backwards()
#    
#    for i in range (100):
#        if not np.any(np.array(sense)):
#            move.right()
#        sense = rob.read_irs()
#        print(sense)
#        time.sleep(0.1)
#
#    while not np.any(sense>500):
#        sense = rob.read_irs()
#        
#        im = rob.get_image_front()
#        sim = vision.color_per_area(im)
#        #sim = ['B','B','B']
#        #R and B are flipped for some reason, even though it worked correctly,
#        #when I loaded the png separately
#        direction = [s[2] for s in sim].count('B') - \
#                    [s[0] for s in sim].count('B')
#        # the second part prevents oscillation, when the target is close
#        if direction<0 and last_direction>=0:
#            move.left()
#            rob.talk('Spotted something on the left side')
#        elif direction>0 and last_direction>=0:
#            move.right()
#            rob.talk('Spotted something on the right side')
#        else:
#            if [s[1] for s in sim].count('B')==0:
#                move.right()    
#                rob.talk('Can\'t see anything. Searching...')
#            else:
#                move.forward();
#                rob.talk('Moving towards target')
#        last_direction = direction
#        
#        rob.talk('\n{}'.format(sim))
#        #rob.reset_position()
#    
#    if [s[1] for s in sim].count('B')>0:
#        rob.talk('Found some food')
#        rob.set_emotion('Intriqued')
#        
#    #image = rob.get_image_front()
#    #rob.stop_world()
#    #rob.play_simulation()
#    #cv2.imwrite("test_pictures_0.png",image)
#    
#    #print(sense)
#    
#    