#!/usr/bin/env python2
from __future__ import print_function
import time
import robobo
import cv2
import numpy as np
import vision

if __name__ == "__main__":
    rob = robobo.SimulationRobobo().connect(address='192.168.178.10', port=19999)
    rob.set_phone_tilt(0.4, 10)
    
    sense = []
    #while not np.any(sense):
    #    rob.move(30, 30,1000)
    #    sense = rob.read_irs()
    im = rob.get_image_front()
    
    image = rob.get_image_front()
    rob.talk(vision.color_per_area(image))
    #cv2.imwrite("test_pictures_0.png",image)
    
    print(sense)
    rob.talk('Found somethign')
    rob.set_emotion('intriqued')
    
    