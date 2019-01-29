import numpy as np
from collections import Counter
import cv2

class Vision:
    def __init__(self,robot,is_simulation,
                 target_color_min= (36, 25, 25), target_color_max=(70, 255,255), 
                 area_size=(4,3), downsampling_rate = 10, threshold= 3):
        """ Handles Input from the camera of the robot and discretizes it, to find the target color
            Example usage under color_per_area method
        Args:
            robot: robot returned from simulation.py or hardware.py
            is_simulation: True for V-REP, False for physical Robobo
            target_color_min: the minimum value in HSV
            target_color_max: the maximum value in HSV
            area_size: Specify the size of the grid returned (h,w)
            downsampling_rate: only every n-th pixel is considered
            threshold: At least this many pixels need to trigger the binary representation
        """
        self.rob = robot
        self.filters = []
        self.filters.append((target_color_min,target_color_max))
        self.area_size = area_size
        self.downsampling_rate = downsampling_rate
        self.threshold = threshold
        
        self.img_counter=0

    def add_filter(self, target_color_min, target_color_max):
        self.filters.append((target_color_min,target_color_max))
        
    def color_per_area(self, image = None):
        """ Captures an image from the robobo and analyses where the target color can be seen
                image: if None is specified, rob.get_image_front() is used
            Returns:
                zones: The sum of target color pixels in each zone. 
                zones_binary: 1 or 0, depending on the sum and threshold
                E.g.  if area_size==(2,3) and threshold==3 
                => zones = [[123,23,0],[32,1,0]]
                => zones_binary= [[1,1,0],[1,0,0]]
                
                
        """
        if image is None:
            image = self.rob.get_image_front()        

        
        image = image[int(image.shape[0]/5)::self.downsampling_rate,::self.downsampling_rate]
        try:
            cv2.imwrite("test_image_cont.png".format(self.img_counter), image)
        except:
            print('Save error')
        
        self.img_counter +=1
        
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        imask = np.zeros((image.shape[0],image.shape[1]),dtype = np.bool)
        for filt in self.filters:
            mask = cv2.inRange(hsv,filt[0], filt[1])
            imask |= mask>0
        
        green = np.zeros((image.shape[0],image.shape[1]),dtype = np.int8)
        green[imask] = 1 
        #print(green)
        
        green_debug = np.zeros_like(image, np.uint8)
        green_debug[imask] = [255,255,255]
        try:
            cv2.imwrite("green.png".format(self.img_counter), green_debug)
        except:
            print('Save error')

        zones = np.empty(self.area_size,dtype=np.int16)
        zones_binary = np.empty(self.area_size,dtype=np.int8)
        
        for i in range(self.area_size[0]):
            for j in range(self.area_size[1]):
                dx = np.int(green.shape[0]/self.area_size[0])
                dy = np.int(green.shape[1]/self.area_size[1])
                
                zones[i][j] = np.sum(green[i*dx:(i+1)*dx,j*dy:(j+1)*dy])
                zones_binary[i][j] = int(zones[i][j]>self.threshold)
        
        return zones, zones_binary
        

    
# [[l[0] for l in Counter(line).most_common()] for line in ds]