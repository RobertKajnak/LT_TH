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
        
        if is_simulation:
            self.colors={'b':(0,0,0),
                'w':(255,255,255),
                'R':(255,0,0),
                'G':(0,255,0),
                'B':(0,0,255),
                'brown':(150, 75, 0)}
        else:
            self.colors={'b':(0,0,0),
                'w':(255,255,255),
                'Bh':(170,170,170),
                'Bl':(150,130,130),
                'grey':(70,70,70),
                'R':(0,0,255),
                'G':(50,170,170),
                'B':(255,0,0),
                'brown':(0, 75, 150)}
        self.img_counter=0

    def add_filter(self, target_color_min, target_color_max):
        self.filters.append((target_color_min,target_color_max))
        
    def color_per_area(self):
        """ Captures an image from the robobo and analyses where the target color can be seen
            Returns:
                zones: The sum of target color pixels in each zone. 
                zones_binary: 1 or 0, depending on the sum and threshold
                E.g.  if area_size==(2,3) and threshold==3 
                => zones = [[123,23,0],[32,1,0]]
                => zones_binary= [[1,1,0],[1,0,0]]
                
                
        """
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
        
    def _cdist(self,a,b):
        """Deprecated. Calculate distance between colors and b in RGB space"""
        return np.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2+(a[2]-b[2])**2)
        #return np.linalg.norm(a-b)
    
    def closest_color(self,color):
        """ Deprecated. Determines which of the colors property colors is closest to the argument """
        best_color = list(self.colors.keys())[0]
        best_dist = 442 #np.sqrt(255**2*3)==441.672
        for c in list(self.colors.items()):
            current_dist = self._cdist(c[1],color)
            if current_dist<best_dist:
                best_dist = current_dist
                best_color = c[0];
        return best_color

    def color_per_area_depr(self):
        """ Old method based on RGB distance. Will be removed in next version"""
        image = self.rob.get_image_front()
        image = image[:][int(image.shape[0]/2):]
        
        
        self.img_counter += 1
        w = image.shape[0]
        h = image.shape[1]
        
        downsampled = np.empty((np.int((np.floor(h/self.downsampling_rate)))+1,
                                np.int(np.floor(w/self.downsampling_rate))+1),
                                dtype=np.unicode)
    
        downsampled_pic = np.empty((np.int((np.floor(h/self.downsampling_rate)))+1,
                                np.int(np.floor(w/self.downsampling_rate))+1,3),
                                )
        xd=0
        for x in np.arange(0,h,self.downsampling_rate):
            yd=0
            for y in np.arange(0,w,self.downsampling_rate):
                downsampled_pic[xd][yd] = image[y][x]
                downsampled[xd][yd] = self.closest_color(image[y][x])[0]
                yd+=1
            xd+=1
            
        
        
        zones = np.empty(self.area_size,dtype=np.unicode)
        fbg = lambda x: x=='G'# x!='b' and x!='w' #skip the background colors
        #most common without filtering: Counter([Counter(line).most_common()[0][0] for line in downsampled]).most_common()[0][0]
        for i in range(self.area_size[0]):
            for j in range(self.area_size[1]):
                dx = np.int(downsampled.shape[0]/self.area_size[0])
                dy = np.int(downsampled.shape[1]/self.area_size[1])
                sm = np.array(downsampled[i*dx:(i+1)*dx,j*dy:(j+1)*dy],dtype=np.unicode)
                lines = [y[0] if y!=[] else 'b' for y in [list(filter(fbg,x)) for x in [[l[0] for l in Counter(line).most_common()] for line in sm ]]]
                zones[i][j] = [y[0] if y!=[] else 'b' for y in [list(filter(fbg,[x[0] for x in Counter(lines).most_common()]))]][0]
                #print([y[0] if y!=[] else 'b' for y in [list(filter(fbg,x)) for x in [[l[0] for l in Counter(line).most_common()] for line in sm ]]])
                #print([y if y!=[] else 'b' for y in [x for x in Counter(lines)]])
        
        fn = ''.join(zones[0])
        cv2.imwrite("test_pictures{}_{}.png".format(self.img_counter,fn),downsampled_pic)
        return zones
    
    
# [[l[0] for l in Counter(line).most_common()] for line in ds]