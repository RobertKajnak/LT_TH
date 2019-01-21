import numpy as np
from collections import Counter

class Vision:

    def __init__(self,robot,is_simulation=True,area_size=(4,3), downsampling_rate = 10):
        self.rob = robot
        self.area_size = area_size
        self.downsampling_rate = downsampling_rate
        
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
                'R':(0,0,255),
                'G':(0,255,0),
                'B':(255,0,0),
                'brown':(0, 75, 150)}
    
    def _cdist(self,a,b):
        return np.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2+(a[2]-b[2])**2)
        #return np.linalg.norm(a-b)
    
    def closest_color(self,color):
        best_color = list(self.colors.keys())[0]
        best_dist = 442 #np.sqrt(255**2*3)==441.672
        for c in list(self.colors.items()):
            current_dist = self._cdist(c[1],color)
            if current_dist<best_dist:
                best_dist = current_dist
                best_color = c[0];
        return best_color
    
    def _most_non_white_color(self,im):
        return []
    
    
    def color_per_area(self):
        image = self.rob.get_image_front()
        w = image.shape[0]
        h = image.shape[1]
        #print(image.shape)
        
        downsampled = np.empty((np.int((np.floor(h/self.downsampling_rate)))+1,
                                np.int(np.floor(w/self.downsampling_rate))+1),
                                dtype=np.unicode)
        xd=0
        for x in np.arange(0,h,self.downsampling_rate):
            yd=0
            for y in np.arange(0,w,self.downsampling_rate):
                downsampled[xd][yd] = self.closest_color(image[y][x])[0]
                yd+=1
            xd+=1
        
        zones = np.empty(self.area_size,dtype=np.unicode)
        fbg = lambda x: True#x!='b' and x!='w' #skip the background colors
        #most common without filtering: Counter([Counter(line).most_common()[0][0] for line in downsampled]).most_common()[0][0]
        for i in range(self.area_size[0]):
            for j in range(self.area_size[1]):
                dx = np.int(downsampled.shape[0]/self.area_size[0])
                dy = np.int(downsampled.shape[1]/self.area_size[1])
                sm = np.array(downsampled[i*dx:(i+1)*dx,j*dy:(j+1)*dy],dtype=np.unicode)
                lines = [y[0] if y!=[] else 'b' for y in [list(filter(fbg,x)) for x in [[l[0] for l in Counter(line).most_common()] for line in sm ]]]
                zones[i][j] = [y[0] if y!=[] else 'b' for y in [list(filter(fbg,[x[0] for x in Counter(lines).most_common()]))]][0]
        
        
        return zones
    
    
# [[l[0] for l in Counter(line).most_common()] for line in ds]