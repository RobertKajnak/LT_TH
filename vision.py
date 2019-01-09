import numpy as np
from collections import Counter

colors={'b':(0,0,0),
        'w':(255,255,255),
        'R':(255,0,0),
        'G':(0,255,0),
        'B':(0,0,255),
        'brown':(150, 75, 0)}

def cdist(a,b):
    return np.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2+(a[2]-b[2])**2)
    #return np.linalg.norm(a-b)

def closest_color(color):
    best_color = list(colors.keys())[0]
    best_dist = 442 #np.sqrt(255**2*3)==441.672
    for c in list(colors.items()):
        current_dist = cdist(c[1],color)
        if current_dist<best_dist:
            best_dist = current_dist
            best_color = c[0];
    return best_color

def _most_non_white_color(im):
    return []


def color_per_area(image, area_size=(3,3), downsampling_rate = 10):
    w = image.shape[0]
    h = image.shape[1]
    
    downsampled = np.empty((np.int(w/downsampling_rate)+1,np.int(h/downsampling_rate)+1),dtype=np.unicode)
    
    xd=0
    for x in np.arange(0,h,downsampling_rate):
        yd=0
        for y in np.arange(0,w,downsampling_rate):
            downsampled[xd][yd] = closest_color(image[x,y])[0]
            yd+=1
        xd+=1
    
    zones = np.empty(area_size,dtype=np.unicode)
    fbg = lambda x: x!='b' and x!='w' #skip the background colors
    #most common without filtering: Counter([Counter(line).most_common()[0][0] for line in downsampled]).most_common()[0][0]
    for i in range(area_size[0]):
        for j in range(area_size[1]):
            dx = np.int(downsampled.shape[0]/area_size[0])
            dy = np.int(downsampled.shape[1]/area_size[1])
            sm = np.array(downsampled[i*dx:(i+1)*dx,j*dy:(j+1)*dy],dtype=np.unicode)
            lines = [y[0] if y!=[] else 'b' for y in [list(filter(fbg,x)) for x in [[l[0] for l in Counter(line).most_common()] for line in sm ]]]
            zones[i][j] = [y[0] if y!=[] else 'b' for y in [list(filter(fbg,[x[0] for x in Counter(lines).most_common()]))]][0]
    
    

    #zones[area_size[0]*h/x,area_size[1]*w/y]
    #print(downsampled)
    return zones


# [[l[0] for l in Counter(line).most_common()] for line in ds]