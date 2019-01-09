import numpy as np

colors={'black':(0,0,0),
        'white':(255,255,255),
        'red':(255,0,0),
        'green':(0,255,0),
        'blue':(0,0,255)}

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

#def color_per_area(image, areas=(3,3), downsampling_rate = 10):
#    w = image.shape[0]
#    h = image.shape[1]
#    
#    
#    for x in range(0,h,downsampling_rate):
#        for y in range(0,w,downsampling_rate):
#            
#    return []