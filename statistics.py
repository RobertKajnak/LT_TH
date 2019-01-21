# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 08:24:46 2019

@author: Hesiris
"""
import json

class Statistics:
    def __init__(self, filename = None):
        if filename is None:
            self.cont_var = {}
            self.const_var = {}
            self.paths = {}
            self.x_labels = {}
            self.y_labels = {}
        else:        
            f = open(filename,"rb")
            ps = json.load(f)
            self.cont_var = ps[0]
            self.const_var = ps[1]
            self.paths = ps[2]
            self.x_labels = ps[3]
            self.y_labels = ps[4]
            
    
    def add_continuous_point(self,var_name, point_value):
        if var_name in self.cont_var:
            self.cont_var[var_name].append(point_value)
        else:
            self.cont_var[var_name] = [point_value]
    
    def set_constant(self,var_name, value):
        self.const_var[var_name] = value
    
    def set_xlabel(self,var_name, label):
        self.x_labels[var_name] = label
        
    def set_ylabel(self,var_name, label):
        self.y_labels[var_name] = label

    def add_path(self,path_value=None, path_category_name=''):
        '''
        if path_value is None, a new empty path is initialized
        path_category_name permits storing multiple paths concurrently
        '''
        
        if path_category_name not in self.paths:
            self.paths[path_category_name] = []
            
        if path_value is not None:
            self.paths[path_category_name].append(path_value)
        else:
            self.paths[path_category_name].append([])
            
    def add_point_to_path(self,point_value,path_category_name=''):
        '''
        appends the point to the last path under the path_name category
        '''
        if self.paths[path_category_name] is None:
            self.paths[path_category_name] = [[]]
            
        self.paths[path_category_name][-1].append(point_value)
        
        
    def save(self,filename):
        json.dump([self.cont_var ,self.const_var ,self.paths ,
            self.x_labels,
            self.y_labels], open(filename, "w"))


import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import polyfit

def dist(a,b):
    return np.linalg.norm(np.asarray(a)-np.asarray(b))

class Statistics_plotter:
    def __init__(self,filename,plot_trend = False, mavg_window_size=1):
        self.stats = Statistics(filename)#json.load
        
        cont = len(self.stats.cont_var.keys())
        
        self.ncols = 1 if cont<=2 else 2
        self.nrows = int(np.ceil(cont/2))
        
        self.mavg_windows = {}
        self.trends = {}
        for k in self.stats.cont_var.keys():
            self.trends[k] = plot_trend
            self.mavg_windows[k] = mavg_window_size
    
    def _plot_single(self,Y,title,y_label,x_label, mavg_window=30,show_trend = False):
        if not Y is None:
            Y = np.asarray(Y)
            ax = self.fig.add_subplot(self.nrows,self.ncols,self._current_sp_index)
            ax.set_xlabel(x_label, fontsize=12)
            ax.set_ylabel(y_label, fontsize=12)
            plt.title(title)
            if mavg_window==1:
                y = Y
            else:
                y = np.asarray([np.mean(Y[i:i+mavg_window-1]) for i in range(Y.shape[0]-mavg_window)])
            x = np.asarray(range(y.shape[0]))
            plt.plot(x, y,'k', linewidth=3)
            if show_trend:
                b, m = polyfit(x, y, 1)
                plt.plot(x, b + m * x, '-', linewidth=3)
            
            '''if not Y is self.cols:
                #ignore the initial random successes
                Y_offset = int(Y.shape[0]/3)
                idx_best = np.argmax(Y[Y_offset:])+Y_offset
                val_best = np.max(y)
                ax.annotate('best episode: '+ str(idx_best), 
                    xy=(idx_best, val_best), 
                    xytext=(0, val_best*1.1), 
                    arrowprops = dict(facecolor='black', shrink=0.05))
            '''
            self._current_sp_index +=1
        
    def plot_path(self,index, category_name = ''):
        X = [v[0] for v in self.stats.paths[category_name][index]]
        Y = [v[1] for v in self.stats.paths[category_name][index]]
        plt.figure()
        plt.scatter(X,Y,alpha=0)
        plt.axes().set_aspect('equal')
        plt.title(category_name + ' Path ' + str(index))
        plt.plot(X[0],Y[0],'gs',markersize =15)
        plt.plot(X[-1],Y[-1],'r^',markersize =15)
        
        #calculate arrow head dimesnions
        left,right = plt.ylim()
        hw0 = (right-left)/6.0
        
        for i in range(len(X)-1):
            if i == len(X)-2:
                hw=0
                hl=0
            else:
                d = dist([X[i],Y[i]],[X[i+1],Y[i+1]])
                hw = hw0 * d/float(right-left)
                hl = hw
                #hl = np.abs(X[i+1]-X[i])/5.0
            plt.arrow(X[i],Y[i],X[i+1]-X[i],Y[i+1]-Y[i],
                      head_width=hw, head_length=hl,
                      length_includes_head=True)
            
    def set_xlabel(self,var_name, label):
        self.stats.set_xlabel(var_name,label)
        
    def set_ylabel(self,var_name, label):
        self.stats.set_ylabel(var_name,label)

    def set_mavg_window(self, var_name, window_size):
        if var_name not in self.trends:
            raise ValueError('Cannot set window size for a variable that does not exist')
        self.mavg_windows[var_name] = window_size

    def set_trend(self, var_name, show_trend = True):
        if var_name not in self.trends:
            raise ValueError('Cannot set trend for a variable that does not exist')
        self.trends[var_name] = show_trend
        

    def plot_all(self):
        self.fig = plt.figure(num=None, figsize=(5*self.ncols, 3*self.nrows), dpi=80, facecolor='w', edgecolor='black')
        plt.subplots_adjust( wspace=0.3, hspace=0.5)
        
        for C in self.stats.const_var.keys():
            print(str(C) + ': ' + str(self.stats.const_var[C]))
        
        self._current_sp_index=1
        for k in self.stats.cont_var.keys():
            if k in self.stats.x_labels:
                x_l = self.stats.x_labels[k]
            else:
                x_l = ''
            
            if k in self.stats.y_labels:
                y_l = self.stats.y_labels[k]
            else:
                y_l = ''
            #self.plot_single(self.cont[k],)
            if k in self.mavg_windows:
                mavg_window = self.mavg_windows[k]
            else:
                mavg_window = 1
            self._plot_single(self.stats.cont_var[k],k,y_l,x_l,mavg_window,self.trends[k])
        
        plt.show()
        
if __name__ == '__main__':
    #%% Record statistics
    st = Statistics()
    
    #Add constants as such
    st.set_constant('alpha',0.1)
    st.set_constant('Number of runs', 20)
    
    #add variables that change during iterations
    for i in range(20):
        st.add_continuous_point('bad',i/2-3+np.random.uniform(0,5))
        st.add_continuous_point('good',i*2+np.random.uniform(0,10))
        st.add_continuous_point('spike',1 if i==13 else 0)
        #generate 4 different paths
        if i%5 ==0:
            st.add_path()
        st.add_point_to_path([20-i-np.random.uniform(1,5),i+np.random.uniform(1,5)])
        
    #add a label to explain the variables. Optional
    st.set_xlabel('bad', 'time')
    st.set_ylabel('bad', 'performance')
    st.set_xlabel('good', 'time')
    st.set_ylabel('good', 'performance')
        
    #save after test is finished
    st.save('test_stats')
        
    #%%Display statistics
    #Displaying the first order approximation seems like a good idea
    #We could specify mav_window_size, but we don't need it this time
    ss = Statistics_plotter('test_stats', plot_trend = True)
    
    ss.plot_all()
    
    #Oh, I spikes doesn't have labels on the axes
    ss.set_xlabel('spike','time')
    ss.set_ylabel('spike','relevant point')
    
    #hmm, maybe good and bad should have a moving average applied with a window
    #size of 3
    ss.set_mavg_window('bad',5)
    ss.set_mavg_window('good',5)
    
    #Maybe we don't need the trend for the spikes
    ss.set_trend('spike',False)
    
    #Let's plot the again
    print('New Plot:')
    print('')
    
    ss.plot_all()
    
    #Let's also plot a path
    ss.plot_path(0)
    
    
    
    