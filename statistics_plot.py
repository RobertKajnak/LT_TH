# -*- coding: utf-8 -*-
"""
Created on Sat Jan 19 12:34:36 2019

@author: Hesiris
"""
import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import polyfit

def dist(a,b):
    return np.linalg.norm(np.asarray(a)-np.asarray(b))

class stats_plotter:
    def __init__(self,directory):
        self.directory = directory
        self.file_count = 0

        self.dists = self._load(self.directory+'/a_furthest_distance.npy')
        self.colls = self._load(self.directory+'/all_collisions.npy')
        self.poss = self._load(self.directory+'/all_positions.npy')
        self.rews = self._load(self.directory+'/all_rewards.npy')
        self.steps = self._load(self.directory+'/all_steps_survived.npy')
        
        self.ncols = 1 if self.file_count-1<=2 else 2
        self.nrows = int(np.ceil(self.file_count-1/2))
        
    def _load(self,filename):
        try:
            A = np.load(filename)
            self.file_count +=1
            return A
        except:
            print(filename+' not found')
            return None
    
    def _plot_single(self,Y,y_label,x_label='episode',mavg=30):
        if not Y is None:
            ax = self.fig.add_subplot(self.nrows,self.ncols,self._current_sp_index)
            ax.set_xlabel(x_label, fontsize=12)
            ax.set_ylabel(y_label, fontsize=12)
            
            if mavg==0:
                y=Y
            else:
                y = np.asarray([np.mean(Y[i:i+mavg]) for i in range(Y.shape[0]-mavg)])
            x = range(y.shape[0])
            plt.plot(x, y,'k', linewidth=3)
            
            if mavg!=0:
                b, m = polyfit(x, y, 1)
                plt.plot(x, b + m * x, '-', linewidth=3)
            
            if not Y is self.colls:
                idx_best = np.argmax(Y)
                val_best = np.max(y)
                ax.annotate('best episode: '+ str(idx_best), 
                    xy=(idx_best, val_best), 
                    xytext=(0, val_best*1.1), 
                    arrowprops = dict(facecolor='black', shrink=0.05))
            
            self._current_sp_index +=1
        
    def plot_path(self,index):
        X = [v[0] for v in self.poss[index]]
        Y = [v[1] for v in self.poss[index]]
        plt.scatter(X,Y,alpha=0)
        plt.plot(X[0],Y[0],'gs',markersize =15)
        plt.plot(X[-1],Y[-1],'r^',markersize =15)
        
        for i in range(len(X)-1):
            plt.arrow(X[i],Y[i],X[i+1]-X[i],Y[i+1]-Y[i],head_width=0.008,length_includes_head=True)
            
        
    def plot_all(self):
        self.fig = plt.figure(num=None, figsize=(5*self.ncols, 3*self.nrows), dpi=80, facecolor='w', edgecolor='black')
        plt.subplots_adjust( wspace=0.3, hspace=0.5)
        
        self._current_sp_index=1
        self._plot_single(self.dists,'Max Distance Covered')
        self._plot_single(self.colls,'End by collision?',mavg=0)
        self._plot_single(self.rews,'Rewards/episode')
        self._plot_single(self.steps,'Steps/episode')
        
        plt.show()
        

