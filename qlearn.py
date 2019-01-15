#!/usr/bin/env python2
from __future__ import print_function
import time
import robobo
#import cv2
import numpy as np
#import vision
import motion
import irsensors
import random
import select
import sys

def keypressed():
    i,_,_ = select.select([sys.stdin],[],[],0.0001)
    if i:
        return True
    else:
        return False
def heardEnter():
    i,o,e = select.select([sys.stdin],[],[],0.0001)
    for s in i:
        if s == sys.stdin:
            return True
    return False

rewards = {0:10,
           1:5,
           2:5,
           3:1,
           4:1}
def get_reward(action, sensors_one_hot):
    if np.any(sensors_one_hot[1::2]):
        return rewards[action]*5
    else:
        return rewards[action]
        

def train(starting_qtable_filename=None, all_collisions_filename=None,all_rewards_filename=None):
    '''
    params:
        starting_episode==None starts from scratch
        
    '''
    #rob = robobo.SimulationRobobo().connect(address='192.168.178.10', port=19997)
    #rob = robobo.SimulationRobobo().connect(address='196.168.137.1', port=19997)
    rob = robobo.SimulationRobobo().connect(address='192.168.1.6', port=19997)
    #rob = robobo.HardwareRobobo(camera=True).connect(address="192.168.1.15")

    move = motion.Motion(rob,True,speed=30,time=500)
    
    sens = irsensors.Sensors(rob,True)
    
    """Training the agent"""
    
    if starting_qtable_filename:
        q_table = np.load(starting_qtable_filename)
    else:
        q_table = np.zeros([2**8, 5])

    # Hyperparameters
    alpha = 0.1
    gamma = 0.6
    epsilon = 0.1
    
    # For plotting metrics
    #all_epochs = []
    all_collisions = np.load(all_collisions_filename) if all_collisions_filename else []
    all_rewards = np.load(all_rewards_filename) if all_rewards_filename else []
    
    for i in range(1, 2200):
        #state = env.reset()
        rob.stop_world()
        time.sleep(3)
        rob.play_simulation()
        state = 0
    
        epochs, collisions, reward, reward_total = 0, 0, 0, 0
        done = False
        
        while not done:
            if random.uniform(0, 1) < epsilon:
                action = random.choice(range(5))
            else:
                action = np.argmax(q_table[state]) # Exploit learned values
            
            move.move(action)
            next_state = 0
            sens_val, collision = sens.binary()
            for j in range(8):
                next_state+=sens_val[j]*2**(j)
            
#            sc = sens.continuous()
#            for v in sc:
#                print("{0:.4f}".format(v),end=', ')
#            print('')
            #print(sc)
            print(sens.strings())
            #print(next_state)
            #print(sens_val)
            reward = get_reward(action, sens_val)
                    
            reward_total+=reward
            
            done = True if (rob.getTime() > 30000 or collision) else False 
            if done:
                reward = -90;
            old_value = q_table[state, action]
            next_max = np.max(q_table[next_state])
            
            new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
            q_table[state, action] = new_value
    
            if collision:
                collisions += 1
    
            state = next_state
            epochs += 1
        all_collisions.append(0 if collision else 1)
        all_rewards.append(reward_total)
        np.save('tables/q_table{}'.format(i),q_table)
        #if i % 10 == 0:
        print('Episode: {}, reward:{}'.format(i,reward_total))
        if heardEnter():
            break
    
    print("Training finished.\n")
    
    np.save('tables/all_rewards',all_rewards)
    np.save('tables/all_collisions',all_collisions)
    rob.stop_world()
    
    
def test(q_table_filename):
    #rob = robobo.SimulationRobobo().connect(address='192.168.178.10', port=19997)
    rob = robobo.HardwareRobobo(camera=True).connect(address="192.168.1.7")
    move = motion.Motion(rob,False,speed=50,time=500)
   
    sens = irsensors.Sensors(rob,False)
    
    q_table = np.load(q_table_filename)
    
    #rob.play_simulation()
    
    done = False
    
    while not done:
        state = 0
        sens_val, collision = sens.one_hot()
        for j in range(1,16,2):
            state+=sens_val[j]*2**((j-1)/2)
        action = np.argmax(q_table[state])
        print(sens_val)
        move.move(action)
        
        
if __name__ == "__main__":
    train()
