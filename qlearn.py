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
           3:2,
           4:2,
           5:1,
           6:1}
def get_reward(action, sensors_one_hot):
    #if np.any(np.array(sensors_one_hot)):
    #    return rewards[action]*5
    #else:
        return rewards[action]
        

def train(starting_qtable_filename=None, all_collisions_filename=None,all_rewards_filename=None):
    '''
    params:
        starting_episode==None starts from scratch
        
    '''
    #rob = robobo.SimulationRobobo().connect(address='192.168.178.10', port=19997)
    #rob = robobo.SimulationRobobo().connect(address='196.168.137.1', port=19997)
    rob = robobo.SimulationRobobo().connect(address='192.168.1.15', port=19997)
    #rob = robobo.HardwareRobobo(camera=True).connect(address="192.168.1.15")

    move = motion.Motion(rob,True,speed=30,time=500)
    
    sens = irsensors.Sensors(rob,True)
    
    """Training the agent"""
    
    if starting_qtable_filename:
        q_table = np.load(starting_qtable_filename)
    else:
        q_table = np.zeros([2**8*7, 7])

    # Hyperparameters
    alpha = 0.1
    gamma = 0.6
    epsilon = 0.1
    
    # For plotting metrics
    #all_epochs = []
    all_collisions = np.load(all_collisions_filename) if all_collisions_filename else []
    all_rewards = np.load(all_rewards_filename) if all_rewards_filename else []
    
    for i in range(1, 5000):
        #state = env.reset()
        rob.stop_world()
        time.sleep(3)
        rob.play_simulation()
        state = 0
    
        epochs, collisions, reward, reward_total = 0, 0, 0, 0
        action = 0
        done = False
        
        while not done:
            prev_action = action
            if random.uniform(0, 1) < epsilon:
                action = random.choice(range(7))
            else:
                action = np.argmax(q_table[state]) # Exploit learned values
            
            move.move(action)
            next_state = 0
            sens_val, collision = sens.binary()
            for j in range(8):
                next_state+=sens_val[j]*2**(j)
            next_state+=256*prev_action
#            sc = sens.continuous()
#            for v in sc:
#                print("{0:.4f}".format(v),end=', ')
#            print('')
            #print(sc)
            #print(sens.strings())
            #print(next_state)
            #print(sens_val)
            reward = get_reward(action, sens_val)
                    
            
            done = True if (rob.getTime() > 90000 or collisions>2) else False 
            if collision:
                reward = -50;
                
            reward_total+=reward
            
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
    #rob = robobo.SimulationRobobo().connect(address='192.168.1.6', port=19997)
    rob = robobo.HardwareRobobo(camera=True).connect(address="192.168.1.7")
    move = motion.Motion(rob,False,speed=30,time=500)
   
    sens = irsensors.Sensors(rob,False)
    
    q_table = np.load(q_table_filename)
    
    #rob.play_simulation()

    init = False
    
    while not heardEnter():
        state = 0
        sens_val, collision = sens.binary()
        for j in range(8):
            state+=sens_val[j]*2**(j)
        action = np.argmax(q_table[state])
        print(sens_val)
        
        if not init:
            if not np.any(np.array(sens.continuous())>0.1):
                print('All sensors null')
                rob.talk('Touch me')
                rob.sleep(1)
                continue
            else:
                init = True
            
        move.move(action)

        
if __name__ == "__main__":
    train()
    #test('q_table590.npy')