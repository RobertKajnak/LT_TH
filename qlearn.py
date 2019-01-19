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

def heardEnter():
    i,o,e = select.select([sys.stdin],[],[],0.0001)
    for s in i:
        if s == sys.stdin:
            return True
    return False

def dist(a,b):
    return np.linalg.norm(np.asarray(a)-np.asarray(b))

rewards = {0:25,
           1:17,
           2:17,
           3:7,
           4:7,
           5:2,
           6:2}

def get_reward(action, last_action, collision):
    if collision:
        return -50;
    #going back and forwards is not a proper movement
    elif (last_action<=1 and action==5) or ((last_action==2 or last_action==0) and action==6) or \
       (last_action==5 and action<=1) or (last_action==6 and (action==2 or action==0)):
           return -2
    else:
        return rewards[action]
        

def train(IP,is_simulation=True,
          starting_qtable_filename=None, all_collisions_filename=None,all_rewards_filename=None,
          steps_survived_filename = None, furthest_distance_filename=None,all_positions_filename=None):
    '''
    params:
        starting_episode==None starts from scratch
        
    '''
    #initialize the robot
    if is_simulation:
        rob = robobo.SimulationRobobo().connect(address=IP, port=19997)
    else:
        rob = robobo.HardwareRobobo(camera=True).connect(address=IP)
    move = motion.Motion(rob,is_simulation,speed=30,time=500)
    sens = irsensors.Sensors(rob,is_simulation)
   
    #initialize Q-table
    if starting_qtable_filename:
        q_table = np.load(starting_qtable_filename)
    else:
        q_table = np.zeros([2**8*7, 7])

    # Hyperparameters
    alpha = 0.1
    gamma = 0.6
    epsilon = 0.1
    
    time_limit = 90000
    
    # For plotting metrics
    all_collisions = np.load(all_collisions_filename) if all_collisions_filename else []
    all_rewards = np.load(all_rewards_filename) if all_rewards_filename else []
    all_steps_survived = np.load(steps_survived_filename) if steps_survived_filename else []
    furthest_distance = np.load(furthest_distance_filename) if furthest_distance_filename else []
    all_positions = np.load(all_positions_filename) if all_positions_filename else []
    
    for i in range(1, 300):
        #initialize World
        rob.stop_world()
        time.sleep(3)
        rob.play_simulation()
    
        #initialize state variables
        next_state, collisions, reward = 0,0,0
        action = 0
        done = False
        
        #initialize statistics
        position_start = rob.position()
        reward_total,steps_survived,distance_max =  0, -1, 0
        position_current = position_start
        position_list = []
        
        while not done:
            #save last states
            prev_action = action
            state = next_state
            
            #choose either best or random action
            if random.uniform(0, 1) < epsilon:
                action = random.choice(range(7))
            else:
                action = np.argmax(q_table[state]) # Exploit learned values
            
            # so that it doesn't jump around
            if (prev_action>=5 and action<5) or (prev_action<5 and action>5):
                time.sleep(1.5)
                
            #perform action, read sensors and calculate state
            move.move(action)
            next_state = 0
            sens_val, collision = sens.binary()
            for j in range(8):
                next_state+=sens_val[j]*2**(j)
            next_state+=256*prev_action
            
            #calculate reward
            reward = get_reward(action, prev_action,collision)             
            
            #update Q-table
            old_value = q_table[state, action]
            next_max = np.max(q_table[next_state])
            
            new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
            q_table[state, action] = new_value
       
            
            #calculate statistics for current step
            reward_total+=reward
            steps_survived+=1
            
            position_current = rob.position()
            position_list.append(position_current)
            distance_max = np.max((distance_max,dist(position_current,position_start)))
            
            collisions += int(collision)
            
            #check for termination criteria            
            done = True if (rob.getTime() > time_limit or collisions>=1) else False 
            
            
        #Append current simulation statistics to total statistics
        all_collisions.append(0 if collision else 1)
        all_rewards.append(reward_total)
        all_steps_survived.append(steps_survived)
        furthest_distance.append(distance_max)
        all_positions.append(position_list)
        
        #save controller
        np.save('tables/q_table{}'.format(i),q_table)
        
        #Display current progress
        print('Episode: {}, total rewards:{}, total steps:{}, max distance:{:.2f}'.format(i,reward_total,steps_survived,distance_max))
        
        #check for termination by user
        if heardEnter():
            break
    
    print("Training finished.\n")
    
    #save statistics
    np.save('tables/all_rewards',all_rewards)
    np.save('tables/all_collisions',all_collisions)
    np.save('tables/all_steps_survived',all_steps_survived)
    rob.stop_world()
    
    
def test(q_table_filename,IP,is_simulation=False):
    #initialize robot
    if is_simulation:
        rob = robobo.SimulationRobobo().connect(address='192.168.178.10', port=19997)
    else:
        rob = robobo.HardwareRobobo(camera=True).connect(address="192.168.1.13")
    move = motion.Motion(rob,is_simulation,speed=30,time=500)
    sens = irsensors.Sensors(rob,is_simulation)

    #load Q-table    
    q_table = np.load(q_table_filename)
    
    if is_simulation:
        rob.play_simulation()

    init = False
    
    while not heardEnter():
        state = 0
        sens_val, collision = sens.binary()
        for j in range(8):
            state+=sens_val[j]*2**(j)
        action = np.argmax(q_table[state])
        
        #print sensor values for debug
        print(sens_val)
        
        #Initialize sensors to base values
        if not is_simulation and not init:
            if not np.any(np.array(sens.continuous())>0.1):
                print('All sensors null')
                rob.talk('Touch me')
                rob.sleep(1)
                continue
            elif not init:
                init = True
                time.sleep(1)
                rob.talk('Get away from me!')
                time.sleep(2)
                sens.set_sensor_baseline()
                rob.talk('Baseline set. Starting Patrol')
            
        move.move(action)
        
    if is_simulation:
        rob.stop_world()
        
if __name__ == "__main__":
    #'192.168.1.15'  '196.168.137.1'
    train('192.168.178.10')
    #test('q_table590_nofollow.npy')