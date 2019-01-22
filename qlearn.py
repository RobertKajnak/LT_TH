#!/usr/bin/env python2
from __future__ import print_function
import time
import robobo
import cv2
import numpy as np
import vision
import motion
import irsensors
import random
import select
import sys
import statistics

def heardEnter():
    i,o,e = select.select([sys.stdin],[],[],0.0001)
    for s in i:
        if s == sys.stdin:
            sys.stdin.readline()
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

def get_reward(action,vision, food_eaten_increased, collision):
    if food_eaten_increased:
        return 200;
    if collision:
        return -90;
    #going back and forwards is not a proper movement
    #elif (last_action<=1 and action==5) or ((last_action==2 or last_action==0) and action==6) or \
    #   (last_action==5 and action<=1) or (last_action==6 and (action==2 or action==0)):
    #       return -2
    elif np.any(vision) and action<=2:
        return rewards[action]*3
    else:
        return rewards[action]
        


def train(IP,is_simulation=True,starting_qtable_filename=None,stats_filename=None):
    '''
    params:
        starting_episode==None starts from scratch
        
    '''
    #initialize the robot
    if is_simulation:
        rob = robobo.SimulationRobobo().connect(address=IP, port=19997)
    else:
        rob = robobo.HardwareRobobo(camera=True).connect(address=IP)
    move = motion.Motion(rob,is_simulation,speed=25,time=500)
    sens = irsensors.Sensors(rob,is_simulation)
    see = vision.Vision(rob,is_simulation,(1,2),downsampling_rate=3)
    
    #set phone position:
    rob.set_phone_tilt(0.4, 10)
    
    #initialize stats
    if stats_filename is None:
        stats = statistics.Statistics()
    else:
        stats = statistics.Statistics(stats_filename)
    
    #initialize Q-table
    if starting_qtable_filename:
        q_table = np.load(starting_qtable_filename)
    else:
        q_table = np.zeros([2**8*7*2**2, 7])

    # Hyperparameters
    # Add adaptive parameters
    alpha_base = 0.4 #learning rate
    gamma = 0.7 #future reward
    epsilon = 0.4 #exploration
    
    time_limit = 500000
    max_iterations= 900
    halving = max_iterations/6
        
    target_color = 'G'
    
    # For plotting metrics
    stats.set_constant('gamma',gamma)   
    
    for i in range(0, max_iterations):
        #update hyperparams
        if i%halving==0 and i>0:
            epsilon/=2
        alpha = alpha_base*(0.03+float(max_iterations-i)/max_iterations)
        print('a={:.3f}, e={:.3f}'.format(alpha,epsilon))
        
        #initialize World
        rob.stop_world()
        time.sleep(3)
        rob.play_simulation()
    
        #initialize state variables
        state, collisions, reward = 0,0,0
        action = 0
        done = False
        
        #initialize statistics
        position_start = rob.position()
        food_eaten = rob.collected_food()
        reward_total,steps_survived,distance_max =  0, -1, 0
        y_max,x_max,y_min,x_min = position_start[1],position_start[0],position_start[1],position_start[0]
        position_current = position_start
        stats.add_path()
        
        while not done:
            #save last states
            action_prev = action
            state_prev = state
            
            #choose either best or random action
            if random.uniform(0, 1) < epsilon:
                action = random.choice(range(7))
            else:
                action = np.argmax(q_table[state_prev]) # Exploit learned values
            
            # so that it doesn't jump around
            if (action_prev>=5 and action<5) or (action_prev<5 and action>5):
                time.sleep(1.5/8)
                
            #perform action, read sensors
            move.move(action)
            state = 0
            sens_val, collision = sens.binary()
            photo = see.color_per_area()
            visual_object = [1 if pixel==target_color else 0 for pixel in photo[0]]
            
            food_eaten_last = food_eaten
            food_eaten = rob.collected_food()
            
            #calculate state
            for j in range(8):
                state+=sens_val[j]*2**(j)
            state+=(2**8)*action_prev
            for j in range(2):
                state+= (2**8)*7* 2**j * visual_object[j]
            
            #calculate reward
            reward = get_reward(action, visual_object,food_eaten-food_eaten_last,collision)             
            
            #update Q-table
            old_value = q_table[state_prev, action]
            next_max = np.max(q_table[state])
            
            new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
            q_table[state_prev, action] = new_value
       
            
            #calculate statistics for current step
            reward_total+=reward
            steps_survived+=1
            
            position_current = rob.position()
            stats.add_point_to_path(position_current)
            distance_max = np.max((distance_max,dist(position_current,position_start)))
            y_min = np.min((y_min,position_current[1]))
            y_max = np.max((y_max,position_current[1]))           
            x_min = np.min((x_min,position_current[0]))
            x_max = np.max((x_max,position_current[0]))
            
            collisions += int(collision and (food_eaten==food_eaten_last))
            
            #check for termination criteria            
            done = True if (rob.getTime() > time_limit or collisions>=3) else False 
            
            
        #Append current simulation statistics to total statistics
        stats.add_continuous_point('Collisions',collisions)
        stats.add_continuous_point('Total Reward',reward_total)
        stats.add_continuous_point('Steps Survived',steps_survived)
        stats.add_continuous_point('Max Distance Achieved',distance_max)
        stats.add_continuous_point('Alpha',alpha)
        stats.add_continuous_point('Epsilon',epsilon)
        stats.add_continuous_point('Food Eaten',food_eaten)
        max_area = (y_max-y_min)*(x_max-x_min)
        stats.add_continuous_point('Widest Area explored',max_area)
        for k in ['Collisions','Total Reward','Steps Survived',
                  'Max Distance Achieved','Alpa','Epsiolon','Widest Area explored',
                  'Food Eaten']:
            stats.set_xlabel(k,'Episode')
        stats.set_ylabel('Max Distance Achieved','m')
        stats.set_ylabel('Widest Area explored','m^2')
        #save controller
        np.save('tables/q_table{}'.format(i),q_table)
        
        #Display current progress
        print('Episode: {}, total rewards:{}, total steps:{}, max area:{:.2f}'.format(i,reward_total,steps_survived,max_area))
        
        #check for termination by user
        if heardEnter():
            break
    
    print("Training finished.\n")
    
    #save statistics
    stats.save('tables/all_stats.json')
    rob.stop_world()
    
    
def test(q_table_filename,IP,is_simulation=False):
    #initialize robot
    if is_simulation:
        rob = robobo.SimulationRobobo().connect(address=IP, port=19997)
    else:
        print('Connecting...')
        rob = robobo.HardwareRobobo(camera=True).connect(address=IP)
        print('Connected')
        
        
    rob.set_phone_tilt(0.4, 10)
    
    move = motion.Motion(rob,is_simulation,speed=30,time=500)
    sens = irsensors.Sensors(rob,is_simulation)
    see = vision.Vision(rob, False,(5,5),3)
    
    #load Q-table    
    q_table = np.load(q_table_filename)
    
    if is_simulation:
        rob.play_simulation()

    init = False
    action = 0
    action_prev = action
    target_color = 'G'
    
    img_counter = 0
    print('Starting action. Press Enter to stop')
    while not heardEnter():
        state = 0
        sens_val, collision = sens.binary()
        
        photo = see.color_per_area()
        visual_object = [1 if pixel==target_color else 0 for pixel in photo[0]]
        
        for j in range(8):
            state+=sens_val[j]*2**(j)
            
            #vision
        state+=(2**8)*action_prev
        for j in range(2):
            state+= (2**8)*7* 2**j * visual_object[j]
    
        action_prev = action
        action = np.argmax(q_table[state])
        
        #print sensor values for debug
        print(sens_val)
        #print(sens.continuous())
        print(see.color_per_area())
        #image = rob.get_image_front()
        #cv2.imwrite("test_pictures{}.png".format(img_counter),image)
        img_counter+=1
        
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
    #test('q_table552_vision_attempt1.npy','192.168.178.10', is_simulation = True )
    #test('q_table185_adaptive_2.npy','192.168.1.13')