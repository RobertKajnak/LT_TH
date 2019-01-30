#!/usr/bin/env python2
from __future__ import print_function
import time
import robobo
import neat
import motion 
import irsensors
import vision
import numpy as np
import pickle 
import statistics
import select
import sys
import os
from os.path import isfile, join
from collections import deque

def dist(a,b):
    return np.linalg.norm(np.asarray(a)-np.asarray(b))

rewards = {0:8,
           1:8,
           2:8,
           3:5,
           4:5,
           5:2,
           6:2,
           7:2,
           8:0}
'''
Moves based on the int provided:
    0 => forwards
    1 => turn right
    2 => turn left
    3 => spin right
    4 => spin left
    5 => go back while turning right (the same right as in the front)
    6 => go back while turning left (the same left as in the front)
    7 => go backwards
    8 => randomly choose from [0,4], i.e. anything but backwards
    anything else => choose from [0,5] i.e. anything goes
'''


def heardEnter():
    i,o,e = select.select([sys.stdin],[],[],0.0001)
    for s in i:
        if s == sys.stdin:
            sys.stdin.readline()
            return True
    return False

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v / norm

def relu_activation(z):
    return z if z > 0.0 else 0.0

class Genomes:
    def __init__(self, ip_adress, input_type):
        """ type refers to the number of input values:
            15,12 or 3
        """
        self.ip_adress = ip_adress
        self.input_type = input_type
        
        self.rob = robobo.SimulationRobobo().connect(address=ip_adress, port=19997)
        
                # motion calss
        self.motions = motion.Motion(self.rob, is_simulation=True)
        # sensors class
        self.sensors = irsensors.Sensors(self.rob, is_simulation=True)
        # camera class
        if self.input_type==3:
            self.camera = vision.Vision(self.rob, is_simulation=True, area_size=(1,3),downsampling_rate=5)
        else:
            self.camera = vision.Vision(self.rob, is_simulation=True, area_size=(1,5),downsampling_rate=3)
            
        # statistics
        self.stats = statistics.Statistics()  
        
        self.mem_length = 4
        self.prev_vals = deque([0]*self.mem_length,self.mem_length) 
        #robobo.HardwareRobobo(camera=True).connect(address="192.168.1.11") 
    def _add_action_to_history(self,action):
        self.prev_vals.append(action)
        
    def _check_previous_actions(self):
        identical = True
        for j in range(self.mem_length-1):
            if self.prev_vals[j]!=self.prev_vals[j+1]:
                identical = False
                break
            
        if identical:
            return self.prev_vals[0]
        else:
            return -1

#    def get_reward(self,action, prev_action, collision, collected_food, distance, \
#                   distance_max, n_green, prev_n_green, movement_since_last_turn):
#        reward_collision = 0
#        reward_distance = 0
#        reward_food = 0
#        reward_action = 0
#        reward_green = 0
#        reward_position = 0
#        
#        if movement_since_last_turn <=0.05:
#            reward_position = -20
#        
#        if collision:
#            reward_collision = -20
#    
#        if distance > distance_max:
#            reward_distance = 20
#        
#        if (action == prev_action) and (action != 0):
#            reward_action = -20
#        
#        if (prev_n_green >= n_green):
#            reward_green = 20
#        
#        reward_food = 50*collected_food
#            
#        return rewards[action] + reward_collision + reward_distance \
#                 + reward_action + reward_food + reward_green + reward_position
#    
    def count_green(self,camera):
        green = 0
        for v, val in enumerate(camera):
            if val == 1:
                green+=1
        return green
    
    def save(self):
        self.stats.save('NEAT_progress/all_stats.json')
    
    def _attempt_read(self,read_function,error_message,max_attempts = 5, delay=1, arg_list = None):
        """ Executes the provided function under a try/except.
            Args:
                read_function: function to be executed
                error_message: message to be displayed
                max_attempts: maximum number of attempts before retrying
                delay: time to sleep between attempts in seconds
                arg_list: a list of arguments to be passed to read_function.
                    a value of None will call the fuction without arguments
            Returns:
                if succesful:
                    value returned by read_function, True
                otherwise:
                    None, False
        """
        for i in range(max_attempts):           
            try:
                if arg_list is None:
                    return read_function(),True
                else:
                    return read_function(*arg_list),True
            except:
                print(error_message + ': ' +  str(sys.exc_info()[0]) + ' attempt {}'.format(i))
                time.sleep(delay)
        return None, False
        
    
    def eval_genomes(self,genomes, config):
        self.stats.add_path()
        # dictionary fitness scores
        fitness_scores_dict = {}
    
        # genome = single neaural network
        for genome_id, genome in genomes:
            print('Genome: ',genome_id)
            # start simulation
            self.rob.play_simulation()
            #self.rob.visuals_off()
            # create random neural network
            net = neat.nn.recurrent.RecurrentNetwork.create(genome, config)
            
            #sensors_inputs = np.asarray(self.sensors.discrete())
            #photo = self.camera.color_per_area()
            sensors_inputs = [0]*8
            if self.input_type==3:
                photo = [0]*3
                photo_binary = [0]*3
            else:
                photo = [0]*5
                photo_binary=[0]*5
                photo_binary_prev = photo_binary
    
            fitness_current = 0
            position_start = self.rob.position()
            prev_position = position_start
            distance_max = 0
            prev_n_green = 0
            done = False
            previous_action = None
            n_collisions = 0
            collected_food = 0
            collected_food_prev = 0
            prev_green=0
            good_read = False
            end_early = False
            output_min = 1
            output_max = -1
            # time_passed = number of movements done
            time_passed = 0
            # number of times in which a movement didn't change the position
            #stuck = 0
            collision = False
            collision_prev = False
            time_limit = 50
            while not done:            
                # give sensors input to the neural net
                if self.input_type==15:
                    prev_action_type = -1 if previous_action >=5 else (1 if previous_action<=2 else 0)
                    nn_input = np.hstack((sensors_inputs, photo,prev_action_type,prev_green))
                elif self.input_type==12:
                    nn_input = np.hstack((sensors_inputs[1::2], photo_binary_prev[0],photo,photo_binary_prev[-1],prev_green))
                elif self.input_type==3:
                    nn_input = photo
                #print (nn_input)
                nnOutput = net.activate(nn_input)
                #print (nnOutput)
                
                if self.input_type ==12 or self.input_type == 15:
                    nn_choice = np.argmax(nnOutput)
                    self._add_action_to_history(nn_choice)
                    #reduce jumping behaviour
                    if (previous_action <=2 and nn_choice>2) or (nn_choice<=2 and previous_action>2) or\
                        (previous_action >=5 and nn_choice<5) or (nn_choice>=5 and previous_action<5) or\
                        (previous_action==3 and nn_choice==4) or (nn_choice==4 and previous_action==5):
                            time.sleep(0.8)
                            
                    # perform action based on neural net output
                    _,good_read = self._attempt_read(self.motions.move,\
                                            error_message='Could not perform motion',\
                                            arg_list=[nn_choice])
                    end_early |= not good_read
                    previous_action = nn_choice
                else:
                    #x = (nnOutput[0]-0.5) * 2# For activation functions [0,1] like sigmoid
                    x = nnOutput[0] #for activation functions [-1,1] like clapmed
                    x = np.min((np.max((x,-1)),1)) #just in case
                    output_min = np.min((x,output_min))
                    output_max = np.max((x,output_max))
                    
                    offset = 0.1 #to reduce oscillation
                    n = self.motions.speed #max speed
                    left = n*(1-2*relu_activation(x-offset)) #refer to documentation 
                    right = n*(1-2*relu_activation(-x+offset)) #for a graph of this logic
                    #self.rob.move(left,right,self.motions.time) 
                    
                    _,good_read = self._attempt_read(self.rob.move,\
                                    error_message='Could not perform motion',\
                                    arg_list=[left,right,self.motions.time])
                    end_early |= not good_read
                    
                    #Special forage-knowledge
                    if (np.sum(sensors_inputs[3:])>=2) and not prev_green:
                        #print('Avoiding Wall')
                        if np.sum(sensors_inputs[3:4]>np.sum(sensors_inputs[6:7])):
                            spin = 4
                        else:
                            spin = 3
                            _,good_read = self._attempt_read(self.motions.backwards,\
                                        error_message='Could not perform motion')
                            time.sleep(0.5)
                        for j in range(5):
                            camera_data,good_read = self._attempt_read(self.camera.color_per_area,'Could not use camera')
                            end_early |= not good_read
                            
                            if not good_read:
                                break;
                                
                            photo,photo_binary = camera_data[0][0], camera_data[1][0]
                            _,good_read = self._attempt_read(self.motions.move,\
                                        error_message='Could not perform motion',\
                                        arg_list=[spin])
                            
                            end_early |= not good_read
                            time.sleep(0.8)
                            if np.any(photo_binary) or not good_read:
                                break;
                        
    


                #Read sensors and other data
               # for i in range(20):#pseudo repeat-until or do-while structure
#                photo_prev = photo
#                sensors_inputs_prev = sensors_inputs
                
                sensors_inputs, good_read =self._attempt_read(self.sensors.discrete, 'IRS failed')
                sensors_inputs = np.asarray(sensors_inputs)  
                end_early |= not good_read

                camera_data,good_read = self._attempt_read(self.camera.color_per_area,'Could not use camera')
                end_early |= not good_read
                prev_green = int(np.any(photo_binary))
                photo_binary_prev = photo_binary
                
                #n_green = np.sum(photo_binary)
                
#                    if not(np.all(photo==photo_prev) and np.all(sensors_inputs==sensors_inputs_prev)\
#                           and not (np.any(sensors_inputs==self.sensors.collision)) ):
#                        break;
#                    else:
#                        time.sleep(0.02)
                
                position_current, good_read = self._attempt_read(self.rob.position, 'Could not obtain position')
                position_current = np.asanyarray(position_current)
                end_early |= not good_read

                collected_food_prev = collected_food
                collected_food, good_read = self._attempt_read(self.rob.collected_food, 'Could not obtain food data')
                end_early |= not good_read

                collision_prev = collision
                collision = np.any(sensors_inputs==self.sensors.collision)
                if not end_early:
                    photo,photo_binary = camera_data[0][0], camera_data[1][0]
                    photo = normalize(photo)
                    
                    if collision and collision_prev and\
                        (collected_food==collected_food_prev):
                        n_collisions += 1
#                        collision = True
#                    else:
#                        collision = False
                    PA = self._check_previous_actions()
                    if not (PA==-1 or PA==0):
                        fitness_current-=1
                    
                    distance = dist(position_current,position_start)
                    #coordinates_sum = np.around(np.sum(position_current[0:2] - prev_position[0:2]),decimals=2)
                    #dx = dist(position_current[0:2],prev_position[0:2])
                    
#                    fitness_current += self.get_reward(nn_choice, previous_action, collision, \
#                                                  collected_food, distance, distance_max, \
#                                                  n_green, prev_n_green, dx)
                    
        
                    
                    distance_max = np.max((distance_max,distance))
                    
#                    if fitness_current > current_max_fitness:
#                        current_max_fitness = fitness_current

#                    if  dx <= 0.05:
#                        stuck+=1
                        
                    #prev_position = position_current
                    
                    #if n_green > 0:
                    #    time_passed = time_passed - 3
                    #else:
                    #    time_passed += 3
                    #prev_n_green = n_green
                    
                    self.stats.add_continuous_point('Collisions'.format(genome_id),n_collisions)
                    self.stats.add_continuous_point('Total Reward'.format(genome_id),fitness_current)
                    self.stats.add_continuous_point('Max Distance Achieved'.format(genome_id),distance_max)
                    self.stats.add_continuous_point('Food Eaten'.format(genome_id),collected_food)
                    

                if not good_read:
                    cause = 'Sensor read failure. Simulating random heart attack'
                    done = True                
                elif (time_passed > time_limit):   
                    cause = 'Our of time'
                    done = True
                elif (n_collisions >= 4) and self.input_type!=3:
                    cause = 'Collisions'
                    done = True
                elif (collected_food == 9):
                    cause = 'All food eaten'
                    done = True
                time_passed+=1
            
            if self.input_type ==12 or self.input_type==15:
                genome.fitness = collected_food - n_collisions/2.0 + fitness_current/7.0
                print('Score={:.3f}, Food = {}, Spinning- = {:.3f}, Collision- = {:.3f}, Termination: {}'.format(\
                      genome.fitness, collected_food, fitness_current/7.0, n_collisions/2.0,cause) )
            elif self.input_type==3:
                food_bonus = collected_food
                time_penalty = 4.0*time_passed/time_limit
                variance_bonus = np.min((2.5*(output_max-output_min),4))
                genome.fitness = food_bonus + variance_bonus - time_penalty
                print('Score={:.3f}, Food = {}, Output delta = {:.3f}, Speed penalty = {:.3f}, Termination: {}'.format(\
                         genome.fitness, food_bonus, variance_bonus, time_penalty,cause) )
            fitness_scores_dict[genome_id] = fitness_current
            
            if fitness_current > 4:
                filename = 'NEAT_progress/' +  str(genome_id)+'.pkl'
                with open(filename, 'wb') as output:
                    pickle.dump(net, output, 1)
            self.save()
            # stop simulation
            self.rob.stop_world()
            time.sleep(0.5)
        if heardEnter():
            return -1

def train(IP,input_type):
    directory = 'NEAT_progress'
    if not os.path.exists(directory):
        os.makedirs(directory)

    onlyfiles = [f for f in os.listdir(directory) if isfile(join(directory, f))]
    latest_checpoint = 'neat-checkpoint--1'
    for fn in onlyfiles:
        if 'neat-checkpoint-' in fn:
            if int(fn[16:])>int(latest_checpoint[16:]):
                latest_checpoint = fn
            
    if int(latest_checpoint[16:])==-1:
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                 neat.DefaultSpeciesSet, neat.DefaultStagnation,
                 './config_neat.py')

        p = neat.Population(config)
    else:
        p = neat.Checkpointer.restore_checkpoint(directory + '/' +latest_checpoint)

    p.add_reporter(neat.StdOutReporter(True))
    
    gen = Genomes(ip_adress = IP,input_type = input_type)
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5,filename_prefix=directory + '/neat-checkpoint-'))

    # start training
    winner = p.run(gen.eval_genomes, 500)

    stats.save_genome_fitness(delimiter=',',filename = directory + '/fitness_history.csv')

    gen.save()

    with open(directory+'/final_winner.pkl', 'wb') as output:
        pickle.dump(winner, output, 1)

def test(IP,filename,filename_config,is_simulation=False):
    with open('final_winner.pkl', 'rb') as file:
        genome = pickle.load(file)
    
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
             neat.DefaultSpeciesSet, neat.DefaultStagnation,
             filename_config)
    net = neat.nn.recurrent.RecurrentNetwork.create(genome, config)    
    
    if is_simulation:
        rob = robobo.SimulationRobobo().connect(address=IP, port=19997)
    else:
        print('Connecting...')
        rob = robobo.HardwareRobobo(camera=True).connect(address=IP)
        print('Connected')
        
    if is_simulation:
        rob.set_phone_tilt(0.4, 10)
    else:
        rob.set_phone_tilt(109,20)
        rob.set_phone_pan(180,20)
        time.sleep(2)
    
    motions = motion.Motion(rob,is_simulation,speed=30,time=500)
    sensors = irsensors.Sensors(rob,is_simulation)
    camera = vision.Vision(rob, is_simulation,area_size=(1,5),downsampling_rate=5,\
                        target_color_min=(170,130,20),target_color_max=(180,255,255))
    camera.add_filter((0,130,20),(10,255,255))
    
    if is_simulation:
        rob.play_simulation()
    
    sensors_inputs = [0]*8
    photo = [0]*5
    photo_binary=[0]*5
    photo_binary_prev = photo_binary
    previous_action = 0
    prev_green = 0
            
    while not heardEnter():            
        # give sensors input to the neural net
        #prev_action_type = -1 if previous_action >=5 else (1 if previous_action<=2 else 0)

        #print(np.hstack((sensors_inputs, photo,prev_action_type,prev_green)))
        #nn_input = np.hstack((sensors_inputs, photo,prev_action_type,prev_green))
        nn_input = np.hstack((sensors_inputs[1::2], photo_binary_prev[0],photo,photo_binary_prev[-1],prev_green))
        nnOutput = net.activate(nn_input)
        nn_choice = np.argmax(nnOutput)    
        previous_action = nn_choice
        
        if nn_choice==0:
            nn_choice = 8
        elif nn_choice==7 or nn_choice==6:
            nn_choice = 0
        motions.move(nn_choice)    
        
        sensors_inputs = sensors.discrete()

        photo_binary_prev = photo_binary
        photo,photo_binary = camera.color_per_area()
        photo=normalize(photo[0])
        photo_binary = photo_binary[0]
        
        prev_green = int(np.any(photo_binary))
        photo = normalize(photo)
                     

    
    
if __name__ == "__main__":
    IP = '192.168.178.10'
    np.set_printoptions(precision=2)
    train(IP,input_type=3)
    #test(IP,filename='final_winner.pkl',filename_config='src/config_neat.py')

