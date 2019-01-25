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

class Genomes:
    def __init__(self, ip_adress):
        self.ip_adress = ip_adress
        self.rob = robobo.SimulationRobobo().connect(address=ip_adress, port=19997)
        
                # motion calss
        self.motions = motion.Motion(self.rob, is_simulation=True)
        # sensors class
        self.sensors = irsensors.Sensors(self.rob, is_simulation=True)
        # camera class
        self.camera = vision.Vision(self.rob, is_simulation=True, area_size=(1,5),downsampling_rate=5)
        # statistics
        self.stats = statistics.Statistics()  
        
        #robobo.HardwareRobobo(camera=True).connect(address="192.168.1.11")    
    def get_reward(self,action, prev_action, collision, collected_food, distance, \
                   distance_max, n_green, prev_n_green, prev_coordinate_sum, coordinate_sum):
        reward_collision = 0
        reward_distance = 0
        reward_food = 0
        reward_action = 0
        reward_green = 0
        reward_position = 0
        
        if coordinate_sum == prev_coordinate_sum:
            reward_position = -20
        
        if collision:
            reward_collision = -20
    
        if distance > distance_max:
            reward_distance = 20
        
        if (action == prev_action) and (action != 0):
            reward_action = -20
        
        if (prev_n_green >= n_green):
            reward_green = 20
        
        reward_food = 50*collected_food
            
        return rewards[action] + reward_collision + reward_distance \
                 + reward_action + reward_food + reward_green + reward_position
    
    def count_green(self,camera):
        green = 0
        for v, val in enumerate(camera):
            if val == 1:
                green+=1
        return green
    
    def save(self):
        self.stats.save('NEAT_progress/all_stats.json')
    
    
    def eval_genomes(self,genomes, config):
        self.stats.add_path()
        # dictionary fitness scores
        fitness_scores_dict = {}
    
        # genome = single neaural network
        for genome_id, genome in genomes:
            print('Genome: ',genome_id)
            # start simulation
            self.rob.play_simulation()
    
            # create random neural network
            net = neat.nn.recurrent.RecurrentNetwork.create(genome, config)
            
            #sensors_inputs = np.asarray(self.sensors.discrete())
            #photo = self.camera.color_per_area()
            sensors_inputs = [0]*8
            photo = [[0]*5]
    
            current_max_fitness = 0
            fitness_current = 0
            position_start = self.rob.position()
            prev_coordinates_sum = 0
            prev_position = position_start
            distance_max = 0
            prev_n_green = 0
            done = False
            previous_action = None
            n_collisions = 0
            collected_food = 0
            collected_food_prev = 0
            # time_passed = number of movements done
            time_passed = 0
            # number of times in which a movement didn't change the position
            stuck = 0
            collision = False
            while not done:            
    
                # give sensors input to the neural net
                nnOutput = net.activate(np.hstack((sensors_inputs, photo[0])))
                nn_choice = np.argmax(nnOutput)
    
                # perform action based on neural net output
                try:
                    self.motions.move(nn_choice)
                except:
                    done = True
                            
                previous_action = nn_choice
                
                read = False
                while read == False:           
                    try:
                        sensors_inputs = np.asarray(self.sensors.discrete())                    
                        read = True 
                    except:
                        print('waiting for sensors recording')
                
                read2 = False
                while read2 == False:
                    try:
                        photo,photo_binary = self.camera.color_per_area()
                        #camera_inputs = [1 if pixel=='G' else 0 for pixel in photo[0]]
                        n_green = np.sum(photo_binary[0])
                        read2 = True
                    except:
                        print('waiting for camera recording')
                        
                if (np.any(sensors_inputs==self.sensors.collision)) and (n_green == 0) and \
                    (collected_food==collected_food_prev):
                    n_collisions += 1
                    collision = True
                else:
                    collision = False
                
                read3 = False
                while read3 == False:
                    try:
                        position_current = np.asarray(self.rob.position())
                        read3 = True
                    except:
                        read3 = False
                        print('waiting for position recording')
                
                distance = dist(position_current,position_start)
                coordinates_sum = np.around(np.sum(position_current[0:2] - prev_position[0:2]),decimals=2)
                
                read4 = False
                collected_food_prev = collected_food
                while read4 == False:
                    try:
                        collected_food = self.rob.collected_food()
                        read4 = True
                    except:
                        read4 = False
                        print('waiting for food recording')
                        
                fitness_current += self.get_reward(nn_choice, previous_action, collision, \
                                              collected_food, distance, distance_max, \
                                              n_green, prev_n_green, prev_coordinates_sum, coordinates_sum)
                
    
                
                distance_max = np.max((distance_max,distance))
                
                if fitness_current > current_max_fitness:
                    current_max_fitness = fitness_current
                
                
                if  coordinates_sum == prev_coordinates_sum:
    #            if distance_max == distance:
                    stuck+=1
                prev_position = position_current
                prev_coordinates_sum = coordinates_sum
                
                if n_green > 0:
                    time_passed = time_passed - 3
                else:
                    time_passed += 3
                prev_n_green = n_green
                
                self.stats.add_continuous_point('{}_Collisions'.format(genome_id),n_collisions)
                self.stats.add_continuous_point('{}_Total Reward'.format(genome_id),fitness_current)
                self.stats.add_continuous_point('{}_Max Distance Achieved'.format(genome_id),distance_max)
                self.stats.add_continuous_point('{}_Food Eaten'.format(genome_id),collected_food)
                
                
                if (time_passed > 100):   
                    print('Stop cause: Stepd')
                    done = True
                elif (n_collisions > 2):
                    print('Stop cause: Collisions')
                    done = True   
                elif (time_passed > 75 and collected_food == 0):
                    print('Stop cause: Not quick enough')
                    done = True
                elif (stuck >= 4):
                    print('Stop cause: Stuck')
                    done = True
                    
                time_passed+=1
            
            genome.fitness = fitness_current
            fitness_scores_dict[genome_id] = fitness_current
            
            if fitness_current > 1500:
                filename = 'NEAT_progress/' +  str(genome_id)+'.pkl'
                with open(filename, 'wb') as output:
                    pickle.dump(net, output, 1)
            
            # stop simulation
            self.rob.stop_world()
            time.sleep(0.5)
        if heardEnter():
            return -1
        
if __name__ == "__main__":
    directory = 'NEAT_progress'
    if not os.path.exists(directory):
        os.makedirs(directory)

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     './config_neat.py')

    p = neat.Population(config)

    p.add_reporter(neat.StdOutReporter(True))
    
    gen = Genomes(ip_adress = '192.168.178.10')
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5,filename_prefix=directory + '/neat-checkpoint-'))

    # start training
    winner = p.run(gen.eval_genomes, 500)

    stats.save_genome_fitness(delimiter=',',filename = directory + '/fitness_history.csv')

    gen.save()

    with open(directory+'/final_winner.pkl', 'wb') as output:
        pickle.dump(winner, output, 1)



