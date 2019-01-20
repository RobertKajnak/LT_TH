#!/usr/bin/env python2
from __future__ import print_function
import time
import robobo
import neat
import motion 
import irsensors
import numpy as np
import pickle 

rewards = {0:10,
           1:2,
           2:2,
           3:2,
           4:2,
           5:2}

def reward(action, sensors, previous_action):

    if (action==previous_action) and (action!=0):
        reward_action = rewards[action]*(-10)

    elif ((sensors[4] <= 0.15)and(sensors[4] > 0.05)) or ((sensors[-1] <= 0.15)and(sensors[-1] > 0.05)): 
        reward_action = rewards[action]*3

    else:
        reward_action = rewards[action]

    return reward_action 

def sensors_inversion(sensors):
    for index, item in enumerate(sensors):
        if item == 0:
            items[index] = 1

    return sensors

def quaternions_angle(quaternions_prev, quaternions_current):
    return 1/np.cos((2*(np.sum((quaternions_prev+quaternions_current))**2))-1)

def quaternions_distance(quaternions_prev, quaternions_current):
    return 1 - (np.sum((quaternions_prev+quaternions_current))**2)

def eval_genomes(genomes, config):

    connection=False
    while connection == False:
        try: 
            time.sleep(0.5)
            rob = robobo.SimulationRobobo().connect(address='192.168.1.15', port=19997)
            #robobo.HardwareRobobo(camera=True).connect(address="192.168.1.66")    
            connection=True
        except:
            connection=False
    # motion calss
    motions = motion.Motion(rob, is_simulation=True)
    # sensors class
    sensors = irsensors.Sensors(rob, is_simulation=True)
    # dictionary fitness scores
    fitness_scores_dict = {}

    # genome = single neaural network
    for genome_id, genome in genomes:
        print(genome_id)
        # start simulation
        time.sleep(0.5)
        rob.play_simulation()

        # create random neural network
        net = neat.nn.recurrent.RecurrentNetwork.create(genome, config)
        
        read = False
        while read == False:
            try:
                time.sleep(0.5)
                sensors_inputs = np.asarray(sensors.continuous()).astype(type('float', (float,), {}))
                sensors_inputs = np.asarray([1 if x==0 else x for x in sensors_inputs])
                read = True
            except:
                rob.stop_world()
                time.sleep(3)
                rob.play_simulation()

        current_max_fitness = 0
        fitness_current = 0
        #initial_position = np.asarray(rob.position())
        #initial_quaternions = rob.quaternions()

        done = False
        previous_action = None
        while not done:

            #print(rob.read_blocks())
            #print(sensors_inputs)
            #print(rob.orientation())
            #current_quaternions = rob.quaternions()
            #print(quaternions_distance(initial_quaternions,current_quaternions))

            # get time and proximity sensors values
            time_passed = rob.getTime()

            if (time_passed > 120000) or (np.any(sensors_inputs[3:] <= 0.05)) or (np.any(sensors_inputs[:2] <= 0.01)):
                done = True
                #print(np.any(np.array(sensors_inputs[3:]) <= 0.05))
                #print(genome_id, fitness_current)            

            # give sensors input to the neural net
            nnOutput = net.activate(sensors_inputs)
            nn_choice = np.argmax(nnOutput)
            #print('nn_choice', nn_choice)

            # perform action based on neural net output
            #try:
            motions.move(nn_choice)
            #except:
             #   done = True

            #print(rob.read_blocks())
            #current_position = np.asarray(rob.position())
            #distance = np.sqrt(np.sum(current_position-initial_position)**2)
             
            fitness_current += reward(nn_choice, sensors_inputs, previous_action)
            #print('fitness_current', fitness_current)
            if fitness_current > current_max_fitness:
                current_max_fitness = fitness_current
                        
            previous_action = nn_choice
            #previous_distance = distance
            genome.fitness = fitness_current

            # update sensors inputs
            time.sleep(0.5)

            #try:
            time.sleep(0.5)
            sensors_inputs = np.asarray(sensors.continuous()).astype(type('float', (float,), {}))
            sensors_inputs = np.asarray([1 if x==0 else x for x in sensors_inputs])
        #except:
            #done = True

        fitness_scores_dict[genome_id] = fitness_current
        print("Robobo collected {} food".format(rob.collected_food()))
        # stop simulation
        rob.stop_world()
        time.sleep(1)


if __name__ == "__main__":

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     './src/config_neat.py')

    p = neat.Population(config)

    print(p)

    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(10))

    # start training
    winner = p.run(eval_genomes, 10)

    stats.save_genome_fitness(delimiter=',')

#    visualize.draw_net(config, winner, True, node_names=node_names)
#    visualize.plot_stats(stats, ylog=False, view=True)
#    visualize.plot_species(stats, view=True)

    with open('final_winner.pkl', 'wb') as output:
        pickle.dump(winner, output, 1)



