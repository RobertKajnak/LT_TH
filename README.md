# LT_TH
This project aims to answer the fundamental question that has plagued humanity for centures: can 3 students create an AI that can eat the red or green box in 3 weeks?

#Work of Team H, Learning Machines 2019

Link to reports: https://drive.google.com/drive/folders/19A0C_Sktwt3vg7eEWrzz8kbWtgOBZrrM?usp=sharing


As discussed in the report we have two Q-learning approaches. The one from the last task, that was used live is the following:
The entry point for the Q-Learning algorithm is qlearn.py. The __main__ sections the training and test sets.  
irsensors.py, motion.py and vision.py contain the classes necessary for handling and preprocessing IR sensor, movement and camera data. These use the provided simulation.py and hardware.py files respectively.
For more detailed documentation please refer to the individual files.  

qlearn2.py and motion2.py are the version that we tested for this task, but ultimately had worse performance

neat_learner.py contains the entry point for the NEAT algorithm. In order to allow for early termination (e.g. max generations 300, but the simulation needs to be stopped at 100 gen for time or other reasons), the neat-python repo was forked and modified. The code will run with the vanilla version as well, but does not allow for early termination. 
The forked repo can be found under https://github.com/RobertKajnak/neat-python  

statistics.py can be used to save the progress while the algorithm is running and to plot the saved values and the path taken by the robot.

The simluation.py and hardware.py files were slightly modified to match up with simulation speed etc. These are under the robobo directory  

The scenes used for training can be found under https://drive.google.com/open?id=1AToF_O7k7pKyGDyfUUgz8lX5EzCR3zok

The repo for the code can be found under https://github.com/RobertKajnak/LT_TH  