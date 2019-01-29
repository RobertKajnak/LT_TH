from collections import deque

class Sensors:
    far = 0
    close = 0.5
    collision = 1
    
    def __init__(self,robot, is_simulation):
        """Handles IR sensor for the robot. 
        Sensor order: back, from left to right, front from right to left
            Args:
                robot: robot returned from simulation.py or hardware.py
                is_simulation: True for V-REP, False for physical Robobo
        """
        self.rob = robot
        self.mem_length = 3
        self.is_simulation = is_simulation
        if not is_simulation:
            self.prev_vals =[deque([0]*self.mem_length,self.mem_length) for i in range(8)]
            self.touching = 1500
            self.thresholds =  [
                [25,200,self.touching],
                [25,200,self.touching],
                [30,150,self.touching],
                [25,150,self.touching],
                [50,150,self.touching],
                [15,220,self.touching],
                [30,150,self.touching],
                [25,160,self.touching],
                ]
        else:
            self.touching = 5
            self.thresholds =  [[0.01,0.15] for i in range(3)]+ \
                                [[0.05,0.15] for i in range(2)] +\
                                [[0.01,0.15] for i in range(1)] +\
                                [[0.05,0.15] for i in range(2)]
                
    def set_sensor_baseline(self, sensor_data=None):
        """Set current sensors values+10% as the comparison baseline for trigger"""
        cont = self.continuous(sensor_data)
        for idx,val in enumerate(cont):
            self.thresholds[idx][0] = val*1.1
    
    def continuous(self, sensor_data=None):
        """ Args:
                sensor_data: if None is specified rob.read_irs() is used
            Returns the values of the raw data from the sensors.
            if not is_simulation also adapts to blocked sensors
            Returns:
                in simulation: False if no detection, and distance from 0->0.2 in meters
                    e.g. [False,False,0.1,0.123 ...]
                physical: 0->10k+ inversely proportional to distance
                    [24,32,130,100,...]
        """
        if sensor_data is None:
            sens = self.rob.read_irs()
            
        if not self.is_simulation:
            for idx,val in enumerate(sens):
                self.prev_vals[idx].append(val)
                identical = True
                for j in range(self.mem_length-1):
                    if self.prev_vals[idx][j]!=self.prev_vals[idx][j+1]:
                        identical = False
                        break
                if identical:
                    self.thresholds[idx][0] = self.prev_vals[idx][0] *1.5
            
        return sens
    
    def discrete(self, sensor_data=None):
        """Calculated discretized values to 3 states: far, close, collision
        """
        if self.is_simulation:
            disc=[]
            sens = self.continuous(sensor_data)
            for idx,val in enumerate(sens):
                if val==False or val>=self.thresholds[idx][1]:
                    disc.append(self.far)
                elif val>=self.thresholds[idx][0]:
                    disc.append(self.close)
                else:
                    disc.append(self.collision)
                        
        else:
            disc=[]
            sens = self.continuous(sensor_data)
            for idx,val in enumerate(sens):
                if val<=self.thresholds[idx][0]:
                    disc.append(self.far)
                elif val<=self.thresholds[idx][1]:
                    disc.append(self.close)
                else:
                    disc.append(self.collision)
        return disc
        
    def one_hot(self, sensor_data=None):
        """ Deprecated. [1 0] for far, [0 1] for close, for each sensor
        """
        disc = self.discrete()
        hot = []
        hit=False
        for val in disc:
            hot+= [int(val==self.far),
                   int(val==self.close)]
            if val==self.collision:
                hit=True
        return hot,hit
    
    def binary(self, sensor_data=None):
        """ Reurns the binary form, i.e. is the object close or not
            Returns:
                hot: [1, 1, 0, ...] if the first two sensors detect an object.
                    far=> 0; close or collision=>1
                hit: True if any of the sensors detect collsion, otherwise false
        """
        disc = self.discrete(sensor_data)
        hot = []
        hit = False
        for val in disc:
            hot.append(int(val==self.close or val==self.collision))
            if val==self.collision:
                hit=True
        return hot, hit
    
    def strings(self, sensor_data=None):
        """ The values converted to strings
        Returns:
            e.g. ['far','far','close','collision'...]
        """
        disc = self.discrete(sensor_data)
        def toString(val):
            if val==self.far:
                return 'far'
            elif val==self.close:
                return 'close'
            else: 
                return 'collision'
        return list(map(toString, disc))