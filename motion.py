import random

class Motion:
    
    def __init__(self,robot,is_simulation,speed=50,time=1000):
        self.rob = robot
        self.speed = speed
        self.time = time
        self.is_simulation = is_simulation
        
    def forward(self):
        self.rob.move(self.speed+8*int(not self.is_simulation), self.speed,self.time)
        
    def turn_right(self):
        self.rob.move(self.speed*2, self.speed/2,self.time)
    
    def turn_left(self):
        self.rob.move(self.speed/2, self.speed*2,self.time)
    
    def spin_right(self):
        self.rob.move(self.speed, -self.speed,self.time)
        
    def spin_left(self):
        self.rob.move(-self.speed, self.speed,self.time)
        
    def backwards(self):
        self.rob.move(-self.speed, -self.speed,self.time)

    def move(self, action_code):
        '''
        Moves based on the int provided:
            0 => forwards
            1 => turn right
            2 => turn left
            3 => spin right
            4 => spin left
            5 => go backwards
            -1 => randomly choose from [0,4], i.e. anything but backwards
            anything else => choose from [0,5] i.e. anything goes
        '''
        if action_code==0:
            self.forward()
        elif action_code==1:
            self.turn_right()
        elif action_code==2:
            self.turn_left()
        elif action_code==3:
            self.spin_right()
        elif action_code==4:
            self.spin_left()
        elif action_code==5:
            self.backwards()
        elif action_code==-1:
            self.move(random.choice(range(5)))
        else:
            self.move(random.choice(range(6)))