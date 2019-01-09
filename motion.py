class Motion:
    def __init__(self,robot):
        self.rob = robot
    def forward(self):
        self.rob.move(20, 20,1000)
        
    def backwards(self):
        self.rob.move(-20,-20,1000)
    
    def right(self):
        self.rob.move(7, -7,1000)
        
    def left(self):
        self.rob.move(-7,7,1000)
