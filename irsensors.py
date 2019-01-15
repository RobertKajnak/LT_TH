class Sensors:
    out = 0
    close = 0.5
    collision = 1
    
    def __init__(self,robot, is_simulation):
        self.rob = robot
        self.is_simulation = is_simulation
        if not is_simulation:
            self.touching = 1500
            self.thesholds =  [
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
            self.thesholds =  [[0.05,0.17] for i in range(3)]+ \
                                [[0.05,0.17] for i in range(5)]
                
        
    def continuous(self):
        return self.rob.read_irs()
    
    def discrete(self):
        if self.is_simulation:
            disc=[]
            sens = self.continuous()
            for idx,val in enumerate(sens):
                if val==False or val>=self.thesholds[idx][1]:
                    disc.append(self.out)
                elif val>=self.thesholds[idx][0]:
                    disc.append(self.close)
                else:
                    disc.append(self.collision)
                        
        else:
            disc=[]
            sens = self.continuous()
            for idx,val in enumerate(sens):
                if val<=self.thesholds[idx][0]:
                    disc.append(self.out)
                elif val<=self.thesholds[idx][1]:
                    disc.append(self.close)
                else:
                    disc.append(self.collision)
        return disc
        
    def one_hot(self):
        disc = self.discrete()
        hot = []
        hit=False
        for val in disc:
            hot+= [int(val==self.out),
                   int(val==self.close)]
            if val==self.collision:
                hit=True
        return hot,hit
    
    def binary(self):
        disc = self.discrete()
        hot = []
        hit = False
        for val in disc:
            hot.append(int(val==self.close))
            if val==self.collision:
                hit=True
        return hot, hit
    
    def strings(self):
        disc = self.discrete()
        def toString(val):
            if val==self.out:
                return 'out'
            elif val==self.close:
                return 'close'
            else: 
                return 'collision'
        return list(map(toString, disc))