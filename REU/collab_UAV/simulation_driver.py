import numpy as np
import math

class simul:
        '''
        simplifying assumptions:
        
        1. Height does not exist, everything will be done in 2d'''



        
        def __init__(self, g, L_s, L_f, o_max, V, R_G, R_U):
            self.g = g #gbs locations
            self.L_s = L_s #UAV start points
            self.L_f = L_f #UAV end points
            self.o_max = o_max #max continous time disconnected
            self.U_t = L_s #UAV locations at time t
            self.V = V #max veloctiy
            self.c_U = len(L_s * [1]) #boolean list of UAV connectedness
            self.R_G = R_G #radial GBS -> UAV
            self.R_U = R_U #Radial UAV -> UAV
            self.t = 0

        def dist(o_1, o_2):
              '''
              o_1- object with x y coordinates in tuple
              returns distance between two objects'''
              return math.sqrt((o_1[0] - o_2[0])**2 + (o_1[1] - o_2[1])**2)
        
        def gbs_in_range(self, i):
              min_dist = min(self.dist(i, j) for j in self.g)
              return int(min_dist < self.R_G)

        
        def GBS_connect(self):
              ind = 0
              for drone in self.U_t:
                    self.c_U[ind] = self.gbs_in_range(drone)
                    ind = ind + 1

        def UAV_connect(self):
              for ind in range(len(self.U_t)):
                    if self.c_U[ind] == 1:
                          continue
                    for q in range(len(self.U_t)):
                          d = float('inf')
                          if q != ind:
                                d = self.dist(self.U_t[q], self.U_t[ind])
                          
                          if d < self.R_U:
                            self.c_U[ind] == 1
                            break
                    continue
                          
                          
                    
              


        def take_step(self, V, direction):
              '''
              V = list of velocities to fly at for each drone
              direction = list of directions to fly in that velocity
              '''
              x = 0
                    
                    
        def get_reward(self):
              x = 0
              #get the reward of the current state 
              # 
              # 
              # 
              # 
              # 

        #Example Initilization
        # L_s = [(10,10), (40,40)] #UAV start points
        # L_f = [(10,10), (40,40)] #UAV end points
        # o_max = 0 #maximum continuous time disconnected
        # U_t = U #set of UAV locations
        # V = 2 #max speed
        # c_U = [1,1] #boolean of UAV connections
        # R_G = 6 #Radial distance GBS to UAV
        # R_U = 3 #Radial distance GBS to UAV 
        #  
