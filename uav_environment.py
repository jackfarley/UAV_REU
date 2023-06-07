import functools
import random
from copy import copy
import math

import numpy as np
from gymnasium.spaces import Discrete, MultiDiscrete

from pettingzoo.utils.env import ParallelEnv


'''
To implement:
-Implement current GBS connection as part of observation, if now closer to a new GBS, give some reward (increases exploration of new GBS routes)
-Implement is collarborating (could just be the difference between c after GBS connection and after UAV connection)
give reward to GBS systems that collaborate (encourages collaboration)'''





'''
Simplifying Assumptions
1. Height does not exist- all drones and GBS are considered to be at the same height
2. Drones can collide, will set up maps so that optimal behavior would never include collisions, but I won't explicitly make it impossible in code
'''

class uav_collab(ParallelEnv):
    metadata = {
        "name": "custom_environment_v0",
    }

    def __init__(self, g, L_s, L_f, o_max, V, R_G, R_U, grid_size, trunc_step):
        '''
        input data is assumed to be in the format of array of tuples ie
        g = [ (x_1,y_1), (x_2,y_2)...]'''


        self.g = g #gbs locations in an array of tuples
        self.L_s = L_s #UAV start points
        self.L_f = L_f #UAV end points
        self.o_max = o_max #max continous time disconnected
        self.U_t = L_s #Drone positions at time t
        self.V = V #max veloctiy
        self.c_U = len(self.L_s * [1]) #boolean list of UAV connectedness
        self.R_G = R_G #radial GBS -> UAV
        self.R_U = R_U #Radial UAV -> UAV
        self.possible_agents = ["uav_1", "uav_2"]
        self.ts = 0
        self.grid_size = grid_size  #takes a single number showing the size of one axis ie 20 would mean a 20*20 grid
        self.trunc_step = trunc_step #step that you want the simulation to truncate and stop








    '''
    below are helper functions I use
    '''

    def tu_form(self, arr):
        emp = arr[0]
        for ind in range(len(arr)):
            if ind == 0:
                continue
            else:
                emp = emp + arr[ind]


        return emp
    
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


    def terminator(self):
         for ind in range(len(self.U_t)):
              if self.U_t[ind] != self.L_f:
                   return False
              
         return True


    








    '''
    Essential functions of the gym env class
    '''

    def reset(self):
        self.agents = copy(self.possible_agents)
        self.ts = 0
        self.c_U = len(self.L_s * [1]) #boolean list of UAV connectedness
        self.U_t = self.L_s #Drone positions at time t




        observations = {
            a: self.tu_form(self.U_t) + self.tu_form(self.L_f) + self.tu_form(self.g)  + self.tu_form(self.c_U)
            for a in self.agents
        }
        return observations, {}
    

   

    def step(self, actions):
        '''
        if action would take UAV off of grid, nothing happens
        '''
        for ind in range(len(self.agents)):
            uav_action = actions[self.agents[ind]]

            if uav_action == 0 and self.U_t[ind][0] > 0:
                self.U_t[ind][0] -= 1
            elif uav_action == 1 and self.U_t[ind][0] < self.grid_size:
                self.U_t[ind][0] += 1
            elif uav_action == 2 and self.U_t[ind][1] > 0:
                self.U_t[ind][1] -= 1
            elif uav_action == 3 and self.U_t[ind][1] < self.grid_size:
                self.U_t[ind][1] += 1


             

        self.GBS_connect()
        self.UAV_connect()





        infos = {a: {} for a in self.agents}
        truncations = {a: False for a in self.agents}
        terminations = {a: False for a in self.agents}
        observations = {
            a: self.tu_form(self.U_t) + self.tu_form(self.L_f) + self.tu_form(self.g)  + self.tu_form(self.c_U)
            for a in self.agents
        }






        j = self.terminator()
        if j:
             terminations = {a: True for a in self.agents}
             rewards = {a: 100 for a in self.agents}
             return observations, rewards, terminations, truncations, infos

        
        
        if self.timestep > (self.grid_size * 10):
            rewards = {a: 0 for a in self.agents}
            truncations = {a: True for a in self.agents}
            self.agents = []
            return observations, rewards, terminations, truncations, infos
            
        
        rewards = {}
        for ind in range(len(self.agents)):
             rewards[self.agents[ind]] = -1 + (-3 * self.c_U[ind])
             
             
        self.timestep += 1

        

        return observations, rewards, terminations, truncations, infos

    def render(self):
        grid = np.zeros((self.grid_size, self.grid_size))
        for ind in range(len(self.possible_agents)):
             grid[self.U_t[ind][0], self.U_t[ind][1]] = str(ind)
        print(f"{grid} \n")

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return MultiDiscrete([self.gridsize] * 4 * len(self.U_t) + [self.gridsize]* 2 * len(self.g) + [2] * len(self.U_t))

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(4)