import functools
import random
from copy import copy
import math


import numpy as np
from gymnasium.spaces import Discrete, MultiDiscrete

from pettingzoo.utils.env import ParallelEnv
import pygame

from copy import copy, deepcopy

'''
To implement:
-indices of which UAV's are collaborating'''



'''
The model 
-I currently find the index of the closest gbs in the state function, but I don't use the information yet

'''





'''
Simplifying Assumptions
1. Height does not exist- all drones and GBS are considered to be at the same height
2. Drones can collide, will set up maps so that optimal behavior would never include collisions, but I won't explicitly make it impossible in code
'''



''''
1. Just learn to do distance'''
class uav_collab(ParallelEnv):
    metadata = {
        "name": "uav_collaboration",
    }

    def __init__(self, g, L_s, L_f, o_max, V, R_G, R_U, grid_size, trunc_step):
        '''
        input data is assumed to be in the format of array of tuples ie
        g = [ (x_1,y_1), (x_2,y_2)...]'''


        self.g = g #gbs locations in an array of tuples
        self.L_s = L_s #UAV start points
        self.L_f = L_f#UAV end points
        self.o_max = o_max #max continous time disconnected
        self.U_t = deepcopy(self.L_s) #Drone positions at time t
        self.V = V #max veloctiy
        self.c_U = len(self.L_s) * [1] #boolean list of UAV connectedness
        self.R_G = R_G #radial GBS -> UAV
        self.R_U = R_U #Radial UAV -> UAV
        self.possible_agents = ["uav_1"]
        self.ts = 0
        self.grid_size = grid_size  #takes a single number showing the size of one axis ie 20 would mean a 20*20 grid
        self.trunc_step = trunc_step #step that you want the simulation to truncate and stop
        #self.using_out = using_out
        self.c_cont = copy(self.c_U)
        self.has_ended = False
        self.GBS_seen = {}
        for q in range(len(self.U_t)):
             self.GBS_seen[str(q)] = [0]
        








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
    
    def dist(self, o_1, o_2):
              '''
              o_1- object with x y coordinates in tuple
              returns distance between two objects'''
              return math.sqrt((o_1[0] - o_2[0])**2 + (o_1[1] - o_2[1])**2)
        
    def gbs_in_range(self, i):
        min_dist = float('inf')
        closest_index = None
        
        for idx, j in enumerate(self.g):
            distance = self.dist(i, j)
            if distance < min_dist:
                min_dist = distance
                closest_index = idx
        
        is_within_range = int(min_dist < self.R_G)
        return is_within_range, closest_index


        
    def GBS_connect(self):
            ind = 0
            for drone in self.U_t:
                self.c_U[ind], _ = self.gbs_in_range(drone)
                ind = ind + 1

    def UAV_connect(self):
            '''
            For sequential connections, add a while loop
            take the diff each time, if nothing changes, or if they all equal 1, break the loop
            yeah like take the diff and sum it, while diff > 0'''
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
              if self.U_t[ind] != self.L_f[ind]:
                   return False
              
         return True
    

    def mild_terminator(self, thresh):
         for ind in range(len(self.U_t)):
              if self.dist(self.U_t[ind],self.L_f[ind]) > thresh:
                   return False
              
         return True


    def check_fin(self):
         fin = 0
         for i in range(len(self.U_t)):
              if self.U_t[i] == self.L_f[i]:
                   fin += 2

         return fin
    
    def check_fin_mild(self, thresh):
         fin = 0
         for i in range(len(self.U_t)):
              if self.dist(self.U_t[i],self.L_f[i]) < thresh and not self.has_ended:
                   fin += 10
                   self.has_ended = True

         return fin
    

    def connec_cont(self):
         for i in range(len(self.U_t)):
              if self.c_U[i] == 1:
                   self.c_cont[i] = 0
              else:
                   self.c_cont[i] += 1



    def reward_cont(self):
        reward_lst = []
        for i in range(len(self.U_t)):
             curr = 0
             if self.c_cont[i] > self.o_max:
                  curr = -1
             reward_lst.append(curr)

        return reward_lst
             
    def reward_new_gbs(self):
        new_gbs_r = []
        for i in range(len(self.U_t)):
            curr_gbs_r = 0
            drone = self.U_t[i]
            ran, gbs = self.gbs_in_range(drone)
            if ran == 1:                    
                if gbs not in self.GBS_seen[str(i)]:
                    self.GBS_seen[str(i)].append(gbs)
                    curr_gbs_r = 50
            new_gbs_r.append(curr_gbs_r)
        return new_gbs_r
    

    def delta_dist(self,past):
         delta_dista = []
         for ind in range(len(self.U_t)):
              delt = 0
              prev = self.dist(past[ind], self.L_f[ind])
              curr = self.dist(self.U_t[ind], self.L_f[ind])
              delt = prev- curr #delta is positive if we got closer

              delta_dista.append(delt)


         return delta_dista

        








    '''
    Essential functions of the gym env class
    '''

    def reset(self, seed=0, options={"options": 1}):
        self.agents = copy(self.possible_agents)
        self.ts = 0
        self.c_U = len(self.L_s) * [1] #boolean list of UAV connectedness



        # non-random self.U_t = deepcopy(self.L_s) #Drone positions at time t
        #gb = random.randint(0, len(self.g)-2) #fix to -1, I'm just kind of cheating here
        #self.U_t = [deepcopy(self.g[gb]),]
        #print(self.U_t)

        #make thing happen in random corner of the map, so in the outer 5 places 


        for i in range(len(self.L_s)):
             #self.U_t[i] = (random.randint(*random.choice([(0, 5), (15, 20)])), random.randint(*random.choice([(0, 5), (15, 20)])))
             self.U_t[i] =( random.randint(0,20), random.randint(0,20))


        for i in range(len(self.L_s)):
             #self.L_f[i] = (20 - self.U_t[i][0], 20 - self.U_t[i][1])
             self.L_f[i] = ( random.randint(0,20), random.randint(0,20))
        self.c_cont = copy(self.c_U)
        self.has_ended = False
        self.GBS_seen = {}
        for q in range(len(self.U_t)):
             self.GBS_seen[str(q)] = [0]









        

        observations = {
            a:  self.U_t[self.agents.index(a)] + self.L_f[self.agents.index(a)]
            for a in self.agents
        }
        return observations, {}
    

   

    def step(self, actions):
        '''
        if action would take UAV off of grid, nothing happens
        '''
        past = copy(self.U_t)
        for ind in range(len(self.agents)):
            uav_action = actions[self.agents[ind]]

            if uav_action == 0 and self.U_t[ind][0] > 0:
                hold = list(self.U_t[ind])
                hold[0] -= 1
                self.U_t[ind] = tuple(hold)
            elif uav_action == 1 and self.U_t[ind][0] < self.grid_size:
                hold = list(self.U_t[ind])
                hold[0] += 1
                self.U_t[ind] = tuple(hold)
            elif uav_action == 2 and self.U_t[ind][1] > 0:
                hold = list(self.U_t[ind])
                hold[1] -= 1
                self.U_t[ind] = tuple(hold)
            elif uav_action == 3 and self.U_t[ind][1] < self.grid_size:
                hold = list(self.U_t[ind])
                hold[1] += 1
                self.U_t[ind] = tuple(hold)

             

        self.GBS_connect()
        curr = np.array(self.c_U)
        self.UAV_connect()
        delta = np.array(self.c_U)
        chang = np.sum(delta - curr) 

        '''simple np.where to check which indices involved in collaboration'''

        while chang > 0:
             curr = np.array(self.c_U)
             self.UAV_connect
             delta = np.array(self.c_U)
             chang = np.sum(delta - curr)

        gbs_new = self.reward_new_gbs()

        
            

             


        self.connec_cont()
        cont_r = self.reward_cont()
        d_dis = self.delta_dist(past)







        infos = {a: {} for a in self.agents}
        truncations = {a: False for a in self.agents}
        terminations = {a: False for a in self.agents}
        observations = {
            a: self.U_t[self.agents.index(a)] + self.L_f[self.agents.index(a)]
            for a in self.agents
        }






        #j = self.terminator()

        #attempting to train with all episodes being the same length, no termination condition
        
        j = self.mild_terminator(self.R_G)
        if j:
             terminations = {a: True for a in self.agents}
             rewards = {a:  25 for a in self.agents}
             return observations, rewards, terminations, truncations, infos


        if self.ts > self.trunc_step:
            rewards = {a: 0 for a in self.agents}
            truncations = {a: True for a in self.agents}
            self.agents = []
            return observations, rewards, terminations, truncations, infos
            
        
        rewards = {}
        for ind in range(len(self.agents)):
             rewards[self.agents[ind]] = 5*d_dis[ind] - 1#+ 4*cont_r[ind]  -1 # gbs_new[ind] #  pscp -r C:\Users\JackF\Documents\REU\uav_environment\env\ farleyj3@tesla.cs.vcu.edu:/home/farleyj3/rl -.1*(self.dist(self.U_t[ind], self.L_f[ind]))  (-3 * self.c_U[ind]) #self.dist(self.U_t[ind], self.L_s[ind]) #(0 * self.c_U[ind])   
        self.ts += 1
        print(observations)
        #print(self.U_t)
        #print(d_dis[0])
        #print(cont_r[0])



        
        return observations, rewards, terminations, truncations, infos

    def render(self):
        grid = np.zeros((self.grid_size, self.grid_size))
        for ind in range(len(self.possible_agents)):
             grid[self.U_t[ind][0], self.U_t[ind][1]] = str(ind)
        # add gbs render- with radius
        print(f"{grid} \n")

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return MultiDiscrete( [self.grid_size] * 4)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(5)
        #four directions and standing still





"""   
j = self.mild_terminator(self.R_G)
if j == True:
        q = 1
else:
        q = 0

"""


   