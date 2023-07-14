import gymnasium as gym

# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/dqn/#dqnpy
import argparse
import os
import random
import time
from distutils.util import strtobool





import functools
import random
from copy import copy
import math


import numpy as np
from gymnasium.spaces import Discrete, MultiDiscrete

from gymnasium import Env


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
class uav_collab(Env):

    def __init__(self, g, L_s, L_f, o_max, V, R_G, R_U, grid_size, trunc_step, num_episodes):
        '''
        input data is assumed to be in the format of np arrays'''

        super().__init__()
        self.g = g #gbs locations in np array

        self.L_s = L_s #UAV start point
        self.L_f = L_f#UAV end point
        self.o_max = o_max #max continous time disconnected
        self.U_t = deepcopy(self.L_s) #Drone positions at time t
        self.V = V #max veloctiy
        self.total_agents = 1
        self.c_U = np.array([1]) #boolean list of UAV connectedness

        self.c_cont = copy(self.c_U)
        self.R_G = R_G #radial GBS -> UAV
        self.R_U = R_U #Radial UAV -> UAV
        self.ts = 0
        self.global_episodes = 0
        self.total_epi = num_episodes
        self.grid_size = grid_size  #takes a single number showing the size of one axis ie 20 would mean a 20*20 grid
        self.trunc_step = trunc_step #step that you want the simulation to truncate and stop

        self.has_ended = False
        self.speed_pen = -2
        self.last_safe = deepcopy(self.U_t)
        self.wall_pen = -3
        

        self.speed_mat = np.linspace(0,self.speed_pen, self.total_epi + 5)


        self.GBS_seen = [0]




        self.observation_space = MultiDiscrete( [self.grid_size] * 3)
        self.action_space = Discrete(5)

        








    '''
    below are helper functions I use
    '''

    
    def dist(self, o_1, o_2):
              '''
              o_1- object with x y coordinates in np arrays
              returns distance between two objects'''
              return math.sqrt((o_1[0] - o_2[0])**2 + (o_1[1] - o_2[1])**2)
        
    def gbs_in_range(self):
        min_dist = float('inf')
        closest_index = None
        
        for i in range(int(len(self.g)/2)):
            distance = self.dist(self.g[2*i:(2*i)+2], self.U_t)
            if distance < min_dist:
                min_dist = distance
                closest_index = i
        
        is_within_range = int(min_dist < self.R_G)
        return is_within_range, closest_index

    


    def terminator(self):
         for ind in range(len(self.U_t)):
              if self.U_t[ind] != self.L_f[ind]:
                   return False
              
         return True
    

    def mild_terminator(self, thresh):

        if self.dist(self.U_t,self.L_f) > thresh:
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

         if self.dist(self.U_t,self.L_f) < thresh and not self.has_ended:
            fin += 10
            self.has_ended = True

         return fin
    

    def connec_cont(self):

            if self.c_U[0] == 1:
                self.c_cont[0] = 0
            else:
                self.c_cont[0] += 1


             
    def reward_new_gbs(self):

        curr_gbs_r = 0
        drone = self.U_t
        ran, gbs = self.gbs_in_range(drone)
        if ran == 1:                    
            if gbs not in self.GBS_seen:
                self.GBS_seen.append(gbs)
                curr_gbs_r = 1

        return curr_gbs_r
    

    def delta_dist(self,past):

        delt = 0
        prev = self.dist(past, self.L_f)
        curr = self.dist(self.U_t, self.L_f)
        delt = prev- curr #delta is positive if we got closer


        return delt

        








    '''
    Essential functions of the gym env class
    '''

    def reset(self, seed=0): #, seed=0, options={"options": 1}
        self.ts = 0
        self.c_U = np.array([1])  #boolean list of UAV connectedness

        self.c_cont = copy(self.c_U)



        self.U_t = deepcopy(self.L_s) #Drone positions at time t
        #will need to fix this since input is different now #gb = random.randint(0, len(self.g)-2) #fix to -1, I'm just kind of cheating here
        #self.U_t = [deepcopy(self.g[gb]),]

        self.last_safe = deepcopy(self.U_t)
        #print(self.U_t)

        #make thing happen in random corner of the map, so in the outer 5 places 


        #for i in range(len(self.L_s)):
             #self.U_t[i] = (random.randint(*random.choice([(0, 5), (15, 20)])), random.randint(*random.choice([(0, 5), (15, 20)])))
         #    self.U_t[i] =( random.randint(0,20), random.randint(0,20))


        #for i in range(len(self.L_s)):
             #self.L_f[i] = (20 - self.U_t[i][0], 20 - self.U_t[i][1])
         #    self.L_f[i] = ( random.randint(0,20), random.randint(0,20))
        self.has_ended = False
       
        self.GBS_seen= [0]

        print(self.global_episodes)









        

        observation = np.concatenate((self.U_t, self.c_cont))

        info = {}
   
        return observation, info
    

   

    def step(self, action):
        '''
        if action would take UAV off of grid, nothing happens
        '''

        reward = None

        if self.c_cont[0] == 0:
             self.last_safe = deepcopy(self.U_t)
        past = deepcopy(self.U_t)

        uav_action = action

        if uav_action == 0 and self.U_t[0] > 0:

            self.U_t[0] -= 1

        elif uav_action == 1 and self.U_t[0] < self.grid_size:

            self.U_t[0] += 1

        elif uav_action == 2 and self.U_t[1] > 0:

            self.U_t[1] -= 1

        elif uav_action == 3 and self.U_t[1] < self.grid_size:

            self.U_t[1] += 1



        d_dis = self.delta_dist(past)


        self.c_U[0], _ = self.gbs_in_range()

        self.connec_cont()


        
        if self.c_cont[0] > self.o_max:
             self.U_t = deepcopy(self.last_safe)
             self.c_cont[0]= 0
             reward = self.wall_pen
             
        







        info = {}

        done= False 
        observation = np.concatenate((self.U_t, self.c_cont))




        j = self.mild_terminator(self.R_G)
        if j:
             done = True
             reward = 100
             self.global_episodes += 1
             return observation, reward, done, done, info
        
    
             


        if self.ts > self.trunc_step:
            done = True
            reward = 0
            self.global_episodes += 1
            return observation, reward, done, done, info 
            
        if reward == None:

            reward = d_dis - 1#+  self.speed_mat[self.global_episodes]#+ 4*cont_r[ind]  -1 # gbs_new[ind] #  pscp -r C:\Users\JackF\Documents\REU\uav_environment\env\ farleyj3@tesla.cs.vcu.edu:/home/farleyj3/rl -.1*(self.dist(self.U_t[ind], self.L_f[ind]))  (-3 * self.c_U[ind]) #self.dist(self.U_t[ind], self.L_s[ind]) #(0 * self.c_U[ind])   

        
        if self.global_episodes % 100 ==0:
             print(observation)
        self.ts += 1
        #print(d_dis[0])
        #print(cont_r[0])



        
        return observation, reward, done, done, info

    def render(self):
        print(0)


    def close(self):
        pass

  



from copy import copy, deepcopy

import supersuit as ss


import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
from torch.distributions.categorical import Categorical
import math


from stable_baselines3 import DQN, PPO



from distutils.util import strtobool
import argparse

def parse_args():
    # system arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default="out_two",
        help="the name of this experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="PPO_trunc_outage",
        help="the wandb's project name")
    parser.add_argument("--save_model", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="whether to save model into the `runs/{run_name}` folder")
    

    #Environment arguments
    parser.add_argument("--g", type=list, default= np.array([ 2,6, 2,10 , 2,14, 6,14, 10,14]),
                    help="list of gbs locations")
    parser.add_argument("--L_s", type=list, default= np.array([2,6]),
                        help="list of drone starting locations")
    parser.add_argument("--L_f", type=list, default=np.array([10,14]),
                        help="list of drone end locations")
    parser.add_argument("--o_max", type=int, default=2,
                        help="outage constraint")
    parser.add_argument("--V", type=int, default=1,
                        help="max veloctiy")
    parser.add_argument("--R_G", type=int, default=2.2,
                        help="Radius of connection from GBS-> UAV")
    parser.add_argument("--R_U", type=int, default=3,
                        help="Radius of connection from UAV-> UAV")
    parser.add_argument("--grid_size", type=int, default=20,
                        help="length of each grid axis (gris is a square)")
    parser.add_argument("--total_timesteps", type=int, default=75,
        help="total timesteps of the experiments")
    parser.add_argument("--act_len", type=int, default=5,
        help="length of actions")
    
    parser.add_argument("--total_episodes", type=int, default=20000,
                        help="total number of episodes to run")
    
    
    args = parser.parse_args()
    return args

    



import par_env


if __name__ == "__main__":



    args = parse_args()

    env = uav_collab( args.g, args.L_s, args.L_f, args.o_max, args.V, args.R_G, args.R_U, args.grid_size, args.total_timesteps, args.total_episodes)




    model = DQN("MlpPolicy", env)
    model.learn(total_timesteps=1000000, log_interval=4)
    model.save("testing_sb")

#del model # remove to demonstrate saving and loading

#model = DQN.load("testing_sb")

#obs, info = env.reset()

'''
q=0
while q < 1000:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
    q +=1 


print('done')

'''