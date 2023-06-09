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
        self.L_f = L_f #UAV end points
        self.o_max = o_max #max continous time disconnected
        self.U_t = deepcopy(self.L_s) #Drone positions at time t
        self.V = V #max veloctiy
        self.c_U = len(self.L_s) * [1] #boolean list of UAV connectedness
        self.R_G = R_G #radial GBS -> UAV
        self.R_U = R_U #Radial UAV -> UAV
        self.possible_agents = ["uav_1", "uav_2"]
        self.ts = 0
        self.grid_size = grid_size  #takes a single number showing the size of one axis ie 20 would mean a 20*20 grid
        self.trunc_step = trunc_step #step that you want the simulation to truncate and stop
        #self.using_out = using_out








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
                   fin += 15

         return fin
    
    def check_fin_mild(self, thresh):
         fin = 0
         for i in range(len(self.U_t)):
              if self.dist(self.U_t[i],self.L_f[i]) < thresh:
                   fin += 15

         return fin
    








    '''
    Essential functions of the gym env class
    '''

    def reset(self, seed=0, options={"options": 1}):
        self.agents = copy(self.possible_agents)
        self.ts = 0
        self.c_U = len(self.L_s) * [1] #boolean list of UAV connectedness
        self.U_t = deepcopy(self.L_s) #Drone positions at time t


        

        observations = {
            a: (self.agents.index(a),) + self.tu_form(self.U_t) + self.tu_form(self.L_f) + self.tu_form(self.g)  + tuple(self.c_U)
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






        infos = {a: {} for a in self.agents}
        truncations = {a: False for a in self.agents}
        terminations = {a: False for a in self.agents}
        observations = {
            a: (self.agents.index(a),) + self.tu_form(self.U_t) + self.tu_form(self.L_f) + self.tu_form(self.g)  + tuple(self.c_U)
            for a in self.agents
        }






        #j = self.terminator()
        
        j = self.mild_terminator(4)
        if j:
             terminations = {a: True for a in self.agents}
             rewards = {a: 1000 for a in self.agents}
             return observations, rewards, terminations, truncations, infos

        
        
        if self.ts > self.trunc_step:
            rewards = {a: 0 for a in self.agents}
            truncations = {a: True for a in self.agents}
            self.agents = []
            return observations, rewards, terminations, truncations, infos
            
        
        rewards = {}
        for ind in range(len(self.agents)):
             rewards[self.agents[ind]] = -(self.dist(self.U_t[ind], self.L_f[ind])) + self.check_fin_mild(2) #(-3 * self.c_U[ind]) #self.dist(self.U_t[ind], self.L_s[ind]) #(0 * self.c_U[ind])   
        self.ts += 1
        print(self.U_t)

        
        return observations, rewards, terminations, truncations, infos

    def render(self):
        grid = np.zeros((self.grid_size, self.grid_size))
        for ind in range(len(self.possible_agents)):
             grid[self.U_t[ind][0], self.U_t[ind][1]] = str(ind)
        # add gbs render- with radius
        print(f"{grid} \n")

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return MultiDiscrete([len(self.possible_agents)] + [self.grid_size] * 4 * len(self.U_t) + [self.grid_size]* 2 * len(self.g) + [2] * len(self.U_t))

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(5)
        #four directions and standing still



'''
    #potential render function

    def callRenderFunction(self):
        # Define some colors
        BLACK = (0, 0, 0)
        WHITE = (255, 255, 255)
        GREEN = (0, 255, 0)
        RED = (255, 0, 0)
        LightRED = (255, 187, 173)
        PURPLE = (159,43,104)
        YELLOW = (255,234,0)

        # This sets the WIDTH and HEIGHT of each grid location
        WIDTH = 20
        HEIGHT = 20

        # This sets the margin between each cell
        MARGIN = 5

        # Create a 2 dimensional array. A two dimensional
        # array is simply a list of lists.
        grid = []
        for row in range(10):
            # Add an empty array that will hold each cell
            # in this row
            grid.append([])
            for column in range(10):
                grid[row].append(0)  # Append a cell

        # Set row 1, cell 5 to one. (Remember rows and
        # column numbers start at zero.)

        for row in range(10):
            for column in range(10):
                if in_range_of_static(row, column, AllGBSs):
                    grid[row][column] = 3

        for i in range(0, len(AllGBSs)):
            grid[AllGBSs[i][0]][AllGBSs[i][1]] = 2

        grid[UAV1_start[0]][UAV1_start[1]] = 4
        grid[UAV1_end[0]][UAV1_end[1]] = 4
        grid[UAV2_start[0]][UAV2_start[1]] = 5
        grid[UAV2_end[0]][UAV2_end[1]] = 5

        grid[self.state[0]][self.state[1]] = 1
        grid[self.state[2]][self.state[3]] = 1



        # Initialize pygame
        pygame.init()

        # Set the HEIGHT and WIDTH of the screen
        WINDOW_SIZE = [255, 255]
        screen = pygame.display.set_mode(WINDOW_SIZE)

        # Set title of screen
        pygame.display.set_caption("Array Backed Grid")

        # Loop until the user clicks the close button.
        done = False

        # Used to manage how fast the screen updates
        clock = pygame.time.Clock()

        # -------- Main Program Loop -----------
        ttime = 0
        time = 200
        while ttime < time:
            ttime += 1
            # Set the screen background
            screen.fill(BLACK)

            # Draw the grid
            for row in range(10):
                for column in range(10):
                    color = WHITE
                    if grid[row][column] == 1:
                        color = GREEN
                    if grid[row][column] == 2:
                        color = RED
                    if grid[row][column] == 3:
                        color = LightRED
                    if grid[row][column] == 4:
                        color = PURPLE
                    if grid[row][column] == 5:
                        color = YELLOW
                    pygame.draw.rect(screen,
                                    color,
                                    [(MARGIN + WIDTH) * column + MARGIN,
                                    (MARGIN + HEIGHT) * row + MARGIN,
                                    WIDTH,
                                    HEIGHT])

            # Limit to 60 frames per second
            clock.tick(60)

            # Go ahead and update the screen with what we've drawn.
            pygame.display.flip()

        pygame.quit()

'''