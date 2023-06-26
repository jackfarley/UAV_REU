import pygame
from copy import copy, deepcopy






class render():
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
