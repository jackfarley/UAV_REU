from stable_baselines3.common.env_checker import check_env
import argparse
import os
import random
import time
from distutils.util import strtobool

import numpy as np

import uav_maze


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

args = parse_args()

env = uav_maze.uav_collab( args.g, args.L_s, args.L_f, args.o_max, args.V, args.R_G, args.R_U, args.grid_size, args.total_timesteps, args.total_episodes)


# It will check your custom environment and output additional warnings if needed
check_env(env, warn=True)