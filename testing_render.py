#testing render
import uav_environment_outage
import argparse

def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default="double_outage",
        help="the name of this experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="cleanRL",
        help="the wandb's project name")
    parser.add_argument("--save-model", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="whether to save model into the `runs/{run_name}` folder")
    
    

    
    


    #algorithm specific arguments
    parser.add_argument("--ent_coef", type=float, default=.01,
                    help="coefficient for entropy loss")
    parser.add_argument("--vf_coef", type=float, default=0.5,
                        help="coefficient for value function loss")
    parser.add_argument("--clip_coef", type=float, default=0.2,
                        help="coefficient for clip loss")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="discount factor (gamma)")
    parser.add_argument("--learning_rate", type=float, default=.0001,
                        help="learning rate")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="batch size for training")
    parser.add_argument("--max_cycles", type=int, default=400,
                        help="maximum number of cycles per episode")
    parser.add_argument("--total_episodes", type=int, default=10000,
                        help="total number of episodes to run")
    parser.add_argument("--decreasing_ent", default=True,
        help="whether or not to decrease the entropy over time")
    parser.add_argument("--cut_ent", default=False,
        help="whether or not to decrease the entropy over time")
    
    
    


    


    #Environment specific arguments
    parser.add_argument("--g", type=list, default=[ (1,1), (5,5), (10,10), (15,5), (5,15), (19,1), (1,19)],
                    help="list of gbs locations")
    parser.add_argument("--L_s", type=list, default=[(1, 1), (2,2)],
                        help="list of drone starting locations")
    parser.add_argument("--L_f", type=list, default=[(19,1), (1,19)],
                        help="list of drone end locations")
    parser.add_argument("--o_max", type=int, default=2,
                        help="outage constraint")
    parser.add_argument("--V", type=int, default=1,
                        help="max veloctiy")
    parser.add_argument("--R_G", type=int, default=4,
                        help="Radius of connection from GBS-> UAV")
    parser.add_argument("--R_U", type=int, default=3,
                        help="Radius of connection from UAV-> UAV")
    parser.add_argument("--grid_size", type=int, default=20,
                        help="length of each grid axis (gris is a square)")
    parser.add_argument("--total_timesteps", type=int, default=50,
        help="total timesteps of the experiments")
    parser.add_argument("--act_len", default=5,
        help="length of actions")
    
    args = parser.parse_args()
    return args


args = parse_args()



env = uav_environment_outage.uav_collab( args.g, args.L_s, args.L_f, args.o_max, args.V, args.R_G, args.R_U, args.grid_size, args.total_timesteps)

env.render()








