# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/dqn/#dqnpy
import argparse
import os
import random
import time
from distutils.util import strtobool
import obs_random_spawn_no_outage


from copy import copy, deepcopy


import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter

'''
1. Unique observations are essential
2. Change to not be in batches, and have much less frequent training updates
3. Look at ways delta can change GBS exploration pog'''


def parse_args():
    # system arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default="Outage Lower Obs attempt",
        help="the name of this experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="DQN",
        help="the wandb's project name")
    parser.add_argument("--save_model", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="whether to save model into the `runs/{run_name}` folder")
    
    


    #algorithm specific arguments
    parser.add_argument("--total_episodes", type=int, default=25000,
        help="total episodes of the experiments")
    parser.add_argument("--max_cycles", type=int, default=400,
                        help="maximum number of cycles per episode")
    
    parser.add_argument("--total_time", type=int, default=50,
                        help= "timesteps per episode" )
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=1,
        help="the number of parallel game environments")
    parser.add_argument("--buffer-size", type=int, default=100000,
        help="the replay memory buffer size")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--tau", type=float, default=1.,
        help="the target network update rate")
    parser.add_argument("--target-network-frequency", type=int, default=50,
        help="the episodes it takes to update the target network")
    parser.add_argument("--batch-size", type=int, default=2048,
        help="the batch size of sample from the reply memory")
    parser.add_argument("--start_e", type=float, default=1,
        help="the starting epsilon for exploration")
    parser.add_argument("--end_e", type=float, default=0.02,
        help="the ending epsilon for exploration")
    parser.add_argument("--exploration-fraction", type=float, default=0.4,
        help="the fraction of `total-episodes` it takes from start-e to go end-e")
    parser.add_argument("--learning_starts", type=int, default=1000,
        help="episode to start learning")
    parser.add_argument("--train_frequency", type=int, default=5,
        help="the frequency of training")
    parser.add_argument("--num_envs", type=int, default=1,
        help="in case we ever do parrallelization")
    parser.add_argument("--linear_ent", default=True,
        help="sloping down the entropy linearly")
    parser.add_argument("--cut_ent", default=True,
        help="cutting the entropy at learning")
    
    



    #Environment arguments
    parser.add_argument("--g", type=list, default=[ (0,0), (5,5), (10,10), (15,5), (20,1)],
                    help="list of gbs locations")
    parser.add_argument("--L_s", type=list, default=[(1,1)],
                        help="list of drone starting locations")
    parser.add_argument("--L_f", type=list, default=[(20,1)],
                        help="list of drone end locations")
    parser.add_argument("--o_max", type=int, default=0,
                        help="outage constraint")
    parser.add_argument("--V", type=int, default=1,
                        help="max veloctiy")
    parser.add_argument("--R_G", type=int, default=3.8,
                        help="Radius of connection from GBS-> UAV")
    parser.add_argument("--R_U", type=int, default=3,
                        help="Radius of connection from UAV-> UAV")
    parser.add_argument("--grid_size", type=int, default=20,
                        help="length of each grid axis (gris is a square)")
    parser.add_argument("--total_timesteps", type=int, default=256,
        help="total timesteps of the experiments")
    parser.add_argument("--act_len", type=int, default=5,
        help="length of actions")
    parser.add_argument("--obser", type=int, default=2,
                        help='kj;lasdkjf')
    
    
    args = parser.parse_args()
    return args

    

#helper functions
def batchify(x, device):
    """Converts petting zoo style returns to batch of torch arrays."""
    # convert to list of np arrays
    x = np.stack([x[a] for a in x], axis=0)
    # convert to torch
    x = torch.FloatTensor(x).to(device)

    return x


def unbatchify(x, env):
    """Converts torch array to petting zoo style arguments."""
    x = x.cpu().numpy()
    x = {a: x[i] for i, a in enumerate(env.possible_agents)}

    return x






class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.obs_len = 4
        network_size_1 = 120
        network_size_2 = 84
        self.network = nn.Sequential(
            nn.Linear(self.obs_len, network_size_1),
            nn.ReLU(),
            nn.Linear(network_size_1, network_size_2),
            nn.ReLU(),
            nn.Linear(network_size_2, args.act_len),
        )

    def forward(self, x):
            return self.network(x)
    
def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)



if __name__ == "__main__":
    import stable_baselines3 as sb3


    args = parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    #setting up tracking progress
    run_name = f"{args.exp_name}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )


    env = obs_random_spawn_no_outage.uav_collab( args.g, args.L_s, args.L_f, args.o_max, args.V, args.R_G, args.R_U, args.grid_size, args.total_timesteps)
    num_agents = 1
    num_actions = args.act_len



    q_network = QNetwork(env).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    target_network = QNetwork(env).to(device)
    target_network.load_state_dict(q_network.state_dict())



    '''
    this will need to be debugged'''
    rb = ReplayBuffer(
        args.buffer_size,
        gym.spaces.MultiDiscrete( [env.grid_size] * 4),
        gym.spaces.Discrete(5),
        device,
        handle_timeout_termination=False,
    )





    '''episode storage'''

    start_time = time.time()
    global_step = 0

    
    for episode in range(args.total_episodes):
        with torch.no_grad():
            next_obs, info = env.reset(seed=0)
            if args.linear_ent:

                epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_episodes, episode)
            elif args.cut_ent:
                if episode < args.learning_starts:
                    epsilon = args.start_e
                else:
                    epsilon = args.end_e
            total_episodic_return = 0
            for step in range(0, args.max_cycles):
                global_step += 1

                obs = batchify(next_obs, device)


 
                if random.random() < epsilon:
                    action = torch.from_numpy(np.array([random.randint(0,4) for _ in range(args.num_envs)]))
                else:
                    q_value = q_network(torch.Tensor(obs).to(device))
                    action = torch.argmax(q_value, dim=1)

                #may need something for truncation?    


                next_obs, rewards, terminated, truncated, infos = env.step(unbatchify(action, env))
                tn = copy(next_obs)
                true_next = batchify(tn,device)
                rb.add(obs, true_next, action, batchify(rewards, device), batchify(terminated, device), infos)
                total_episodic_return += batchify(rewards, device).cpu().numpy()


                if any([terminated[a] for a in terminated]) or any([truncated[a] for a in truncated]):
                        end_step = step
                        for _ in range(1000):
                            rb.add(obs, true_next, action, batchify(rewards, device), batchify(terminated, device), infos)

                        break
                

            print("Episode: ", episode)
            print("Return: ", total_episodic_return)
            print("Length: ", end_step)
            writer.add_scalar("charts/length", end_step, global_step)
            writer.add_scalar("charts/return", total_episodic_return, global_step)





        # ALGO LOGIC: training.
        if episode > args.learning_starts:
            if episode % args.train_frequency == 0:
                data = rb.sample(args.batch_size)

                nobs = data.next_observations.cpu().numpy()
                nobs = torch.FloatTensor(nobs)

                aged = data.observations.cpu().numpy()
                aged = torch.FloatTensor(aged)

                with torch.no_grad():
                    target_max, _ = target_network(nobs).max(dim=1)
                    td_target = data.rewards.flatten() + args.gamma * target_max * (1 - data.dones.flatten())
                old_val = q_network(aged).gather(1, data.actions).squeeze()
                loss = F.mse_loss(td_target, old_val)

                if episode % 100 == 0:
                    writer.add_scalar("losses/td_loss", loss, global_step)
                    writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)
                    #print("SPS:", int(global_step / (time.time() - start_time)))
                    writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

                # optimize the model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # update target network
            if global_step % args.target_network_frequency == 0:
                for target_network_param, q_network_param in zip(target_network.parameters(), q_network.parameters()):
                    target_network_param.data.copy_(
                        args.tau * q_network_param.data + (1.0 - args.tau) * target_network_param.data
                    )

    if args.save_model:
        model_path = f"saving/{args.exp_name}"
        torch.save(q_network.state_dict(), model_path)
        print(f"model saved to {model_path}")



    
    env.close()
    writer.close()
    print(time.time()- start_time)


