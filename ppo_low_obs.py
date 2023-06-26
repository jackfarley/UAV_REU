# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/dqn/#dqnpy
import argparse
import os
import random
import time
from distutils.util import strtobool
import trying_fewer_obs


import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
from torch.distributions.categorical import Categorical


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default="1. Just learning distance, joint",
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
    parser.add_argument("--ent_coef", type=float, default=.01
                        ,
                    help="coefficient for entropy loss")
    parser.add_argument("--vf_coef", type=float, default=0.5,
                        help="coefficient for value function loss")
    parser.add_argument("--clip_coef", type=float, default=0.6,
                        help="coefficient for clip loss")
    parser.add_argument("--gamma", type=float, default=0.995,
                        help="discount factor (gamma)")
    parser.add_argument("--learning_rate", type=float, default=.00005,
                        help="learning rate")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="batch size for training")
    parser.add_argument("--max_cycles", type=int, default=400,
                        help="maximum number of cycles per episode")
    parser.add_argument("--total_episodes", type=int, default=3000,
                        help="total number of episodes to run")
    parser.add_argument("--decreasing_ent", default=False,
        help="whether or not to decrease the entropy over time")
    parser.add_argument("--cut_ent", default=False,
        help="whether or not to decrease the entropy over time")
    
    
    


    


    #Environment specific arguments
    parser.add_argument("--g", type=list, default=[ (1,1), (5,5), (10,10), (15,5),  (5,15), (20,1), (1,20), (20,20)],
                    help="list of gbs locations")
    parser.add_argument("--L_s", type=list, default=[(1, 1), (20,20)],
                        help="list of drone starting locations")
    parser.add_argument("--L_f", type=list, default=[(20,1), (1,20)],
                        help="list of drone end locations")
    parser.add_argument("--o_max", type=int, default=2,
                        help="outage constraint")
    parser.add_argument("--V", type=int, default=1,
                        help="max veloctiy")
    parser.add_argument("--R_G", type=int, default=3.3,
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

    



#helper functions
def batchify(x, device):
    """Converts petting zoo style returns to batch of torch arrays."""
    # convert to list of np arrays
    x = np.stack([x[a] for a in x], axis=0)
    # convert to torch
    x = torch.FloatTensor(x).to(device)

    return x


def unbatchify(x, env):
    """Converts np array to petting zoo style arguments."""
    x = x.cpu().numpy()
    x = {a: x[i] for i, a in enumerate(env.possible_agents)}

    return x


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.obs_len = 4  
        network_size = 256
        self.critic = nn.Sequential(
            layer_init(nn.Linear(self.obs_len, network_size)),
            nn.ReLU(),
            layer_init(nn.Linear(network_size, network_size)),
            nn.ReLU(),
            layer_init(nn.Linear(network_size, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(self.obs_len, network_size)),
            nn.ReLU(),
            layer_init(nn.Linear(network_size, network_size)),
            nn.ReLU(),
            layer_init(nn.Linear(network_size, args.act_len), std=0.01),
        )


    def get_value(self, x):
            return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)



if __name__ == "__main__":
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


    env = trying_fewer_obs.uav_collab( args.g, args.L_s, args.L_f, args.o_max, args.V, args.R_G, args.R_U, args.grid_size, args.total_timesteps)
    num_agents = len(env.possible_agents)
    num_actions = 5
    observation_size = env.observation_space(env.possible_agents[0]).shape


    '''settting up learning'''
    agent = Agent(env = env).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)


    '''episode storage'''
    global_step = 0
    start_time = time.time()

    end_step = 0
    total_episodic_return = 0
    rb_obs = torch.zeros((args.max_cycles, num_agents, agent.obs_len)).to(device)
    rb_actions = torch.zeros((args.max_cycles, num_agents)).to(device)
    rb_logprobs = torch.zeros((args.max_cycles, num_agents)).to(device)
    rb_rewards = torch.zeros((args.max_cycles, num_agents)).to(device)
    rb_terms = torch.zeros((args.max_cycles, num_agents)).to(device)
    rb_values = torch.zeros((args.max_cycles, num_agents)).to(device)



    """ TRAINING LOGIC """
    # train for n number of episodes
    ent_mat = np.linspace(args.ent_coef, 0, args.total_episodes)
    cut = np.zeros(args.total_episodes)
    cut[:round(args.total_episodes*.8)] = args.ent_coef
    for episode in range(args.total_episodes):

        '''
        add function to have entropy bonus decrease over time
        '''
        # collect an episode
        with torch.no_grad():
            # collect observations and convert to batch of torch tensors
            next_obs, info = env.reset(seed=0)
            # reset the episodic return
            total_episodic_return = 0

            # each episode has num_steps
            for step in range(0, args.max_cycles):
                # rollover the observation
                obs = batchify(next_obs, device)
                global_step += 1


                # get action from the agent
                actions, logprobs, _, values = agent.get_action_and_value(obs)

                # execute the environment and log data
                next_obs, rewards, terms, truncs, infos = env.step(
                    unbatchify(actions, env)
                )

                # add to episode storage
                rb_obs[step] = obs
                rb_rewards[step] = batchify(rewards, device)
                rb_terms[step] = batchify(terms, device)
                rb_actions[step] = actions
                rb_logprobs[step] = logprobs
                rb_values[step] = values.flatten()

                # compute episodic return
                total_episodic_return += rb_rewards[step].cpu().numpy()


                # if we reach termination or truncation, end
                if any([terms[a] for a in terms]) or any([truncs[a] for a in truncs]):
                    end_step = step
                    break

        # bootstrap value if not done
        with torch.no_grad():
            rb_advantages = torch.zeros_like(rb_rewards).to(device)
            for t in reversed(range(end_step)):
                delta = (
                    rb_rewards[t]
                    + args.gamma * rb_values[t + 1] * rb_terms[t + 1]
                    - rb_values[t]
                )
                rb_advantages[t] = delta + args.gamma * args.gamma * rb_advantages[t + 1]
            rb_returns = rb_advantages + rb_values

        # convert our episodes to batch of individual transitions
        b_obs = torch.flatten(rb_obs[:end_step], start_dim=0, end_dim=1)
        b_logprobs = torch.flatten(rb_logprobs[:end_step], start_dim=0, end_dim=1)
        b_actions = torch.flatten(rb_actions[:end_step], start_dim=0, end_dim=1)
        b_returns = torch.flatten(rb_returns[:end_step], start_dim=0, end_dim=1)
        b_values = torch.flatten(rb_values[:end_step], start_dim=0, end_dim=1)
        b_advantages = torch.flatten(rb_advantages[:end_step], start_dim=0, end_dim=1)

        # Optimizing the policy and value network
        b_index = np.arange(len(b_obs))
        clip_fracs = []
        for repeat in range(3):
            # shuffle the indices we use to access the data
            np.random.shuffle(b_index)
            for start in range(0, len(b_obs), args.batch_size):
                # select the indices we want to train on
                end = start + args.batch_size
                batch_index = b_index[start:end]

                _, newlogprob, entropy, value = agent.get_action_and_value(
                    b_obs[batch_index], b_actions[batch_index]
                )
                logratio = newlogprob - b_logprobs[batch_index]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clip_fracs += [
                        ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()
                    ]

                # normalize advantaegs
                advantages = b_advantages[batch_index]
                advantages = (advantages - advantages.mean()) / (
                    advantages.std() + 1e-8
                )

                # Policy loss
                pg_loss1 = -b_advantages[batch_index] * ratio
                pg_loss2 = -b_advantages[batch_index] * torch.clamp(
                    ratio, 1 - args.clip_coef, 1 + args.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                value = value.flatten()
                v_loss_unclipped = (value - b_returns[batch_index]) ** 2
                v_clipped = b_values[batch_index] + torch.clamp(
                    value - b_values[batch_index],
                    -args.clip_coef,
                    args.clip_coef,
                )
                v_loss_clipped = (v_clipped - b_returns[batch_index]) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()

                entropy_loss = entropy.mean()
                if args.cut_ent:
                    entc = cut[episode]

                elif args.decreasing_ent:
                    entc = ent_mat[episode]
                else:
                    entc = args.ent_coef
                loss = pg_loss - entc * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        print(f"Training episode {episode}")
        print(f"Episodic Return: {np.mean(total_episodic_return)}")
        print(f"Episode Length: {end_step}")
        mter = np.mean(total_episodic_return)
        # print("")
        # print(f"Value Loss: {v_loss}")
        # print(f"Policy Loss: {pg_loss.item()}")
        # print(f"Old Approx KL: {old_approx_kl.item()}")
        # print(f"Approx KL: {approx_kl.item()}")
        # print(f"Clip Fraction: {np.mean(clip_fracs)}")
        # print(f"Explained Variance: {explained_var.item()}")
        # print("\n-------------------------------------------\n")

        writer.add_scalar("charts/return", mter.item(), global_step)
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)


    

    env.close()
    writer.close()
    print(time.time()- start_time)


