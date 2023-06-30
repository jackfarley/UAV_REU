# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/dqn/#dqnpy
import argparse
import os
import random
import time
from distutils.util import strtobool
import obs_random_spawn


from copy import copy, deepcopy


import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
from torch.distributions.categorical import Categorical

'''
1. Unique observations are essential
2. Change to not be in batches, and have much less frequent training updates'''


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default="Outage Lower Obs attempt",
        help="the name of this experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="cleanRL",
        help="the wandb's project name")
    parser.add_argument("--save_model", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="whether to save model into the `runs/{run_name}` folder")
    
    

    
    


    #algorithm specific arguments
    parser.add_argument("--ent_coef", type=float, default=.00
                        ,
                    help="coefficient for entropy loss")
    parser.add_argument("--vf_coef", type=float, default=0.5,
                        help="coefficient for value function loss")
    parser.add_argument("--clip_coef", type=float, default=0.2,
                        help="coefficient for clip loss")
    parser.add_argument("--gamma", type=float, default=0.999,
                        help="discount factor (gamma)")
    parser.add_argument("--gae_lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--learning_rate", type=float, default=.00025,
                        help="learning rate")
    #parser.add_argument("--batch_size", type=int, default=32,
    #                    help="batch size for training")
    parser.add_argument("--max_cycles", type=int, default=400,
                        help="maximum number of cycles per episode")
    parser.add_argument("--total_episodes", type=int, default=1000,
                        help="total number of episodes to run")
    parser.add_argument("--decreasing_ent", default=False,
        help="whether or not to decrease the entropy over time")
    parser.add_argument("--cut_ent", default=False,
        help="whether or not to cut the entropy")
    parser.add_argument("--runs_per_batch", default=10,
        help="whether or not to cut the entropy")
    
    
    
    
    


    


    #Environment specific arguments
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
        self.obs_len = 2 
        network_size = 64
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(self.obs_len,), network_size)),
            nn.Tanh(),
            layer_init(nn.Linear(network_size, network_size)),
            nn.Tanh(),
            layer_init(nn.Linear(network_size, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(self.obs_len,), network_size)),
            nn.Tanh(),
            layer_init(nn.Linear(network_size, network_size)),
            nn.Tanh(),
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

    tes = np.array([1,2])
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


    env = obs_random_spawn.uav_collab( args.g, args.L_s, args.L_f, args.o_max, args.V, args.R_G, args.R_U, args.grid_size, args.total_timesteps)
    num_agents = len(env.possible_agents)
    args.batch_size = int(num_agents * args.total_timesteps*args.runs_per_batch)
    num_actions = 5
    observation_size = env.observation_space(env.possible_agents[0]).shape


    '''settting up learning'''
    agent = Agent(env = env).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)


    '''episode storage'''
    global_step = 0
    start_time = time.time()

    end_step = 0
    
    rb_obs = torch.zeros((args.total_timesteps + 2, args.runs_per_batch ) + tes.shape).to(device)
    rb_actions = torch.zeros((args.total_timesteps+ 2, args.runs_per_batch)).to(device)
    rb_logprobs = torch.zeros((args.total_timesteps+ 2, args.runs_per_batch)).to(device)
    rb_rewards = torch.zeros((args.total_timesteps+ 2, args.runs_per_batch)).to(device)
    rb_terms = torch.zeros((args.total_timesteps+ 2,  args.runs_per_batch)).to(device)
    rb_values = torch.zeros((args.total_timesteps + 2, args.runs_per_batch)).to(device)

    
    



    """ TRAINING LOGIC """
    # train for n number of episodes
    ent_mat = np.linspace(args.ent_coef, 0, args.total_episodes)
    cut = np.zeros(args.total_episodes)
    cut[:round(args.total_episodes*.8)] = args.ent_coef
 
    for episode in range(args.total_episodes):
        end_list = np.zeros(args.runs_per_batch)
        ret_list = np.zeros(args.runs_per_batch)

        for dum in range(args.runs_per_batch):


        


            # collect an episode
            with torch.no_grad():
                # collect observations and convert to batch of torch tensors
                next_obs, info = env.reset(seed=0)
                # reset the episodic return
                total_episodic_return = 0
                next_done = torch.zeros(num_agents).to(device)

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




                    fut_obs = copy(batchify(next_obs, device))

                    next_done = batchify(terms, device)



                    # add to episode storage
                    rb_obs[step,dum] = obs
                    rb_rewards[step,dum] = batchify(rewards, device)
                    rb_terms[step,dum] = batchify(terms, device)
                    rb_actions[step,dum] = actions
                    rb_logprobs[step,dum] = logprobs
                    rb_values[step,dum] = values.flatten()

                    # compute episodic return
                    total_episodic_return += rb_rewards[step,dum].cpu().numpy()


                    # if we reach termination or truncation, end
                    if any([terms[a] for a in terms]) or any([truncs[a] for a in truncs]):
                        end_step = step
                        break

            end_list[dum] = end_step
            ret_list[dum] = total_episodic_return

        # bootstrap value if not done
            with torch.no_grad():
                next_value = agent.get_value(fut_obs)
                rb_advantages = torch.zeros_like(rb_rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(end_step)):
                    if t == end_step - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value

                    else:
                        nextnonterminal = 1.0 - rb_terms[t+1, dum]
                        nextvalues = rb_values[t+1,dum]


                    delta = rb_rewards[t,dum] + args.gamma * nextvalues * nextnonterminal- rb_values[t,dum]
                    #print(delta)
                    rb_advantages[t,dum] = lastgaelam =  delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                    #print(rb_advantages[t,dum])
                rb_returns = rb_advantages + rb_values

            b_obs_curr = rb_obs[:end_step,dum].reshape((-1,)+ tes.shape)
            b_actions_curr = rb_actions[:end_step,dum].reshape((-1,))
            b_logprobs_curr = rb_logprobs[:end_step,dum].reshape((-1,))
            b_returns_curr = rb_returns[:end_step,dum].reshape((-1,))
            b_values_curr = rb_values[:end_step,dum].reshape((-1,))
            b_advantages_curr = rb_advantages[:end_step,dum].reshape((-1,))
            # convert our episodes to batch of individual transitions


            if dum == 0:

                b_obs = b_obs_curr
                b_logprobs= b_logprobs_curr
                b_actions= b_actions_curr 
                b_returns= b_returns_curr
                b_values= b_values_curr 
                b_advantages= b_advantages_curr 
            else:
                b_obs = torch.cat((b_obs, b_obs_curr),0)
                b_logprobs = torch.cat((b_logprobs, b_logprobs_curr))
                b_actions = torch.cat((b_actions, b_actions_curr),0)
                b_returns = torch.cat((b_returns, b_returns_curr))
                b_values = torch.cat((b_values, b_values_curr))
                b_advantages = torch.cat((b_advantages, b_advantages_curr))




  

    # Optimizing the policy and value network
        b_index = np.arange(len(b_obs))
        clip_fracs = []
        for repeat in range(3):
            # shuffle the indices we use to access the data
            np.random.shuffle(b_index)

            # select the indices we want to train on
            batch_index = b_index





            _, newlogprob, entropy, value = agent.get_action_and_value(
                b_obs[batch_index], b_actions[batch_index]#b_obs[batch_index], b_actions.long()[batch_index]
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
        reti = np.mean(ret_list)
        leni = np.mean(end_list)
        print(f"Training episode {episode}")
        print(f"Episodic Return: {np.mean(ret_list)}")
        print(f"Episode Length: {np.mean(end_list)}")
        # print("")
        # print(f"Value Loss: {v_loss}")
        # print(f"Policy Loss: {pg_loss.item()}")
        # print(f"Old Approx KL: {old_approx_kl.item()}")
        # print(f"Approx KL: {approx_kl.item()}")
        # print(f"Clip Fraction: {np.mean(clip_fracs)}")
        # print(f"Explained Variance: {explained_var.item()}")
        # print("\n-------------------------------------------\n")

        writer.add_scalar("charts/return", reti.item(), global_step)
        writer.add_scalar("charts/average_length", leni.item(), global_step )
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        

        if episode%100 == 0:

            if args.save_model :

    
                torch.save(agent.state_dict(), 'test_save' + str(episode) + '.pth')

    if args.save_model :

    
        torch.save(agent.state_dict(), 'test_save')
    env.close()
    writer.close()
    print(time.time()- start_time)


