import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from supersuit import color_reduction_v0, frame_stack_v1, resize_v1
from torch.distributions.categorical import Categorical

from pettingzoo.butterfly import pistonball_v6



from pettingzoo.butterfly import pistonball_v6

env = pistonball_v6.parallel_env(render_mode="human")
observations = env.reset()

for i in range(3):
    # this is where you would insert your policy
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}  

    observations, rewards, terminations, truncations, infos = env.step(actions)
    print(observations)
env.close()
