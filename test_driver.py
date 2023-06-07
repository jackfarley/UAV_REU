from uav_environment import uav_collab

from pettingzoo.test import parallel_api_test


env = uav_collab()
parallel_api_test(env, num_cycles=1_000_000)