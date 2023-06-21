from uav_environment import uav_collab

from pettingzoo.test import parallel_api_test

g = [(1,3), (4,5), (6,5)]
L_s = [(1,1), (4,5)]
L_f = [(2,4), (7,10)]
o_max = 20
V = 1 
R_G = 20
R_U = 5
grid_size = 20
trunc_step = 200
env = uav_collab(g, L_s, L_f, o_max, V, R_G, R_U, grid_size, trunc_step)
#parallel_api_test(env, num_cycles=10_000_000)

q =env.dist(env.U_t[0], env.L_f[0])
print(q)
print(type(q))
print(round(q))
print(int(q))
print(int(round(q)))