print("Hi - first Reinforcement code")

import math
import gym
import random
import os
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
import pygame




# GBSs
gbs1 = [0, 0]
gbs2 = [2, 2]
gbs3 = [5, 3]
gbs4 = [9, 3]
gbs5 = [0, 9]
gbs6=[2,7]
gbs7=[3,5]
gbs8 = [5, 5]
gbs9 = [8, 5]

UAV1_start = [0,0]
UAV1_end = [9,3]
UAV2_start = [0,9]
UAV2_end = [9,5]

range_gbs =1


AllGBSs = [gbs1, gbs2, gbs3, gbs4, gbs5, gbs8, gbs9]


def calDistanceState(state, param):
    a = state[0] - param[0]
    b = state[1] - param[1]
    a = a ** 2
    b = b ** 2
    distance = a + b
    return math.sqrt(distance)


def calDistance(x1, y1, x2, y2):
    a = x1 - x2
    b = y1 - y2
    a = a ** 2
    b = b ** 2
    distance = a + b
    return math.sqrt(distance)


def in_range_of(state, all_gbs, gbs_range):
    for i in range(0, len(all_gbs)):
        if calDistanceState(state, all_gbs[i]) <= gbs_range:
            return True
    return False


def in_range_of_static(row, column, AllGBSs):
    for i in range(0, len(AllGBSs)):
        if calDistance(row, column, AllGBSs[i][0], AllGBSs[i][1]) <= range_gbs:
            return True
    return False


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




class UAV(Env):
    def __init__(self):
        self.action_space = Discrete(16)

        self.observation_space = Box(0, 9, shape=(4,), dtype=int)
        self.state = [UAV1_start[0],UAV1_start[1],UAV2_start[0],UAV2_start[1]]
        self.outage1 = 0
        self.outage2 = 0
        self.max1 = 0
        self.max2 = 0


    # action probablity
    def step(self, action):
        # apply action

        lastLocation1 = [self.state[0],self.state[1]]
        lastLocation2 = [self.state[2],self.state[3]]

        if action == 0 and self.state[0]<9 and self.state[2]<9:
            self.state[0] += 1
            self.state[2] +=1
        if action == 1 and self.state[0]<9 and self.state[3]<9:
            self.state[0] += 1
            self.state[3]+=1
        if action == 2 and self.state[0]<9 and self.state[2]>0:
            self.state[0] +=1
            self.state[2]-=1
        if action == 3 and self.state[0]<9 and self.state[3]>0:
            self.state[0] +=1
            self.state[3]-=1

        if action == 4 and self.state[1]<9 and self.state[2]<9:
            self.state[1] += 1
            self.state[2] +=1
        if action == 5 and self.state[1]<9 and self.state[3]<9:
            self.state[1] += 1
            self.state[3]+=1
        if action == 6 and self.state[1]<9 and self.state[2]>0:
            self.state[1] +=1
            self.state[2]-=1
        if action == 7 and self.state[1]<9 and self.state[3]>0:
            self.state[1] +=1
            self.state[3]-=1

        if action == 8 and self.state[0] >0 and self.state[2] < 9:
            self.state[0] -= 1
            self.state[2] += 1
        if action == 9 and self.state[0] >0 and self.state[3] < 9:
            self.state[0] -= 1
            self.state[3] += 1
        if action == 10 and self.state[0] >0 and self.state[2] > 0:
            self.state[0] -= 1
            self.state[2] -= 1
        if action == 11 and self.state[0] >0 and self.state[3] > 0:
            self.state[0] -= 1
            self.state[3] -= 1

        if action == 12 and self.state[1] >0 and self.state[2] < 9:
            self.state[1] -= 1
            self.state[2] += 1
        if action == 13 and self.state[1] >0 and self.state[3] < 9:
            self.state[1] -= 1
            self.state[3] += 1
        if action == 14 and self.state[1] >0 and self.state[2] > 0:
            self.state[1] -= 1
            self.state[2] -= 1
        if action == 15 and self.state[1] >0 and self.state[3] > 0:
            self.state[1] -= 1
            self.state[3] -= 1




        state1 = [self.state[0],self.state[1]]
        state2= [self.state[2],self.state[3]]


        DifDistance1 =  calDistance(self.state[0],self.state[1],UAV1_end[0],UAV1_end[1])-calDistance(lastLocation1[0],lastLocation1[1],UAV1_end[0],UAV1_end[1])
        DifDistance2 =  calDistance(self.state[2],self.state[3],UAV2_end[0],UAV2_end[1])-calDistance(lastLocation2[0],lastLocation2[1],UAV2_end[0],UAV2_end[1])


        if(DifDistance1<0):
            distanceReward1 = -1
        else:
            distanceReward1 = -3

        if (DifDistance2 < 0):
            distanceReward2 = -1
        else:
            distanceReward2 = -3



        reward1= distanceReward1
        reward2 = distanceReward2

        if in_range_of(state1, AllGBSs, range_gbs):
            self.outage1 = 0
        else:
            if(self.outage1==1):
                reward1 = reward1 * (self.outage1+1)
                self.outage1+=1
            else:
                self.outage1+=1

        if in_range_of(state2, AllGBSs, range_gbs):
            self.outage2 = 0
        else:
            if (self.outage2 >= 1):
                reward2 = reward2 * (self.outage2+1)
                self.outage2+=1
            else:
                self.outage2 += 1



        reward = min(reward1,reward2)+reward1+reward2


        # if self.max1 >=30 or self.max2 >= 30:
        #     reward = -10000

        done1= False
        done2= False

        if  self.state[0] == UAV1_end[0] and self.state[1] == UAV1_end[1]:
            reward+=100
            done1 = True

        if  self.state[2]==UAV2_end[0] and self.state[3]==UAV2_end[1]:
            reward+=100
            done2 = True

        if done1 and done2:
            done = True
        else:
            done = False




        info = {}

        return self.state, reward, done, info

    def render(self):
        callRenderFunction(self)

    def reset(self):
        self.state = [UAV1_start[0],UAV1_start[1],UAV2_start[0],UAV2_start[1]]
        self.outage1=0
        self.outage2=0
        #self.max1=0
        #self.max2=0
        return self.state


env = UAV()


states = env.observation_space.shape[0]
actions = env.action_space.n

print(actions)


#
# episodes = 10
# for episode in range(1,episodes):
#     states = env.reset()
#     done = False
#     score = 0
#
#     while not done:
#         env.render()
#         action = random.choice([0,1,2,3])
#         n_state,reward,done,info = env.step(action)
#         score+=reward
#     print('Episode: {} Score: {}'.format(episode,score))



def build_model(states, actions):
    model = Sequential()
    model.add(Flatten(input_shape=(1, states)))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(actions, activation='linear'))
    return model


model = build_model(states, actions)
model.summary()


def build_agent(model, actions):
    policy = BoltzmannQPolicy()
    memory = SequentialMemory(limit=200000, window_length=1)
    dqn = DQNAgent(model=model, memory=memory, policy=policy, nb_actions=actions, nb_steps_warmup=50,
                   target_model_update=1e-3)
    print("gamma")
    print(dqn.gamma)
    print("batch size")
    print(dqn.batch_size)
    print("step")
    print(dqn.step)



    return dqn


dqn = build_agent(model, actions)

dqn.compile(Adam(lr=0.001), metrics=['mae'])
dqn.fit(env, nb_steps=50000, visualize=False, verbose=1)

scores = dqn.test(env, nb_episodes=10, visualize=False)




print(scores)
print(scores.history)


# dqn.save_weights('dqn1.h5f',overwrite=True)
#
# episodes = 1
# for episode in range(1, episodes + 1):
#     print('Episode:{} '.format(episode))
#     obs = env.reset()
#     done = False
#     score = 0
#     while not done:
#         #env.render()
#         action, _ = model.predict(obs)
#         obs, reward, done, info = env.step(action)
#         print('action selected: {} reward: {} new location'.format(action,reward))
#         print(obs)
#         score += reward
#     print('Episode:{} Score:{}'.format(episode, score))
#     print('-------------------------------------')
#
# env.close()
