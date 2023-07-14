import gymnasium as gym

from stable_baselines3 import DQN

from gym import wrappers

env_name = "CartPole-v1"

env = gym.make(env_name) # env_name = "Pendulum-v0"


model = DQN("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000, log_interval=4)
model.save("dqn_cartpole")

del model # remove to demonstrate saving and loading

model = DQN.load("dqn_cartpole")

obs, info = env.reset()
q=0
while q < 1000:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
    q +=1 


print('done')