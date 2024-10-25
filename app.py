import gymnasium as gym
import gym_cellular_automata as gymca
import matplotlib.pyplot as plt
import numpy as np
#from gym_cellular_automata.forest_fire.bulldozer import bulldozer

# benchmark mode
env_id = gymca.envs[0]
env = gym.make(env_id, render_mode="human")

# prototype mode
ProtoEnv = gymca.prototypes[0]
env = ProtoEnv(nrows=10, ncols=10,reward_per_empty=0)
print(env._reward_per_empty)
obs, info = env.reset()

total_reward = 0.0
done = False
step = 0
threshold = 25


# Random Policy for at most "threshold" steps
while not done and step < threshold:
    #agent_bull = bulldozer.ForestFireBulldozerEnv(env.nrows, env.ncols)
    action = env.action_space.sample()  # Your agent goes here!
    obs, reward, terminated, truncated, info = env.step(action)
    print(obs, reward, terminated, truncated, info)
    done = terminated or truncated
    total_reward += reward
    step += 1


print(f"{env_id}")
print(f"Total Steps: {step}")
print(f"Total Reward: {total_reward}")


final = np.array(obs[0])
plt.matshow(final)
plt.show()






