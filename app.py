import gymnasium as gym
import gym_cellular_automata as gymca
import matplotlib.pyplot as plt
import numpy as np
from  matplotlib.animation import FuncAnimation, PillowWriter
#from gym_cellular_automata.forest_fire.bulldozer import bulldozer

# benchmark mode
env_id = gymca.envs[1]
env = gym.make(env_id, render_mode="human")

# prototype mode
ProtoEnv = gymca.prototypes[0]
env = ProtoEnv(nrows=20, ncols=20,reward_per_empty=0)
print(env._reward_per_empty)
obs, info = env.reset()

total_reward = 0.0
done = False
step = 0
threshold = 65
#data = np.zeros(shape=(25, 1))
data = []
# Random Policy for at most "threshold" steps
while not done and step < threshold:
    
    #agent_bull = bulldozer.ForestFireBulldozerEnv(env.nrows, env.ncols)
    action = env.action_space.sample()  # Your agent goes here!
    obs, reward, terminated, truncated, info = env.step(action)
    print(obs, reward, terminated, truncated, info)
    done = terminated or truncated

    print("step", step)
    data.append(obs[0])
    total_reward += reward
    step += 1


print(f"{env_id}")
print(f"Total Steps: {step}")
print(f"Total Reward: {total_reward}")

print(data[1])
print(type(data))
print(len(data))

#fig = plt.figure()
#plot = plt.matshow(data[0], fignum=0)
fig, ax = plt.subplots()
im = ax.imshow(data[0], cmap = 'viridis', animated = True)

def init():
    plot.set_data(data[0])
    return plot

def update(j):
    im.set_array(data[j])
    return [im]
    #plot.set_data(data[j])
    #return [plot]

#anim = FuncAnimation(fig, update, init_func = init, frames=n_frames, interval = 500)
anim = FuncAnimation(fig, update, frames=len(data), interval = 200, blit = True)
anim.save("Test3.gif", writer = PillowWriter(fps = 5))

plt.show()





