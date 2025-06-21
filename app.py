import gymnasium as gym
import gym_cellular_automata as gymca
import matplotlib.pyplot as plt
import numpy as np
from  matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import ListedColormap
#from gym_cellular_automata.forest_fire.bulldozer import bulldozer

# benchmark mode
env_id = gymca.envs[2]
env = gym.make(env_id, render_mode="human")

# prototype mode
ProtoEnv = gymca.prototypes[2]
env = ProtoEnv(nrows=12, ncols=12,reward_per_empty=0)
print(env._reward_per_move)
#obs, info = env.reset()



episodes=300
alpha=0.5
gamma=0.8
epsilon=0.1

n_states = env.nrows * env.ncols
print(env.action_space.n)
n_actions = env.action_space.n

Q = np.zeros((env.nrows, env.ncols, n_actions))

def choose_action_greedy(state):
    if np.random.rand() < epsilon:
        return np.random.randint(n_actions)
    return np.argmax(Q[state[0]][state[1]])

# Random Policy for at most "threshold" steps
for episode in range(episodes):
    if episode//100 == int:
        print(episode)
    obs, info = env.reset()
    #print(obs)
    state_Q = obs[1][1]

    action_num = choose_action_greedy(state_Q)
    total_reward = 0.0
    done = False
    step = 0
    threshold = 100
    
    data = []
    while not done and step < threshold:
        
        #action = env.action_space[action_num]  # Your agent goes here!
        action = action_num
        obs, reward, terminated, truncated, info = env.step(action)
        #print(obs[0])
        done = terminated or truncated
################
        next_state = obs[1][1]
        next_action = choose_action_greedy(next_state)

        Q[tuple(state_Q)][action] += alpha * (
            reward + gamma * Q[tuple(next_state)][next_action] - Q[tuple(state_Q)][action]
            )

        state_Q = next_state ####### select state from OBS
        action = next_action

        #print("step", step)
        data.append(obs[0].copy())
        #print(f'state_Q {state_Q}')
        data[step][tuple(state_Q)] = 3
        #print(data[step])
        total_reward += reward
        step += 1


#print(f"{env_id}")
print(f"Total Steps: {step}")
print(f"Total Reward: {total_reward}")


#fig = plt.figure()
#plot = plt.matshow(data[0], fignum=0)

cmap = ListedColormap(['blue', 'green', 'red', 'yellow'])
bounds = [-0.5, 0.5, 1.5, 2.5, 3.5] # Define the boundaries for each color
norm = plt.matplotlib.colors.BoundaryNorm(bounds, cmap.N)
   
fig, ax = plt.subplots()
im = ax.imshow(data[0], cmap = cmap, norm = norm, animated = True)

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
anim.save("Test7.gif", writer = PillowWriter(fps = 5))

plt.show()





