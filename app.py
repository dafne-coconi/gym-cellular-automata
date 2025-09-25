import gymnasium as gym
import gym_cellular_automata as gymca
import matplotlib.pyplot as plt
import numpy as np
from  matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import ListedColormap
#from gym_cellular_automata.forest_fire.bulldozer import bulldozer

# benchmark mode
env_id = gymca.envs[2] # Sarsa env
env = gym.make(env_id, render_mode="human")

# prototype mode
ProtoEnv = gymca.prototypes[2]
env = ProtoEnv(nrows=12, ncols=12,reward_per_empty=0)
print(env._reward_per_move)
#obs, info = env.reset()



episodes=400
ep_steps = []

alpha=0.5
gamma=0.8
epsilon=0.1

n_states = env.nrows * env.ncols
print(env.action_space.n)
n_actions = env.action_space.n

Q = np.zeros((env.nrows, env.ncols, n_actions))

num_done = 0
num_burned = 0
num_out_trh = 0
def choose_action_greedy(state):
    if np.random.rand() < epsilon:
        return np.random.randint(n_actions)
    return np.argmax(Q[state[0]][state[1]])

obs, info = env.reset()
# Random Policy for at most "threshold" steps
for episode in range(episodes):
    print(episode)
    obs, info = env.reset_same_env()
    state_Q = obs[1][1]

    action_num = choose_action_greedy(state_Q)
    total_reward = 0.0
    done = False
    step = 0
    threshold = 300
    
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
        ep_steps.append(episode)
        
        if done or step >= threshold:
            print(f"truncated {truncated} or terminated {terminated}")
            print(f'total reward {total_reward}')
            print(f'step {step}')
            if terminated:
                num_done += 1
            elif truncated:
                num_burned += 1
            else: 
                num_out_trh += 1
                

num_total = num_out_trh + num_burned + num_done

sizes = [num_burned/num_total, num_out_trh/num_total, num_done/num_total]
labels = ['Burned', 'Out of threshold', 'Safe']

# Create the pie chart
plt.pie(sizes, labels=labels, autopct='%1.1f%%')

# Add a title
plt.title('Success rate SARSA')

# Ensure the circle is drawn as a circle
plt.axis('equal') 

# Display the chart
plt.savefig('Piechart30.png')
plt.show()

#print(f"{env_id}")
print(f"Total Steps: {step}")
print(f"Total Reward: {total_reward}")


#fig = plt.figure()
#plot = plt.matshow(data[0], fignum=0)
plt.plot(ep_steps)
plt.title('AC Forest Fire Sarsa (∈=0.1,α=0.5)')
plt.xlabel("Number of steps")
plt.ylabel("Number of episodes")
plt.savefig('Fig_30.png')
plt.show()


cmap = ListedColormap(['black', 'green', 'red', 'yellow','blue'])
bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5] # Define the boundaries for each color
norm = plt.matplotlib.colors.BoundaryNorm(bounds, cmap.N)
   
fig, ax = plt.subplots()
im = ax.imshow(data[0], cmap = cmap, norm = norm, animated = True)
plt.title('AC (∈=0.1,α=0.5)')

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
anim.save("Test30.gif", writer = PillowWriter(fps = 5))

plt.show()





