import numpy as np
from Environment import Cartpole3D
from ActorAndCritic import Actor, Critic

# Load model
def load_model(actor, critic, filename):
    data = np.load(filename)
    actor.W1, actor.b1 = data['W1'], data['b1']
    actor.W2, actor.b2 = data['W2'], data['b2']
    actor.W3 = data['W3_mean']  # FIXED: assign to self.W3
    critic.W1, critic.b1 = data['cW1'], data['cb1']
    critic.W2, critic.b2 = data['cW2'], data['cb2']
    critic.W3, critic.b3 = data['cW3'], data['cb3']

std_fixed = 1.0

# Test a model
def test_model(filename, num_episodes=100, render=False):
    env = Cartpole3D()
    actor = Actor()
    critic = Critic()
    load_model(actor, critic, filename)
    
    total_reward = 0
    survived = 0
    
    for _ in range(num_episodes):
        state = env.reset()
        ep_reward = 0
        done = False
        
        while not done:
            mean, _ = actor.forward(state, std_fixed)
            action = np.clip(mean.flatten(), -1.0, 1.0)
            state, reward, done = env.time_step(action)
            ep_reward += reward
        
        total_reward += ep_reward
        if ep_reward >= 200:  # Survived full episode
            survived += 1
    
    print(f"{filename}: Avg reward = {total_reward/num_episodes:.2f}, Survived = {survived}/{num_episodes}")

# Run top 5
for i in range(1, 6):
    test_model(f'cartpole3d_best_{i}.npz', num_episodes=100)



env = Cartpole3D()
actor = Actor()
critic = Critic()
load_model(actor, critic, 'cartpole3d_best_1.npz')

state = env.reset()  # ← ADD THIS
for _ in range(10):
    mean, std = actor.forward(state, std_fixed)
    print(f"Mean: {mean}")
    action = np.clip(mean.flatten(), -1.0, 1.0)
    state, reward, done = env.time_step(action)
    print(f"Alpha: {state[4]:.4f}, Alpha_dot: {state[5]:.4f}")
