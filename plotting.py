
import numpy as np

from Environment import Cartpole3D
from ActorAndCritic import Actor, Critic


def load_model(actor, critic, filename):
    data = np.load(filename)
    actor.W1, actor.b1 = data['W1'], data['b1']
    actor.W2, actor.b2 = data['W2'], data['b2']
    actor.W3 = data['W3_mean']
    critic.W1, critic.b1 = data['cW1'], data['cb1']
    critic.W2, critic.b2 = data['cW2'], data['cb2']
    critic.W3, critic.b3 = data['cW3'], data['cb3']

env = Cartpole3D()
actor = Actor()
critic = Critic()
load_model(actor, critic, '/kaggle/input/datasets/aurorcys/winnermodel/CartPole3dWINNER2.npz')

# Hard reset
def reset_hard(self):
    self.x = 0.0; self.x_dot = 0.0
    self.y = 0.0; self.y_dot = 0.0
    self.alpha = np.random.uniform(-1, 1)
    self.alpha_dot = np.random.uniform(-0.3, 0.3)
    self.beta = np.random.uniform(-0.2, 0.2)
    self.beta_dot = np.random.uniform(-0.0, 0.0)
    return self._get_state()

Cartpole3D.reset_hard = reset_hard
env.reset = lambda: env.__class__.reset_hard(env)

# Collect full episode
state = env.reset()
states_all = []
actions_all = []
done = False
while not done:
    mean, _ = actor.forward(state, std_fixed=0.0)
    action = np.clip(mean.flatten(), -1.0, 1.0)
    state, _, done = env.time_step(action)
    states_all.append(state)
    actions_all.append(action)

states_all = np.array(states_all)
actions_all = np.array(actions_all)
T = len(states_all)

import plotly.graph_objects as go

# Sample every 3rd frame for performance
sample = 3
frames = []
for i in range(0, T, sample):
    x, y = states_all[i, 0], states_all[i, 1]
    alpha, beta = states_all[i, 4], states_all[i, 6]
    px = x + np.sin(alpha) * np.cos(beta)
    py = y + np.sin(alpha) * np.sin(beta)
    pz = np.cos(alpha)
    
    frames.append(go.Frame(
        data=[
            go.Scatter3d(x=[x], y=[y], z=[0], mode='markers', marker=dict(size=12, color='cyan')),
            go.Scatter3d(x=[x, px], y=[y, py], z=[0, pz], mode='lines', line=dict(width=6, color='red')),
            go.Scatter3d(x=states_all[:i, 0], y=states_all[:i, 1], z=np.zeros(i), 
                        mode='lines', line=dict(width=1, color='white'), opacity=0.3)
        ],
        name=str(i)
    ))

fig = go.Figure(
    data=[
        go.Scatter3d(
    x=[x], y=[y], z=[0], 
    mode='markers', 
    marker=dict(size=10, color='cyan', symbol='square'), 
    name='Cart'
),
        go.Scatter3d(
    x=[x, px], y=[y, py], z=[0, pz], 
    mode='lines', 
    line=dict(width=10, color='red'), 
    name='Pole'
),
        go.Scatter3d(x=[0], y=[0], z=[0], mode='lines', line=dict(width=1, color='white'), name='Trail')
    ],
    frames=frames
)

fig.update_layout(
    scene=dict(
        xaxis=dict(range=[-9, 9], autorange=False),
        yaxis=dict(range=[-9, 9], autorange=False),
        zaxis=dict(range=[-2, 2], autorange=False),
        aspectmode='manual',
        aspectratio=dict(x=1, y=1, z=1),
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z'
    ),
    title='PPO 3D CartPole — Trained Controller',
    updatemenus=[dict(type='buttons', showactive=False, 
                      buttons=[dict(label='Play', method='animate', 
                                    args=[None, dict(frame=dict(duration=20), fromcurrent=True)])])]
)

fig.show()
fig.write_html('cartpole3d.html')