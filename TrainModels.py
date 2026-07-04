import numpy as np

np.random.seed(100)




def log_probability(action, mean, std):
    return np.sum(-0.5 * ((action - mean) / std)**2 - np.log(std) - 0.5*np.log(2*np.pi), axis=-1)




episodes = 1000
time_steps_per_batch = 2048
update_epochs = 30
gamma = 0.99
lambda_gae = 0.95
clip_epsilon = 0.2
lr = 0.00005




def GAE_computation(rewards, values, dones, gamma, lambda_gae):
    advantages = np.zeros_like(rewards)
    gae = 0.0
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = 0.0
        else:
            next_value = values[t+1] * (1 - dones[t])
        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * lambda_gae * (1 - dones[t]) * gae 
        advantages[t] = gae
    returns = advantages + values
    return returns, advantages




def collect_trajectories(env, actor, critic, n_steps, std_fixed):
    states, actions, rewards, values, old_log_probs, dones = [], [], [], [], [], []
    state = env.reset()
    for _ in range(n_steps):
        mean, _ = actor.forward(state, std_fixed)
        action = np.random.normal(mean, std_fixed)
        action = np.clip(action, -1.0, 1.0)
        log_prob = log_probability(action, mean, std_fixed).item()
        value = critic.forward(state).squeeze()
        next_state, reward, done = env.time_step(action.flatten())
        states.append(state)
        values.append(value)
        old_log_probs.append(log_prob)
        dones.append(done)
        rewards.append(reward)
        actions.append(action.flatten())
        state = next_state
    return (np.array(states), np.array(values),
            np.array(old_log_probs).squeeze(), np.array(dones),
            np.array(rewards), np.array(actions))



def ppo_update(actor, critic, returns, states, old_log_probs, actions, advantages,
               epochs, batch_size, clip_epsilon, lr, std_fixed):
    n = len(states)
    for _ in range(epochs):
        idx = np.random.permutation(n)
        for start in range(0, n, batch_size):
            batch_idx = idx[start:start+batch_size]
            s = states[batch_idx]
            a = actions[batch_idx]
            old_lp = old_log_probs[batch_idx]
            ret = returns[batch_idx]
            adv = advantages[batch_idx].flatten()
            adv = (adv - adv.mean()) / (adv.std() + 1e-5)
            adv = adv.reshape(-1, 1)

            mean, _ = actor.forward(s, std_fixed)
            new_log_probs = log_probability(a, mean, std_fixed)
            diff = np.clip(new_log_probs - old_lp, -10.0, 10.0)
            ratio = np.exp(diff).reshape(-1, 1)

            surr1 = ratio * adv
            surr2 = np.clip(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * adv
            mask = (surr2 < surr1).astype(np.float64)
            d_ratio = adv * (1.0 - mask)
            d_log_prob = d_ratio * ratio
            d_mean = d_log_prob * (a - mean) / (std_fixed ** 2)

            critic.backward(s, ret, lr)
            actor.backward(d_mean, lr)



env = Cartpole3D()
actor = Actor()
critic = Critic()

best_rewards = []
std_fixed = 1.0

for ep in range(episodes):
    std_fixed = max(0.45, 1.0 * (0.998 ** ep))
    
    states, values, old_lp, dones, rewards, actions = collect_trajectories(
        env, actor, critic, time_steps_per_batch, std_fixed)
    returns, advantages = GAE_computation(rewards, values, dones, gamma, lambda_gae)
    ppo_update(actor, critic, returns, states, old_lp, actions, advantages,
               update_epochs, 64, clip_epsilon, lr, std_fixed)
    
    if ep % 50 == 0:
        for i in range(5):
            idx = np.random.randint(0, len(states))
            mean_sample, _ = actor.forward(states[idx], std_fixed)
            print(f"  State {i}: Mean: {mean_sample.flatten()}")
        print(f"Ep {ep}, Reward: {np.mean(rewards):.2f}, Std: {std_fixed:.3f}")

    avg_reward = np.mean(rewards)
    best_rewards.append((avg_reward, ep))
    best_rewards.sort(key=lambda x: x[0], reverse=True)
    best_rewards = best_rewards[:8]
    
    for rank, (reward, episode) in enumerate(best_rewards):
        if episode == ep:
            np.savez(f'cartpole3d_best_{rank+1}.npz',
                     W1=actor.W1, b1=actor.b1,
                     W2=actor.W2, b2=actor.b2,
                     W3_mean=actor.W3,
                     cW1=critic.W1, cb1=critic.b1,
                     cW2=critic.W2, cb2=critic.b2,
                     cW3=critic.W3, cb3=critic.b3)
            print(f"Ep {ep}: Saved #{rank+1}, Reward: {reward:.3f}")
            break
