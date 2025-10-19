import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import gymnasium as gym
import flappy_bird_gymnasium
from model import FlappyNet

num_cpus = torch.get_num_threads()
torch.set_num_threads(min(num_cpus, 8))
print(f"CPU threads set to: {torch.get_num_threads()}")

env = gym.make("FlappyBird-v0")

model = FlappyNet()
if torch.cuda.is_available():
    model.cuda()
    print("Using GPU with multi-threaded CPU support")
else:
    print("CUDA not available, using CPU with multi-threading")
    torch.set_num_threads(torch.get_num_threads())

optimizer = optim.Adam(model.parameters(), lr=1e-3)
gamma = 0.99
reward_scale = 10.0

num_episodes = 10000
best_reward = -float('inf')
best_episode = 0

for episode in range(num_episodes):
    state, _ = env.reset()
    done = False
    log_probs = []
    rewards = []

    while not done:
        state_tensor = torch.tensor(state, dtype=torch.float32)
        state_tensor = (state_tensor - 0.5) * 2
        
        if torch.cuda.is_available():
            state_tensor = state_tensor.cuda()
        
        logits = model(state_tensor)
        probs = torch.softmax(logits, dim=-1)
        action_dist = torch.distributions.Categorical(probs)
        
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        
        next_state, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated
        
        scaled_reward = reward * reward_scale
        
        log_probs.append(log_prob)
        rewards.append(scaled_reward)
        
        state = next_state

    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    returns = torch.tensor(returns, dtype=torch.float32)
    
    if torch.cuda.is_available():
        returns = returns.cuda()
    
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    
    loss = []
    for log_prob, G in zip(log_probs, returns):
        loss.append(-log_prob * G)
    loss = torch.stack(loss).sum()
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    total_reward = sum(rewards)
    print(f"Episode {episode+1}, Total Reward: {total_reward:.2f}")
    
    if total_reward > best_reward:
        best_reward = total_reward
        best_episode = episode + 1
        torch.save(model.state_dict(), 'flappy_model.pt')
        print(f"*** NEW BEST MODEL *** Episode {episode+1}, Reward: {total_reward:.2f}")
    
    if (episode + 1) % 1000 == 0:
        torch.save(model.state_dict(), f'flappy_model_checkpoint_{episode+1}.pt')
        print(f"Checkpoint saved at episode {episode+1}, Best so far: {best_reward:.2f} (ep {best_episode})")

env.close()
print(f"\nTraining complete!")
print(f"Best model: Episode {best_episode} with reward {best_reward:.2f}")