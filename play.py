import torch
import torch.nn as nn
import gymnasium as gym
import flappy_bird_gymnasium
import sys
import argparse
from model import FlappyNet

parser = argparse.ArgumentParser(description='Play Flappy Bird with a trained model')
parser.add_argument('--model', type=str, default='flappy_model.pt', 
                    help='Path to the model file to load (default: flappy_model.pt). Use --list to see available models.')
parser.add_argument('--greedy', action='store_true', default=True,
                    help='Use greedy action selection (default: True)')
parser.add_argument('--sample', action='store_true',
                    help='Use sampling from policy instead of greedy')
parser.add_argument('--list', action='store_true',
                    help='List all available model checkpoints')
args = parser.parse_args()

if args.list:
    import os
    import glob
    models = sorted(glob.glob('flappy_model*.pt'))
    if models:
        print("Available models:")
        for model in models:
            print(f"  - {model}")
    else:
        print("No models found. Run train.py first!")
    sys.exit(0)

greedy_mode = not args.sample

model_path = args.model
print(f"Loading model: {model_path}")

env = gym.make("FlappyBird-v0", render_mode="human")

model = FlappyNet()
if torch.cuda.is_available():
    model.cuda()

try:
    model.load_state_dict(torch.load(model_path))
    print(f"Successfully loaded model from {model_path}")
except FileNotFoundError:
    print(f"Error: Model file '{model_path}' not found!")
    sys.exit(1)

model.eval()

state, _ = env.reset()
done = False
total_reward = 0

while not done:
    state_tensor = torch.tensor(state, dtype=torch.float32)
    state_tensor = (state_tensor - 0.5) * 2
    
    if torch.cuda.is_available():
        state_tensor = state_tensor.cuda()
    
    with torch.no_grad():
        logits = model(state_tensor)
        if greedy_mode:
            action = torch.argmax(logits, dim=-1).item()
        else:
            probs = torch.softmax(logits, dim=-1)
            action_dist = torch.distributions.Categorical(probs)
            action = action_dist.sample().item()
    
    state, reward, terminated, truncated, _ = env.step(action)
    total_reward += reward
    done = terminated

env.close()
scaled_reward = total_reward * 10
steps_survived = int(total_reward / 0.1)
print(f"\nGame Over!")
print(f"Total Reward (actual): {total_reward:.2f}")
print(f"Total Reward (training scale): {scaled_reward:.2f}")
print(f"Steps Survived: {steps_survived}")
print(f"Mode: {'Greedy (max confidence)' if greedy_mode else 'Sampling (exploratory)'}")