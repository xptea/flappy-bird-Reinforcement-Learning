# Flappy Bird Reinforcement Learning Agent

A complete Python project that trains a neural network to autonomously play Flappy Bird using reinforcement learning (REINFORCE algorithm).

## Features

- **REINFORCE Algorithm**: Policy gradient method for training the agent
- **Optimized Neural Network**: FlappyNet with 180 input features, 2 hidden layers (64 neurons each), ReLU activations
- **GPU/CPU Support**: Automatic detection and utilization of CUDA GPU with multi-threaded CPU fallback
- **Advanced Training**: State normalization, reward scaling, baseline adjustment for stable learning
- **Multiple Checkpoints**: Saves models every 1000 episodes plus the best performing model
- **Flexible Playback**: Greedy or sampling modes for testing trained agents
- **Comprehensive Evaluation**: Detailed performance metrics and comparison tools

## Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (optional, CPU fallback available)
- uv package manager

### Setup
```bash
# Clone the repository
git clone <your-repo-url>
cd flappy-bird

# Install dependencies with uv
uv sync

# Activate virtual environment
uv run python --version  # Verify setup
```

## Usage

### Training
Train the agent for maximum performance:
```bash
uv run python train.py
```
- Trains for 10,000 episodes
- Saves checkpoints every 1,000 episodes
- Automatically saves the best performing model as `flappy_model.pt`
- Uses GPU acceleration when available

### Playing
Test trained models with visual gameplay:

```bash
# Play with the best model (greedy mode)
uv run python play.py

# Play with a specific checkpoint
uv run python play.py --model flappy_model_checkpoint_5000.pt

# List all available models
uv run python play.py --list

# Use sampling mode instead of greedy
uv run python play.py --sample
```

## Model Architecture

### FlappyNet
- **Input**: 180 features from Flappy Bird game state
- **Hidden Layer 1**: 64 neurons, ReLU activation
- **Hidden Layer 2**: 64 neurons, ReLU activation
- **Output**: 2 logits (flap/no-flap actions)

### Training Details
- **Algorithm**: REINFORCE (Monte Carlo Policy Gradient)
- **Optimizer**: Adam (learning rate: 0.001)
- **Discount Factor**: γ = 0.99
- **Reward Processing**:
  - Scale rewards by 10x for stronger learning signal
  - Normalize returns with baseline adjustment
- **State Processing**: Normalize inputs to [-1, 1] range
- **Episodes**: 10,000 total training episodes

## Performance

The agent learns to navigate Flappy Bird obstacles autonomously. Key metrics:
- **Survival Time**: Measured in game steps
- **Training Reward**: Scaled reward during training
- **Best Model**: Automatically tracked and saved

Example performance:
- Early training: ~100-200 steps survival
- Mid training: ~500-1000 steps survival
- Late training: ~2000+ steps survival (varies by training run)

## Dependencies

- `flappy-bird-gymnasium`: Flappy Bird environment
- `gymnasium`: Reinforcement learning framework
- `torch`: PyTorch deep learning library
- `numpy`: Numerical computations

## Project Structure

```
flappy-bird/
├── train.py          # Training script with REINFORCE algorithm
├── play.py           # Playback script with model selection
├── model.py          # FlappyNet neural network definition
├── pyproject.toml    # Project configuration and dependencies
├── flappy_model.pt   # Best trained model (generated)
└── flappy_model_checkpoint_*.pt  # Training checkpoints (generated)
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is open source. Feel free to use and modify as needed.

## Acknowledgments

- Built using the `flappy-bird-gymnasium` environment
- Implements REINFORCE algorithm from reinforcement learning literature
- Optimized for both CPU and GPU training</content>
<parameter name="filePath">c:\Users\webst\OneDrive\Desktop\flappy-bird\README.md
