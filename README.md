# E2EC: End-to-End Coding Framework for Semantic Communication

This repository contains the PyTorch implementation of the paper ***Variable-Length End-to-End Joint Source-Channel Coding for Semantic Communication*** submitted to IEEE ICC 2026.

## Overview 

E2EC is an end-to-end coding framework that addresses the incompatibility between existing joint source-channel coding (JSCC) schemes and digital communication systems. The framework enables:
- *Variable-length discrete coding*: Direct compressing real-linear vectors into finite binary codebooks
- *End-to-end trainability*: Using policy gradient optimization across non-differentiable operations, e.g., sampling and channel with digital modulation
- *Semantic-oriented transmission*: Minimizing semantic distortion over noisy channels

## Key Features
- *Structural Decomposition*: Separate design of code length (`f_l`) and content (`f_z`) to enhance flexibility and coding efficiency for achieving information-theoretic bound
- *One-to-One Embedding*: Learned embeddings for each bit position to enable semantic reconstruction
- *Policy Gradient Optimization*: Gradient-based training across sampling and non-differentiable channel noise
- *Information Bottleneck Extension*: Rate-distortion tradeoff formed over noisy channels

## Requirements

The code requires the following dependencies:

- Python >= 3.7
- PyTorch >= 1.9.0
- torchvision >= 0.10.0
- numpy >= 1.19.0

## Installation

1. Clone the repository:
```bash
git clone https://github.com/SamuChamp/E2ECframework.git
cd E2ECframework
```

2. Install the required dependencies:
```bash
pip install torch torchvision numpy
```

## Default Configuration

The default hyperparameters are configured in `config.py`.

## Usage

### Quick Start

Run training and evaluation with default settings:
```bash
python main.py
```

### Custom Training

You can customize hyperparameters via command-line arguments:

```bash
# Train with different rate-distortion tradeoff multiplier
python main.py --lam 1e-5

# Train with different BSC error probability
python main.py --p-e 0.01

# Train with different maximum code length
python main.py --max-length 32

# Train with custom settings
python main.py --max-length 128 --lam 5e-7 --p-e 0.05 --epochs 150 --bs 256
```

### Save and Load Models

```bash
# Train and save model
python main.py --save-model checkpoint --model-dir ./checkpoints/
```

### Evaluation Only

```bash
# Evaluate a trained model
python main.py --train False --eval True --load-model path/to/model.pth
```

## Project Structure

```
E2ECframework/
|- main.py              # Main training and evaluation script
|- config.py            # Configuration and argument parser
|- model/
   |- __init__.py       
   |- encoder.py        # E2EC encoder (length + content modules)
   |- decoder.py        # E2EC decoder (embedding + classifier)
   |- data.py           # Data loader for MNIST
|- utils/
   |- __init__.py       
   |- utils.py          # Utility functions (channel, truncation, MI estimation)
   |- logs/             # Training logs (auto-generated)
   |- checkpoints/      # Saved models (optional)
|- README.md            # This file
```

## Model

### Encoder
- *Length* (`f_l`): FFN that outputs categorical distribution over code lengths
- *Content* (`f_z`): FFN that outputs a product of Bernoulli distributions for each code
- *Regularizer*: Auxiliary Gaussian latent code for training stability and grounding

### Decoder
- *Embedding* (`g_x`): One-to-one embedding for each bit position (0, 1, or truncated)
- *Classifier* (`g_y`): FFN that predicts semantic labels from embedded representations

### Channel
- *Binary Symmetric Channel (BSC)*: Bit-flipping with error probability `p_e`
