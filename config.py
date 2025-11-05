import argparse
from utils.utils import add_flags_from_config


config_args = {
    'data_config': {
        'dataset': ('mnist', 'which dataset to use'),
    },
    'model_config': {
        'embd-dim': (64, 'embedding dimension'),
        'max-length': (64, 'maximum code length for variable-length coding'),
        'input-dim': (784, 'input dimension (28x28 for MNIST)'),
        'output-dim': (10, 'output dimension (number of classes)'),
    },
    'training_config': {
        'lr': (1e-4, 'learning rate for encoder and decoder'),
        'bs': (128, 'batch size'),
        'epochs': (100, 'number of training epochs'),
        'seed': (114514, 'seed for training'),
        'update-freq': (1, 'update frequency of parameters'),
        'log-freq': (100, 'printing frequency of train/val metrics (in epochs)'),
        'eval-freq': (50, 'computing frequency of val metrics (in epochs)'),
        'grad-clip': (None, 'max norm for gradient clipping, or None for no gradient clipping'),
        'scal': (1, 'scaling coefficient to implement regularization'),
        'lam': (1e-6, 'Lagrange multiplier for rate-distortion tradeoff'),
        'p-e': (1e-1, 'channel error probability'),
        'T': (3, 'temperature for knowledge distillation'),
    },
    'mode_config': {
        'train': (True, 'whether to train the model'),
        'eval': (True, 'whether to evaluate the model'),
        'load-model': (None, 'path to load model from (None for no loading)'),
        'save-model': (None, 'path to save model to (None for no saving)'),
        'model-dir': (None, 'directory to save/load models'),
    },
}

parser = argparse.ArgumentParser()
for _, config_dict in config_args.items():
    parser = add_flags_from_config(parser, config_dict)
    