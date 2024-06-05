import argparse
from os.path import join as ospj

def lr_scheduler_argparse(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    learning_rate_parser = parser.add_argument_group('Learning rate scheduler arguments')
    learning_rate_parser.add_argument('--cycle_mult', type=float, default=1.0, help='cycle multiplier')
    learning_rate_parser.add_argument('--warmup_steps', type=int, default=0, help='warmup steps')
    learning_rate_parser.add_argument('--lr_scheduler', choices=['linear', 'cosine', 'cosine_warmup', 'exploration', 'none'], default='cosine', help='learning rate scheduler')

    learning_rate_parser.add_argument('--lr_patience', type=int, default=5, help='patience for lr scheduler')
    learning_rate_parser.add_argument('--lr_threshold', type=float, default=0.01, help='threshold for lr scheduler')
    learning_rate_parser.add_argument('--lr_reduction_factor', type=float, default=0.5, help='reduction factor for lr scheduler')
    learning_rate_parser.add_argument('--lr_exploration_factor', type=float, default=1.5, help='exploration factor for lr scheduler')

    return parser