import argparse
from os.path import join as ospj

def checkpoint_argparse(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    checkpoint_parser = parser.add_argument_group('Checkpoint arguments')

    checkpoint_parser.add_argument('--checkpoint_path', type=str, default='', help='Path to a checkpoint file to resume training')
    checkpoint_parser.add_argument('--ignore_checkpoint', action='store_true', help='Start training from scratch, ignore any checkpoints')

    return parser