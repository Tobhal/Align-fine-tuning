import argparse
from os.path import join as ospj

def clip_fine_tune_argparse(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    clip_fine_tune_parser = parser.add_argument_group('Align fine-tune arguments')

    # clip_fine_tune_parser.add_argument('--clip_fine_tune_config', type=str, default=ospj('configs', 'clip', 'fine-tune.yml'), help='Path to the clip fine-tune configuration file')

    clip_fine_tune_parser.add_argument('--name', type=str, default='clip-fine-tune', help='name of the experiment')

    return parser