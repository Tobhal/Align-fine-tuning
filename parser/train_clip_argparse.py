import argparse
from os.path import join as ospj

def train_clip_argparse(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    train_clip_parser = parser.add_argument_group('Train clip arguments')
    
    # Model
    train_clip_parser.add_argument('-n', '--name', type=str, default='Clip trained')
    train_clip_parser.add_argument('--clip_model_name', type=str, default='ViT-B/32')

    # Config
    train_clip_parser.add_argument(
        '--config_dir', 
        type=str, 
        default=ospj('train_clip', 'models', 'configs', 'ViT.yaml')
    )

    # Training
    train_clip_parser.add_argument('--num_epochs', type=int, default=100)

    return parser

