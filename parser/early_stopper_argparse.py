import argparse
from os.path import join as ospj

def early_stopper_argparse(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    early_stopper_parser = parser.add_argument_group('Early stopper arguments')
    # early_stopper_parser.add_argument('--early_stopper_config', type=str, default=ospj('configs', 'early_stopper', 'default.yaml'), help='Path to the early stopper configuration file')

    early_stopper_parser.add_argument('--stop_patience', type=int, default=10, help='patience for early stopping')
    early_stopper_parser.add_argument('--verbose', action='store_true', default=False, help='verbose for early stopping')
    early_stopper_parser.add_argument('--save_every', type=int, default=10, help='save model every n epochs')
    early_stopper_parser.add_argument('--validate', action='store_true', default=False, help='validate model every n epochs')
    early_stopper_parser.add_argument('--save', action='store_true', default=False, help='save model every n epochs')

    return parser