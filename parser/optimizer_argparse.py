
import argparse
from os.path import join as ospj

def optimizer_argparse(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    optimizer_parser = parser.add_argument_group('Optimizer arguments')

    optimizer_parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay')
    optimizer_parser.add_argument('--eps', type=float, default=1e-8, help='eps for Adam optimizer')
    optimizer_parser.add_argument('--optimizer', choices=['adam', 'lamb', 'none'], default='adam', help='optimizer for training')
    optimizer_parser.add_argument('--maximize', action='store_true', help='maximize the metric')
    # optimizer_parser.add_argument('--maximize', type=bool, default=False, help='maximize the metric')

    return parser
