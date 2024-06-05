import argparse
from os.path import join as ospj

def phosc_net_argparse(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    phosc_parser = parser.add_argument_group('PHOSC arguments')
    # phosc_parser.add_argument('--phosc_config', type=str, default=ospj('configs', 'phoscnet', 'default.yaml'), help='Path to the phoscnet configuration file')

    phosc_parser.add_argument('--phosc_model_name', type=str, default='ResNet18Phosc', help='model name')
    phosc_parser.add_argument('--phos_size', type=int, default=195, help='size of the phos')
    phosc_parser.add_argument('--phoc_size', type=int, default=1200, help='size of the phoc')
    phosc_parser.add_argument('--phos_layers', type=int, default=1, help='number of layers in the phos')
    phosc_parser.add_argument('--phoc_layers', type=int, default=1, help='number of layers in the phoc')
    phosc_parser.add_argument('--phosc_dropout', type=float, default=0.5, help='dropout')
    phosc_parser.add_argument('--phosc_version', type=str, choices=['en', 'no', 'ben'], default='ben', help='version of phosc')
    phosc_parser.add_argument('--phosc_language_name', type=str, default='Bengali', help='name of the language')
    phosc_parser.add_argument('--image_resize_x', type=int, default=1170, help='resize image x')
    phosc_parser.add_argument('--image_resize_y', type=int, default=414, help='resize image y')

    return parser