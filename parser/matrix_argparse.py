import argparse
from os.path import join as ospj

def matrix_argparse(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    matrix_parser = parser.add_argument_group('Matrix arguments')

    # matrix_parser.add_argument('--num', type=int, default=0, help='number of matrices to generate')
    matrix_parser.add_argument('--nums', nargs='+', type=int, help='A list of model numbers to process')
    matrix_parser.add_argument('--evaluate', choices=['text', 'model'], default='text', help='evaluate text embeddings or model')
    matrix_parser.add_argument('--evaluation_model', choices=['fine-tuned', 'pre-trained'], default='fine-tuned', help='evaluate fine-tuned or pre-trained model')
    matrix_parser.add_argument('--checkpoint_name', type=str, default='best.pt', help='name of the checkpoint file')

    return parser