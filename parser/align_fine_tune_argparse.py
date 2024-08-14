import argparse
from os.path import join as ospj

def aling_fine_tune_argparse(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    align_fine_tune_parser = parser.add_argument_group('Align fine-tune arguments')

    # align_fine_tune_parser.add_argument('--align_fine_tune_config', type=str, default=ospj('configs', 'align', 'fine-tune.yml'), help='Path to the align fine-tune configuration file')

    align_fine_tune_parser.add_argument('--name', type=str, default='align-fine-tune', help='name of the experiment')

    align_fine_tune_parser.add_argument('--loss_func', choices=['triplet', 'contrastive', 'simple'], default='triplet', help='loss function for training')
    align_fine_tune_parser.add_argument('--accumulation_steps', type=int, default=4, help='number of steps to accumulate gradients')

    align_fine_tune_parser.add_argument('--description', choices=['word', 'description', 'phosc_number'], default='word', help='description to use for align fine-tuning')

    # align_fine_tune_parser.add_argument('--no_tqdm_bar', action='store_true', default=False, help='Show tqdm progress bar or not')

    return parser
    