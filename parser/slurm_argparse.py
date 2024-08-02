import argparse
from os.path import join as ospj


def slurm_argparse(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    slurm_parser = parser.add_argument_group('Slurm arguments')

    slurm_parser.add_argument('--slurm_job_id', type=int, default=0, help='ID of slurm job')
    slurm_parser.add_argument('--slurm_job_desc', type=str, default='No description given', help='Description for slurm job')

    return parser