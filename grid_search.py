from align_fine_tuning import main as fine_tune_main
from matix import main as matix_main
from itertools import product

import subprocess

from os.path import join as ospj

import argparse
from tqdm import tqdm

from utils.dbe import dbe

from typing import Generator, List

# Function to generate all combinations of parameters
def generate_param_combinations(param_grid: dict) -> Generator[dict, None, None]:
    """
    Generate all combinations of parameters from a dictionary of lists.

    Args:
        param_grid (dict): A dictionary where keys are parameter names and values are lists of possible values.

    Yields:
        dict: A dictionary where keys are parameter names and values are parameter values.
    """
    keys, values = zip(*param_grid.items())

    for combination in product(*values):
        yield dict(zip(keys, combination))


def convert_dict_to_arglist(arg_dict: dict) -> List[str]:
    """
    Convert a dictionary of arguments to a list format expected by argparse.
    
    Args:
        arg_dict (dict): A dictionary where keys are argument names and values are argument values.

    Returns:
        List[str]: A list of arguments in the format ['--arg1', 'value1', '--arg2', 'value2', ...]
    """
    args_list = []

    for key, value in arg_dict.items():
        args_list.append(key)

        # Assuming all values need to be converted to string
        args_list.append(str(value))

    return args_list


def run_fine_tune(args):
    # Convert args_list to a string of command-line arguments
    args_str = " ".join(args)
    command = f'python align_fine_tuning.py {args_str}'
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    return result.stdout


def run_matrix(args):
    args_str = " ".join(args)
    command = f'python matix.py {args_str}'
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    return result.stdout


def main():
    save_location = ospj('grid_search_results', 'grid_search_results.txt')
    runs_per_param = 1

    param_grid = {
        # '--accumulation_steps': [1, 2, 4],  # Replace batch_size to simulate a larger batch size

        # Data augmentation
        # '--lr': [1e-4, 1e-3, 1e-2],
        # '--margin': [1.0, 0.5, 0.1],

        # '--augmented': [True, False],

        '--description': ['word', 'description'],

        # Loss function
        # '--loss_function': ['triplet', 'contrastive', 'simple'],

        # Learning rate scheduler
        # '--lr_scheduler': ['linear', 'cosine', 'cosine_warmup', 'none'],
        # '--cycle_mult': [1.0, 2.0, 3.0],
        # '--warmup_steps': [0, 5, 10],

        # Optimizer
        # '--weight_decay': [0.0, 0.1, 0.2],
        # '--eps': [1e-8, 1e-7, 1e-6],
        '--optimizer': ['adam', 'lamb'],
        # '--maximize': ['true', 'false'],
    }

    defaut_params_dict = {
        '--stop_patience': 10,
        '--split_name': 'fold_0_t',
        '--batch_size': 20,
        '--accumulation_steps': 4,
        '--print_info': 'false',
        # '--no_early_stopper': True,
        '--maximize': 'false'
    }

    defaut_params_array = [
        '--save'
    ]

    param_score = {}

    with open(save_location, 'w') as result_file:

        for index, param_combination in tqdm(enumerate(generate_param_combinations(param_grid)), desc='Runs', position=0):
            args_list = convert_dict_to_arglist(param_combination)
            args_list.extend(convert_dict_to_arglist(defaut_params_dict))
            args_list.extend(defaut_params_array)

            scores = []
            avrages = []

            for i in tqdm(range(runs_per_param), desc='Runs', position=0):
                # dbe(args_list)

                output = run_fine_tune(args_list)

                dbe(output)

                index *= 100
                idx = index + i

                # compare_main(fine_tune_path=model_path)

                # minimum_value, maximum_value, avrage_value = matix_main(args=args_list, model=model, index=idx)
                output = run_matrix(args_list)

                minimum_value, maximum_value, avrage_value = 0, 0, 0

                scores.append(minimum_value)
                avrages.append(avrage_value)

            avg_score = sum(scores) / len(scores)
            avg_avrage = sum(avrages) / len(avrages)

            param_combination_str = ",\n".join([f"{k}={v}" for k, v in param_combination.items()])
            
            result_file.write(f"{param_combination_str}\nAverage Score = {avg_score}\nAvrage Avrage = {avg_avrage}\n\n")



if __name__ == '__main__':
    main()
