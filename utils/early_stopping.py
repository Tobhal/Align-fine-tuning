import torch
import torch.nn as nn
import numpy as np
import os
import toml
import json

from argparse import ArgumentParser

from utils.dbe import dbe

class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    """
    def __init__(
            self, 
            save_path: os.PathLike, 
            loss: float,
            patience=7, 
            verbose=True, 
            save_every=5,
            model_arguments: dict = {}, 
            model_argument_parser: ArgumentParser=None, 
            save=True, 
            maximize=False,
            validate=False,
        ):
        """
        args:
            save_path (Path): Path to save the model and model arguments.
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: True
            save_every (int): Save the model every n epochs.
                            Default: 5
            model_arguments (dict): Arguments used to instantiate the model.
            save (bool): If True, saves the model and model arguments.
                            Default: True
            maximize (bool): If True, the metric to be maximized.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.early_stop = False
        self.min_loss = loss
        self.save_every = save_every
        self.model_arguments = model_arguments
        self.best_model_path = None
        self.save = save
        self.maximize = maximize
        self.validate = validate

        if self.save:
            # Initialize save_path with a new run-specific folder
            self.save_path = self.initialize_save_path(save_path)

            # Print the save path
            print(f"Saving model to {self.save_path}")

            args_save_path = os.path.join(self.save_path, f'model_args.toml')
        
            # Organize arguments by group
            grouped_args = {}
            for group in model_argument_parser._action_groups:
                group_dict = {action.dest: getattr(model_arguments, action.dest, None) for action in group._group_actions}
                if group.title not in ['positional arguments', 'optional arguments']:
                    grouped_args[group.title] = group_dict

            try:
                with open(args_save_path, 'w') as toml_file:
                    toml.dump(grouped_args, toml_file)
                print("Config saved successfully.")
            except Exception as e:
                print(f"Failed to save config: {e}")

            # Create csv file to store the training metrics
            self.metrics_save_path = os.path.join(self.save_path, 'metrics.csv')
            with open(self.metrics_save_path, 'w') as metrics_file:
                if self.validate:
                    metrics_file.write('epoch,train_loss,val_loss\n')
                else:
                    metrics_file.write('epoch,train_loss\n')
            

    def initialize_save_path(self, base_path: os.PathLike) -> os.PathLike:
        """
        Initializes the save path for the model and model arguments.

        args:
            base_path (path): Base path to save the model and model arguments.

        returns:
            run_save_path (path): Path to save the model and model arguments.
        """
        if not os.path.exists(base_path):
            os.makedirs(base_path)
            run_number = 1
        else:
            existing_runs = [int(folder) for folder in os.listdir(base_path) if folder.isdigit() and os.path.isdir(os.path.join(base_path, folder))]
            run_number = max(existing_runs) + 1 if existing_runs else 1

        run_save_path = os.path.join(base_path, str(run_number))

        if self.save:
            os.makedirs(run_save_path, exist_ok=True)

        return run_save_path

    def __call__(self, train_loss: float, val_loss, model: nn.Module, epoch: int) -> bool:
        """
        Determines if the training should stop based on the validation loss. And saves the model, model arguments and metrics.

        args:
            loss (float): Validation loss of the model.
            model (nn.Module): Model to save.
            epoch (int): Current epoch number.

        returns:
            should_stop (bool): If True, training should stop.
        """
        should_stop = False

        loss = val_loss if self.validate else train_loss

        # If loss is less than the previous minimum loss, save the model
        is_better = loss < self.min_loss if not self.maximize else loss > self.min_loss

        if is_better:
            self.save_checkpoint(loss, model, 'best', checkpoint_type='best')

            self.counter = 0
            self.min_loss = loss

        # If loss is not better, increment the counter
        else:
            self.counter += 1

            if self.verbose and self.patience != 0:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')

            """
            if self.counter >= self.patience and self.patience != 0:
                self.early_stop = True
                should_stop = True
            """

        # Save latest model as a checkpoint
        self.save_checkpoint(loss, model, 'latest', checkpoint_type='latest')

        # Save the model every n epochs
        if self.save_every is not None:
            if (epoch + 1) % self.save_every == 0:
                self.save_checkpoint(loss, model, f'epoch {epoch + 1}', checkpoint_type='epoch')

        # If the training should stop, print a message
        if should_stop:
            print('Early stopping')
            print(f'Best model saved at {self.best_model_path}')
            print(f'Best model validation loss: {self.min_loss:.6f}')

        if self.save:
            # Save loss to csv
            with open(self.metrics_save_path, 'a') as metrics_file:
                metrics_file.write(f'{epoch},{train_loss},{val_loss}\n')

        return should_stop


    def save_checkpoint(self, loss: float, model: nn.Module, name: str, checkpoint_type: str):
        '''
        Saves model and model arguments when validation loss decreases.
        
        args:
            loss (float): Validation loss of the model.
            model (nn.Module): Model to save.
            name (str): Name of the model to save.
            checkpoint_type (str): Type of checkpoint to save. Can be 'best' or 'epoch'.
        '''
        # If save is False, don't save the model
        if not self.save:
            return

        # Print a message if the validation loss decreases
        if self.verbose:
            if checkpoint_type == 'best':
                text = 'decreased' if not self.maximize else 'increased'

                loss_diff = abs(loss - self.min_loss)

                print(f'Loss {text}: {self.min_loss:.6f} -> {loss:.6f} | {loss_diff:.6f}. Saving model and arguments ...') 
            elif checkpoint_type == 'epoch':
                print(f'Saving model and arguments for epoch {name} ...')

        # Save the model
        model_save_path = os.path.join(self.save_path, f'{name}.pt')
        torch.save(model.state_dict(), model_save_path)
        
        if checkpoint_type == 'best':
            self.best_model_path = model_save_path
