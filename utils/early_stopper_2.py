import torch
import torch.nn as nn
import numpy as np
import os
import json

class EarlyStopping:
    def __init__(self, save_path: os.PathLike, patience=7, verbose=True, save_every=5, model_arguments: dict = {}, save=True, max_epochs=100):
        """
        Args:
            save_path (Path): Path to save the model and model arguments.
            patience (int): How long to wait after the last time validation loss improved. Default: 7.
            verbose (bool): If True, prints a message for each validation loss improvement. Default: True.
            save_every (int): Save the model every n epochs. Default: 5.
            model_arguments (dict): Arguments used to instantiate the model.
            save (bool): If True, saves the model and model arguments. Default: True.
            max_epochs (int): Maximum number of epochs to run the training for. Default: 100.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.save_every = save_every
        self.model_arguments = model_arguments
        self.best_model_path = None
        self.save = save
        self.epoch = 0
        self.max_epochs = max_epochs
        
        self.save_path = self.initialize_save_path(save_path)
        print(f"Saving model to {self.save_path}")
        
        args_save_path = os.path.join(self.save_path, 'model_args.json')
        if self.save:
            with open(args_save_path, 'w') as json_file:
                json.dump(self.model_arguments, json_file, indent=4)

    def __iter__(self):
        """
        Initialize the iteration. Resets the epoch counter and returns the instance.
        """
        self.epoch = 0  # Reset epoch counter for new training run
        return self

    def __next__(self):
        """
        Proceed to the next epoch, if early stopping criteria are not met.
        """
        if self.epoch >= self.max_epochs or self.early_stop:
            raise StopIteration
        self.epoch += 1
        return self.epoch

    def step(self, val_loss: float, model: nn.Module):
        """
        Should be called at the end of each epoch to determine if training should continue.
        
        Args:
            val_loss (float): Validation loss for the current epoch.
            model (nn.Module): The model being trained.
        """
        score = -val_loss

        if self.best_score is None or score > self.best_score:
            self.best_score = score
            self.save_checkpoint(val_loss, model, 'best', checkpoint_type='best')
            self.best_model_path = os.path.join(self.save_path, 'best.pt')
            self.counter = 0  # reset counter if validation loss has decreased
        else:
            self.counter += 1  # increase counter if validation loss has not decreased
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True

        # Save the model every n epochs
        if self.epoch % self.save_every == 0:
            self.save_checkpoint(val_loss, model, f'epoch {self.epoch}', checkpoint_type='epoch')

        if self.early_stop:
            print('Early stopping')
            print(f'Best model saved at {self.best_model_path}')
            print(f'Best model validation loss: {-self.best_score}')
