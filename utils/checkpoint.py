import torch
import os
from typing import Tuple, Optional, Any, Dict
from torch.optim import Optimizer
from torch.nn import Module
from torch.optim.lr_scheduler import _LRScheduler


def save_checkpoint(state: Dict[str, Any], filename: str = "checkpoint.pth.tar") -> None:
    """Save the training model checkpoint."""
    torch.save(state, filename)


def load_checkpoint(
    checkpoint_path: str,
    model: Module,
    optimizer: Optimizer,
    lr_scheduler: _LRScheduler,
    maximize: bool = False
) -> Tuple[int, float]:
    """Load checkpoint and return the epoch number and best validation loss."""
    if os.path.isfile(checkpoint_path):
        print(f"Loading checkpoint '{checkpoint_path}'")
        
        checkpoint = torch.load(checkpoint_path)
        
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']

        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

        print(f"Loaded checkpoint '{checkpoint_path}' (epoch {checkpoint['epoch']})")
        return start_epoch, best_val_loss
    else:
        print("No checkpoint found at '{}'".format(checkpoint_path))
        return 1, float('-inf') if maximize else float('inf')  # Start from the beginning if no checkpoint is found