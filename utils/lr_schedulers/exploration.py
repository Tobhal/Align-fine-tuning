import torch


class ExplorationOptimizationScheduler(torch.optim.lr_scheduler._LRScheduler):
    """
    Exploration Optimization Scheduler

    This is a learning rate scheduler that reduces the learning rate by a factor of 'reduction_factor' when the loss
    does not decrease significantly for 'patience' epochs. The learning rate is increased by a factor of
    'exploration_factor' when the loss decreases significantly.

    Args:
        optimizer (torch.optim.Optimizer): Optimizer for which the learning rate will be adjusted.
        patience (int): Number of epochs to wait before reducing the learning rate.
        threshold (float): Threshold for significant loss decrease.
        reduction_factor (float): Factor by which the learning rate will be reduced.
        exploration_factor (float): Factor by which the learning rate will be increased.
        last_epoch (int): The index of the last epoch. Default: -1.
    """

    def __init__(self, optimizer, patience=5, threshold=0.01, reduction_factor=0.5, exploration_factor=1.5, last_epoch=-1):
        self.patience = patience
        self.threshold = threshold
        self.reduction_factor = reduction_factor
        self.exploration_factor = exploration_factor
        self.num_bad_epochs = 0
        self.last_loss = None  # Changed from float('inf') to None to avoid reducing LR on init
        super().__init__(optimizer, last_epoch, verbose=False)

    def step(self, loss=None):
        if loss is not None:
            if self.last_loss is None:
                # First call to step() should setup the last_loss without changing LR
                self.last_loss = loss
            else:
                if loss + self.threshold < self.last_loss:
                    # Loss is decreasing significantly
                    self.last_loss = loss
                    self.num_bad_epochs = 0
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] *= self.reduction_factor
                else:
                    self.num_bad_epochs += 1
                    if self.num_bad_epochs >= self.patience:
                        # No significant improvement in 'patience' epochs
                        self.last_loss = loss
                        self.num_bad_epochs = 0
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] *= self.exploration_factor
        else:
            # This is the call from __init__
            # Ideally do nothing here since we don't want to change LR without any loss provided
            pass