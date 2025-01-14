# Standard library imports
import logging
import math
import os
import random
import sys
from enum import Enum
from os.path import join as ospj
from typing import Callable, Tuple, List
import copy

# Third-party imports
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.functional import cosine_similarity
from torchvision import transforms
from timm import create_model
from tqdm import tqdm
from transformers import AlignProcessor, AlignModel, AutoTokenizer
import argparse
import pynvml
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.nn import Module

# Local application/library specific imports
from data import dataset_bengali as dset
from data.dataset_bengali import ImageLoader
from flags import DATA_FOLDER, device
from modules import models, residualmodels
from modules.utils import set_phos_version, set_phoc_version, gen_shape_description, gen_shape_description_simple
from train_clip.utils.clip_utils import gen_word_objs_embeddings
from utils.dbe import dbe
from utils.early_stopping import EarlyStopping
from parser import phosc_net_argparse, dataset_argparse, early_stopper_argparse, aling_fine_tune_argparse, optimizer_argparse, lr_scheduler_argparse, checkpoint_argparse, slurm_argparse
from utils.utils import load_args
from utils.get_dataset import get_training_loader, get_validation_loader, get_test_loader, get_phoscnet
from utils.loss_functions import compute_triplet_margin_loss, compute_contrastive_loss, simple_loss
from utils.lamb_optimizer import Lamb
from modules.utils.utils import get_phosc_description, get_phosc_number_description
from utils.lr_schedulers.exploration import ExplorationOptimizationScheduler
from utils.checkpoint import save_checkpoint, load_checkpoint

pynvml.nvmlInit()


def create_file_with_job_id(save_path: os.PathLike, job_id: int, job_description: str):
    with open(ospj(save_path, f'slurm_{job_id}.txt'), 'w') as file:
        file.write(job_description)


def list_allocated_gpus():
    # Checks if CUDA is available at all
    if torch.cuda.is_available():
        # Count the number of GPUs available
        num_gpus = torch.cuda.device_count()
        print(f"Number of GPUs available: {num_gpus}")
        
        # List each GPU's ID and its properties
        for i in range(num_gpus):
            gpu_properties = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {gpu_properties.name}, Total Memory: {gpu_properties.total_memory / (1024 ** 3)} GB")
    else:
        print("No GPUs are available")


def get_gpu_memory_usage():
    """Get current and peak (max) memory usage of the GPU in gigabytes, including total system GPU memory usage."""
    device_id = 0  # Assuming we're querying the first GPU
    handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)

    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    allocated_memory = torch.cuda.memory_allocated(device_id)  # current memory allocated by PyTorch
    peak_memory = torch.cuda.max_memory_allocated(device_id)  # peak memory allocated by PyTorch
    total_memory = torch.cuda.get_device_properties(device_id).total_memory  # total memory of GPU

    # System-wide GPU memory usage
    system_allocated_memory = info.used  # total memory used by all processes

    # Convert bytes to gigabytes
    allocated_memory_gb = allocated_memory / (1024 ** 3)
    peak_memory_gb = peak_memory / (1024 ** 3)
    total_memory_gb = total_memory / (1024 ** 3)
    system_allocated_memory_gb = system_allocated_memory / (1024 ** 3)

    return {
        'Allocated Memory (GB)': allocated_memory_gb,
        'Peak Memory (GB)': peak_memory_gb,
        'Total Memory (GB)': total_memory_gb,
        'System Allocated Memory (GB)': system_allocated_memory_gb
    }


# Function to check if the model save path's directory exists
def verify_model_save_path(path):
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        print(f"Model save directory does not exist, creating: {directory}")
        os.makedirs(directory)


def create_learning_rate_fn(
    optimizer: Optimizer,
    train_ds_size: int,
    train_batch_size: int,
    num_train_epochs: int,
    num_warmup_steps: int,
    learning_rate: float,
    linear=False
):
    """Returns a PyTorch learning rate scheduler."""
    steps_per_epoch = train_ds_size // train_batch_size
    num_train_steps = steps_per_epoch * num_train_epochs

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        if linear:
            return max(
                0.0, float(num_train_steps - current_step) / float(max(1, num_train_steps - num_warmup_steps))
            )
        else:  # Cosine decay
            return 0.5 * (1 + np.cos(np.pi * (current_step - num_warmup_steps) / (num_train_steps - num_warmup_steps)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def create_cosine_annealing_lr_scheduler(
        optimizer: Optimizer, 
        T_0, 
        T_mult=1, 
        eta_min=0, 
        last_epoch=-1
    ):
    """Returns a PyTorch Cosine Annealing scheduler with Warm Restarts."""
    return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=T_0, T_mult=T_mult, eta_min=eta_min, last_epoch=last_epoch
    )


def custom_loss(image_features, text_features):
    # Assuming image_features and text_features are normalized
    similarity = torch.nn.functional.cosine_similarity(text_features, image_features)
    loss = torch.mean(1 - similarity)  # Penalize high similarity
    return loss


def normalize_features(features):
    return features / features.norm(dim=1, keepdim=True)


# Cross entropy helper function
def cross_entropy(logits, axis):
    logprobs = torch.log_softmax(logits, axis=axis)
    nll = torch.diag(logprobs)
    ce = -torch.mean(nll)
    return ce


def custom_loss_same_class(anchor_image_features, positive_text_features):
    # Ensure features are normalized
    image_features = F.normalize(anchor_image_features, dim=1)
    text_features = F.normalize(positive_text_features, dim=1)

    # Calculate similarity
    similarity = torch.matmul(image_features, text_features.T)

    # Compute CLIP loss
    loss = -((cross_entropy(similarity, axis=0) + cross_entropy(similarity, axis=1)) / 2)
    return loss


def custom_loss_different_class(anchor_image_features, negative_text_features):
    # Ensure features are normalized
    image_features = F.normalize(anchor_image_features, dim=1)
    text_features = F.normalize(negative_text_features, dim=1)

    # Calculate similarity
    similarity = torch.matmul(image_features, text_features.T)

    # Compute CLIP loss
    loss = 1 - ((cross_entropy(similarity, axis=0) + cross_entropy(similarity, axis=1)) / 2)
    return loss


def custom_triplet_loss(anchor_image_features, positive_text_features, negative_text_features, margin=1.0):
    # Calculate triplet loss
    loss = F.triplet_margin_loss(
        anchor_image_features,
        positive_text_features,
        negative_text_features,
        margin=margin
    )
    return loss


class Loss_method(Enum):
    DIFFRENT_SAME = 1
    CUSTOM_TRIPLET_LOSS = 2


def calc_loss(anchor_image_features, positive_text_features, negative_text_features, is_same_class: bool, loss_method: Loss_method):
    if loss_method == Loss_method.DIFFRENT_SAME:
        if is_same_class:
            return custom_loss_same_class(anchor_image_features, positive_text_features)
        else:
            return custom_loss_different_class(anchor_image_features, negative_text_features)

    elif loss_method == Loss_method.CUSTOM_TRIPLET_LOSS:
        return custom_triplet_loss(anchor_image_features, positive_text_features, negative_text_features)


def train_epoch(
        epoch: int, 
        train_loader: DataLoader, 
        base_model: Module, 
        processors: List,
        image_loader: ImageLoader, 
        loss_func: str, 
        optimizer: Optimizer,
        save_path: str,
        lr_scheduler: List[_LRScheduler]=None, 
        margin=1.0, 
        accumulation_steps=4,
        description='word'
    ):
    base_model.train()
    running_loss = 0.0
    
    num_gpus = torch.cuda.device_count()

    # train_bar = tqdm(train_loader, desc=f'TE: {epoch}', position=0, leave=True, disable=True)
    for i, batch in enumerate(train_loader):
        gpu_id = i % num_gpus  # Round-robin distribution of batches
        device = torch.device(f'cuda:{gpu_id}')

        print(f'Device = cuda:{gpu_id}')

        # Copy model to the current GPU
        model = copy.deepcopy(base_model).to(device)

        optimizer.zero_grad()

        *_, image_names, _, words = batch
        images = [image_loader(img_name) for img_name in image_names]

        if description == 'word':
            descriptions = words
        elif description == 'description':
            descriptions = [get_phosc_description(word) for word in words]
        elif description == 'phosc_number':
            description = [get_phosc_number_description(word) for word in words]
        else:
            raise ValueError('Invalid description')
        
        # Save 1 description to model dir
        description_example_file = ospj(save_path, 'description_example.txt')
        if not os.path.exists(description_example_file):
            with open(description_example_file, 'w') as description_file:
                description_file.write(f'{words[0]}\n{descriptions[0]}')
        
        # Prepare the inputs and get the model's output
        processor = processors[gpu_id]
        inputs = processor(text=descriptions, images=images, padding=True, return_tensors="pt")
        inputs = {name: tensor.to(device) for name, tensor in inputs.items()}

        outputs = model(**inputs)

        # Find unice classes
        unique_classes = set(words)
        class_to_index = {cls: idx for idx, cls in enumerate(unique_classes)}
        class_labels = torch.tensor([class_to_index[cls] for cls in words]).cuda(device)  # Ensure labels are on the right GPU

        # Calculate the loss
        logits_per_image = outputs.logits_per_image

        if loss_func == 'triplet':
            loss = compute_triplet_margin_loss(logits_per_image, class_labels, margin)
        elif loss_func == 'contrastive':
            loss = compute_contrastive_loss(logits_per_image, class_labels, margin)
        elif loss_func == 'simple':
            loss = simple_loss(logits_per_image)
        else:
            raise ValueError('Invalid loss function')

        loss = loss / accumulation_steps
        loss.backward()

        # Move gradients back to the base model before optimizer step
        if (i + 1) % accumulation_steps == 0:
            # Aggregate gradients from the current GPU model to the base model
            for base_param, param in zip(base_model.parameters(), model.parameters()):
                if param.grad is not None:  # Ensure param grad is not None
                    if base_param.grad is None:
                        base_param.grad = param.grad.clone().to('cuda:0')  # Copy grad to primary device
                    else:
                        base_param.grad += param.grad.clone().to('cuda:0')  # Sum grad to existing grad on primary device
                # If param.grad is None, do nothing; base_param.grad should remain None or existing value

            optimizer.step()
            optimizer.zero_grad()
            if lr_scheduler is not None:
                lr_scheduler.step()

            running_loss += loss.item() * accumulation_steps  # Correct for the scaling done by loss normalization

    average_loss = running_loss / len(train_loader)
    return average_loss


def validate_epoch(
        epoch: int, 
        val_loader: DataLoader, 
        model: Module,
        processor,
        image_loader: ImageLoader, 
        loss_func: str,
        save_path: str,
        margin=1.0, 
        description='word'
    ):
    model.eval()  # Set the model to evaluation mode
    running_loss = 0.0

    for i, batch in enumerate(val_loader):
        with torch.no_grad():  # No gradients needed
            *_, image_names, _, words = batch

            # Assuming each image is paired with a matching description
            images = [image_loader(img_name) for img_name in image_names]

            if description == 'word':
                descriptions = words
            elif description == 'description':
                descriptions = [get_phosc_description(word) for word in words]
            else:
                raise ValueError('Invalid description')
            
            unique_classes = set(words)  # Find the unique classes
            class_to_index = {cls: idx for idx, cls in enumerate(unique_classes)}  # Create a mapping

            class_indices = [class_to_index[cls] for cls in words]
            class_labels = torch.tensor(class_indices)

            # Prepare the inputs and get the model's output
            inputs = processor(text=descriptions, images=images, padding=True, return_tensors="pt")

            # Move the inputs to the device
            inputs = {name: tensor.to(device) for name, tensor in inputs.items()}

            model.to(device)

            outputs = model(**inputs)

            # Calculate the loss
            logits_per_image = outputs.logits_per_image

            if loss_func == 'triplet':
                loss = compute_triplet_margin_loss(logits_per_image, class_labels, margin)
            elif loss_func == 'contrastive':
                loss = compute_contrastive_loss(logits_per_image, class_labels, margin)
            elif loss_func == 'simple':
                loss = simple_loss(logits_per_image)
            else:
                raise ValueError('Invalid loss function')

            running_loss += loss.item()
            # val_bar.set_description(f'VE: {epoch} | Loss: {loss.item():.4f}')

    average_loss = running_loss / len(val_loader)

    # print(f'| loss {average_loss}')

    return average_loss


def main(_args=None):
    list_allocated_gpus()

    parser = argparse.ArgumentParser()

    parser = aling_fine_tune_argparse(parser)
    parser = phosc_net_argparse(parser)
    parser = dataset_argparse(parser)
    parser = early_stopper_argparse(parser)
    parser = optimizer_argparse(parser)
    parser = lr_scheduler_argparse(parser)
    parser = checkpoint_argparse(parser)
    parser = slurm_argparse(parser)

    # Parse arguments
    if _args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(_args)

    # Assuming torch.cuda.is_available() checks for GPU availability
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
    else:
        raise RuntimeError("This setup requires GPUs but none were found.")

    # Initialize models on each GPU
    align_model = AlignModel.from_pretrained("kakaobrain/align-base").to("cuda:0")
    align_processors = []
    # align_tokenizer = AutoTokenizer.from_pretrained("kakaobrain/align-base")

    for i in range(num_gpus):
        device = torch.device(f'cuda:{i}')

        align_processors.append(
            AlignProcessor.from_pretrained("kakaobrain/align-base")
        )

    train_loader, train_set = get_training_loader(args)
    validation_loader, _ = get_validation_loader(args)
    # test_loader, _ = get_test_loader(args, phosc_model)

    phosc_model = get_phoscnet(args, device)

    image_loader = ImageLoader(ospj(DATA_FOLDER, args.data_dir, args.split_name))

    optimizer = None
    lr_scheduler = None

    # Create optimizer for each model
    if args.optimizer in ['lamb', 'adam']:
        optimizer = Lamb(
            align_model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            adam=(args.optimizer == 'adam'),
            maximize=args.maximize,
        )
    elif args.optimizer == 'none':
        optimizer = None
    else:
        raise ValueError('Invalid optimizer')
    
    # Select learning rate scheduler
    if args.lr_scheduler == 'cosine':
        lr_scheduler = create_cosine_annealing_lr_scheduler(optimizer, T_0=10)
    elif args.lr_scheduler == 'cosine_warmup':
        lr_scheduler = create_learning_rate_fn(
            optimizer,
            len(train_set),
            args.batch_size,
            args.epochs,
            args.warmup_steps,
            args.lr,
            linear=False,  # set False to activate cosine annealing
        )
    elif args.lr_scheduler == 'linear':
        lr_scheduler = create_learning_rate_fn(
            optimizer,
            len(train_set),
            args.batch_size,
            args.epochs,
            args.warmup_steps,
            args.lr,
            linear=True,
        )
    elif args.lr_scheduler == 'exploration':
        lr_scheduler = ExplorationOptimizationScheduler(
            optimizer,
            patience=args.lr_patience,
            threshold=args.lr_threshold,
            reduction_factor=args.lr_reduction_factor,
            exploration_factor=args.lr_exploration_factor,
        )
    elif args.lr_scheduler == 'none':
        lr_scheduler = None

    save_path = ospj(args.save_dir, args.name, args.split_name)

    best_loss = None

    # Load checkpoint if there is any
    if args.ignore_checkpoint:
        start_epoch = 1
        best_loss = float('-inf') if args.maximize else float('inf')
    else:
        print('Loading checkpoint')
        checkpoint_path = save_path if args.checkpoint_path == None else args.checkpoint_path
        start_epoch, best_loss = load_checkpoint(checkpoint_path, align_model, optimizer, lr_scheduler, maximize=args.maximize)

    early_stopping = EarlyStopping(
        save_path=save_path,
        loss=best_loss,
        patience=args.stop_patience,
        verbose=args.verbose,
        save_every=args.save_every,
        model_arguments=args,
        model_argument_parser=parser,
        save=args.save,
        maximize=args.maximize,
        validate=args.validate
    )

    # Save slurm job to model folder
    create_file_with_job_id(save_path, args.slurm_job_id, args.slurm_job_desc)

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss = train_epoch(
            epoch               = epoch,
            train_loader        = train_loader,
            base_model          = align_model,
            processors          = align_processors,
            image_loader        = image_loader,
            loss_func           = args.loss_func,
            optimizer           = optimizer,
            lr_scheduler        = lr_scheduler,
            margin              = args.margin,
            accumulation_steps  = args.accumulation_steps,
            description         = args.description,
            save_path           = early_stopping.save_path,
        )

        val_loss = 0

        if args.validate:
            val_loss = validate_epoch(
                epoch           = epoch,
                val_loader      = validation_loader,
                model           = align_model,
                processor       = align_processors[0],
                image_loader    = image_loader,
                loss_func       = args.loss_func,
                margin          = args.margin,
                description     = args.description,
                save_path       = early_stopping.save_path,
            )

        if early_stopping(train_loss, val_loss, align_model, epoch):
            return early_stopping.min_loss, early_stopping.best_model_path

    return early_stopping.min_loss, early_stopping.best_model_path


if __name__ == '__main__':
    try:
        best_score, best_model_path = main()

        print(f'Best model validation loss: {best_score}')
        print(f'Best model path: {best_model_path}')

    except KeyboardInterrupt:
        print('Ctrl-C exit')