import yaml
import copy
import math

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

import numpy as np

from tqdm import tqdm

from cosine_annealing_warmup import CosineAnnealingWarmupRestarts

from flags import device, DATA_FOLDER

from os.path import join as ospj

from clip.clip import tokenize

import clip

from parser import train_clip_argparse, phosc_net_argparse, dataset_argparse, early_stopper_argparse
from utils.dbe import dbe
from utils.early_stopping import EarlyStopping

from train_clip.utils.clip_utils import gen_word_objs_embeddings

from data.dataset_bengali import ImageLoader
from torch.utils.data import DataLoader

# CLIP
from train_clip.models.model import CLIP

from typing import List, Tuple

from PIL import Image

from utils.get_dataset import (
    get_phoscnet,
    get_training_loader,
    get_validation_loader,
)


def num_training_steps(
        train_dataloader: DataLoader, 
        max_epochs: int, 
        batch_size: int, 
        accumulate_grad_batches=1
    ) -> int:
    """
    Calculate the total number of training steps based on the number of epochs, the batch size, and the number of gradient accumulation steps.

    Args:
        train_dataloader(Dataloader): The training dataloader.
        max_epochs(Int): The maximum number of epochs.
        batch_size(Int): The batch size.
        accumulate_grad_batches(Int): The number of gradient accumulation steps.

    Returns:
        total_steps: The total number of training steps.
    """    
    dataset_size = len(train_dataloader.dataset)

    # Effective batch size
    effective_batch_size = batch_size * accumulate_grad_batches
    total_steps = (dataset_size // effective_batch_size) * max_epochs

    return total_steps


def train_step_batch(
        images: List[Image.Image], 
        text_feat: torch.Tensor, 
        class_labels: torch.Tensor,  # Add this to pass class labels
        batch_size: int,
        clip_model: CLIP, 
        optimizer: torch.optim.Optimizer, 
        lr_scheduler: torch.optim.lr_scheduler.LambdaLR,
        margin: float = 1.0
    ) -> float:
    """
    Train step for one batch.

    args:
        images (List[Image.Image]): The images.
        text_feat (torch.Tensor): Text features.
        class_labels (torch.Tensor): The class labels.
        batch_size (int): The batch size.
        clip_model (CLIP): The CLIP model.
        optimizer (torch.optim.Optimizer): The optimizer.
        lr_scheduler (torch.optim.lr_scheduler.LambdaLR): The learning rate scheduler.
        margin (float): The margin for the triplet loss.

    returns:
        total_loss (float): The total loss for the batch.
    """    
    # Transform setup
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Process images
    images = torch.stack([transform(img) for img in images])

    # Split images into batches
    image_emb = torch.chunk(images, math.ceil(len(images) / batch_size))

    # Normalize image embeddings after encoding
    ims = [F.normalize(clip_model.encode_image(batch), dim=1) for batch in image_emb]
    ims = torch.cat(ims)

    # Normalize text features after averaging across the token dimension
    txt = F.normalize(text_feat, dim=1)
    txt = txt.squeeze(1)

    # Compute similarity scores between images and texts
    image_logits = ims @ txt.t() * clip_model.logit_scale.exp()

    # Mask for identifying pairs of the same class
    same_class_mask = class_labels.unsqueeze(1) == class_labels.unsqueeze(0)

    # Positive pairs: Maximize similarity for pairs of the same class
    positive_scores = []
    for i in range(class_labels.size(0)):
        # Mask to select same-class items, excluding self-comparison
        mask = (class_labels == class_labels[i]) & (torch.arange(class_labels.size(0)) != i)
        if mask.any():
            positive_scores.append(image_logits[i][mask].max())
        else:
            # Handle the case where there are no other same-class items
            positive_scores.append(torch.tensor(0.0, device=image_logits.device))

    if positive_scores:
        positive_pairs = torch.stack(positive_scores)
    else:
        # Fallback or alternative handling if no positive pairs are found
        positive_pairs = torch.tensor([0.0], device=image_logits.device)

    # Negative pairs: For each anchor, take the closest negative that's not of the same class
    negative_mask = ~same_class_mask
    max_negative_logits = torch.where(negative_mask, image_logits, torch.tensor(float('-inf')).to(image_logits.device))
    negative_pairs = max_negative_logits.max(dim=1)[0]

    # Compute triplet loss
    total_loss = F.relu(positive_pairs - negative_pairs + margin).mean()

    # Backpropagation
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if lr_scheduler is not None:
        lr_scheduler.step()

    # Clamp the logit scale
    clip_model.logit_scale.data = clip_model.logit_scale.data.clamp(-np.log(100), np.log(100))

    return total_loss.item()


def train_one_epoch(
        epoch: int, 
        train_loader: DataLoader, 
        clip_model: CLIP,
        image_loader: ImageLoader, 
        batch_size: int,
        optimizer: torch.optim.Optimizer, 
        lr_scheduler: torch.optim.lr_scheduler.LambdaLR,
        margin: float = 1.0,
        pos=0
    ) -> Tuple[float, float]:
    """
    Train one epoch.

    Args:
        epoch(int): The current epoch.
        train_loader(DataLoader): The training dataloader.
        clip_model(CLIP): The CLIP model.
        image_loader(ImageLoader): The image loader.
        optimizer(torch.optim.Optimizer): The optimizer.
        lr_scheduler(torch.optim.lr_scheduler.LambdaLR): The learning rate scheduler.

    Returns:
        loss(float): The loss for the epoch.
        acc(float): The accuracy for the epoch.
    """
    clip_model.train()
    running_loss = 0

    # Iterate over the training loader
    train_bar = tqdm(train_loader, desc=f'Training epoch {epoch + 1}', position=pos, leave=False)
    for batch in train_bar:
        optimizer.zero_grad()

        *_, image_names, _, words = batch
        
        images = [image_loader(image) for image in tqdm(image_names, position=1, desc='Processing Images', leave=False)]

        emb_descriptions = [gen_word_objs_embeddings(description, clip_model) for description in tqdm(words, position=2, desc='Generating Embeddings', leave=False)]
        emb_descriptions = torch.stack(emb_descriptions)

        unique_classes = set(words)  # Find the unique classes
        class_to_index = {cls: idx for idx, cls in enumerate(unique_classes)}  # Create a mapping

        class_indices = [class_to_index[cls] for cls in words]
        class_labels_tensor = torch.tensor(class_indices)

        temp_loss = train_step_batch(images, emb_descriptions, class_labels_tensor, batch_size, clip_model, optimizer, lr_scheduler, margin)

        running_loss += temp_loss

    loss = running_loss / len(train_loader)
    # loss.backward()

    optimizer.step()
    lr_scheduler.step()

    return loss


def validation_step_batch(
        images: List[Image.Image], 
        text: List[str], 
        clip_model: CLIP, 
        batch_size: int
    ) -> Tuple[float, float]:
    """
    Validation step for one batch.

    Args:
        image(List[Image.Image]): The image.
        text(List[str]): The text.
        clip_model(CLIP): The CLIP model.
        batch_size(int): The batch size.

    Returns:
        loss(float): The loss for the batch.
        acc(float): The accuracy for the batch.
    """
    loss, acc = 0, 0

    n = math.ceil(len(images) // batch_size)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    images = torch.stack([transform(img) for img in images])

    # Tokenize each text sample
    text_tokenized = [tokenize(txt).squeeze(0) for txt in text]
    text_tensor = torch.stack(text_tokenized)

    n = n if n > 0 else 1

    image_emb = torch.chunk(images, n)
    text_emb = torch.chunk(text_tensor, n)

    # Compute similarity scores between images and texts
    with torch.no_grad():
        ims = [F.normalize(clip_model.encode_image(img), dim=1) for img in image_emb]
        txt = [F.normalize(clip_model.encode_text(t), dim=1) for t in text_emb]

        ims = torch.cat(ims)
        txt = torch.cat(txt)

        # If the input is a single image and text, we need to convert it to a list
        if len(ims.shape) == 3:
            ims = list(ims)
            txt = list(txt)
        else:
            ims = [ims]
            txt = [txt]

        # Compute similarity scores between images and texts
        image_logits = torch.cat(ims) @ torch.cat(txt).t() * clip_model.logit_scale.exp()
        ground_truth = torch.arange(len(image_logits)).long().to(image_logits.device)

        # Compute loss
        loss = (F.cross_entropy(image_logits, ground_truth) + F.cross_entropy(image_logits.t(), ground_truth)).div(2)

        # Compute accuracy
        acc_i = (torch.argmax(image_logits, 1) == ground_truth).sum()
        acc_t = (torch.argmax(image_logits, 0) == ground_truth).sum()

        loss = loss / len(ims)
        acc = (acc_i + acc_t) / 2 / len(images) / len(ims)

    return loss, acc


def validation_one_epoch(
        epoch: int, 
        val_loader: DataLoader, 
        clip_model: CLIP, 
        image_loader: ImageLoader,
        pos=0
    ) -> Tuple[float, float]:
    """
    Validation step for one epoch.

    Args:
        epoch(int): The current epoch.
        val_loader(DataLoader): The validation dataloader.
        clip_model(CLIP): The CLIP model.
        image_loader(ImageLoader): The image loader.

    Returns:
        avg_loss(float): The average loss for the epoch.
        avg_acc(float): The average accuracy for the epoch.
    """
    clip_model.eval()
    running_loss = 0
    running_acc = 0

    # Iterate over the validation loader
    val_bar = tqdm(val_loader, desc=f'Validation epoch {epoch + 1}', position=pos, leave=False)
    for batch in val_bar:
        *_, image_names, _, descriptions = batch

        # preprocess images and descriptions
        images = [image_loader(image) for image in tqdm(image_names, position=pos+1, desc='Processing Images', leave=False)]
        descriptions = [description for description in descriptions]

        temp_loss, temp_acc = validation_step_batch(images, descriptions, clip_model, len(batch))

        running_loss += temp_loss.item()
        running_acc += temp_acc.item()

    avg_loss = running_loss / len(val_loader)
    avg_acc = running_acc / len(val_loader)

    return avg_loss, avg_acc


def main(args=None, pos=0) -> Tuple[float, CLIP]:
    # Setup arguments
    parser = argparse.ArgumentParser()

    parser = train_clip_argparse(parser)
    parser = phosc_net_argparse(parser)
    parser = dataset_argparse(parser)
    parser = early_stopper_argparse(parser)

    if args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args)

    # Set up clip model   
    with open(args.config_dir) as conf:
        config = yaml.safe_load(conf)[args.clip_model_name]

    clip_model = CLIP(
        **config
    )
    
    # Define phosc model
    phosc_model = get_phoscnet(args, device)

    # Get dataset
    train_loader, _ = get_training_loader(args)
    validation_loader, _ = get_validation_loader(args)

    optimizer = torch.optim.AdamW(
        clip_model.parameters(),
        lr=args.lr,
        betas=(
            0.9,
            0.98
        ),
        eps=args.eps,
        weight_decay=args.weight_decay
    )

    lr_scheduler = CosineAnnealingWarmupRestarts(
        optimizer,
        first_cycle_steps=num_training_steps(train_loader, args.num_epochs, args.batch_size),
        cycle_mult=args.cycle_mult,
        max_lr=args.lr,
        min_lr=0,
        warmup_steps=args.warmup_steps
    )

    image_loader = ImageLoader(ospj(DATA_FOLDER, args.data_dir, args.split_name))

    early_stopping = EarlyStopping(
        save_path=ospj('models', 'trained_clip', args.split_name, args.name),
        patience=args.stop_patience,
        verbose=args.verbose,
        save_every=args.save_every,
        model_arguments=args,
        save=False
    )

    for epoch in range(args.num_epochs):
        train_loss = train_one_epoch(
            epoch,
            train_loader, 
            clip_model,
            image_loader,
            args.batch_size,
            optimizer,
            lr_scheduler,
            pos=pos
        )

        """
        validation_loss, _ = validation_one_epoch(
            epoch,
            validation_loader,
            clip_model,
            image_loader,
            pos=pos
        )
        """

        if early_stopping(train_loss, clip_model, epoch):
            return early_stopping.best_score, clip_model


if __name__ == '__main__':
    main()
