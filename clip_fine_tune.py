# Standard library imports, sorted alphabetically
import argparse
import logging
import math
import os
from enum import Enum
from os.path import join as ospj
from typing import Any, List, Tuple

# Third-party imports, sorted alphabetically
import clip
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torch.optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

# Application-specific imports, sorted alphabetically
from data.dataset_bengali import ImageLoader
from flags import DATA_FOLDER, device
from parser import (
    clip_fine_tune_argparse,
    dataset_argparse,
    early_stopper_argparse,
    phosc_net_argparse,
)
from train_clip.models.model import CLIP
from train_clip.utils.clip_utils import gen_word_objs_embeddings
from utils.early_stopping import EarlyStopping
from utils.get_dataset import (
    get_phoscnet,
    get_training_loader,
    get_validation_loader,
)
from utils.dbe import dbe


def setup_logging(log_file_path: os.PathLike):
    """
    Setup logging for the training process.
    
    args:
        log_file_path (os.PathLike): Log file path.
    """
    logging.basicConfig(filename=log_file_path, level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s', 
                        datefmt='%Y-%m-%d %H:%M:%S')
    
    logging.info("Initialized logging")


# Function to check if the model save path's directory exists
def verify_model_save_path(path: os.PathLike):
    """
    Verify if the model save path exists, if not, create it.

    args:
        path (os.PathLike): Model save path.
    """
    directory = os.path.dirname(path)

    if not os.path.exists(directory):
        print(f"Model save directory does not exist, creating: {directory}")
        os.makedirs(directory)


def custom_loss(
        image_features: torch.Tensor, 
        text_features: torch.Tensor
    ) -> torch.Tensor:
    # Assuming image_features and text_features are normalized
    similarity = torch.nn.functional.cosine_similarity(text_features, image_features)
    loss = torch.mean(1 - similarity)  # Penalize high similarity
    return loss


def normalize_features(features: torch.Tensor) -> torch.Tensor:
    return features / features.norm(dim=1, keepdim=True)


def cross_entropy(logits: torch.Tensor, axis: int = 1) -> torch.Tensor:
    """
    Calculate the cross entropy loss.

    args:
        logits (torch.Tensor): Logits.
        axis (int): Axis.

    returns:
        ce (torch.Tensor): Cross entropy loss.
    """
    # Calculate log probabilities
    logprobs = torch.log_softmax(logits, axis=axis)

    # Calculate negative log likelihood
    nll = torch.diag(logprobs)

    # Calculate cross entropy
    ce = -torch.mean(nll)
    return ce


def custom_loss_same_class(
        anchor_image_features: torch.Tensor, 
        positive_text_features: torch.Tensor
    ) -> torch.Tensor:
    """
    Calculate the loss for the same class.
    
    args:
        anchor_image_features (torch.Tensor): Anchor image features.
        positive_text_features (torch.Tensor): Positive text features.
    
    returns:
        loss (torch.Tensor): Loss.
    """
    # Ensure features are normalized
    image_features = F.normalize(anchor_image_features, dim=1)
    text_features = F.normalize(positive_text_features, dim=1)

    # Calculate similarity
    similarity = torch.matmul(image_features, text_features.T)

    # Compute CLIP loss
    loss = -((cross_entropy(similarity, axis=0) + cross_entropy(similarity, axis=1)) / 2)
    return loss


def custom_loss_different_class(
        anchor_image_features: torch.Tensor, 
        negative_text_features: torch.Tensor
    ) -> torch.Tensor:
    """
    Calculate the loss for different classes.

    args:
        anchor_image_features (torch.Tensor): Anchor image features.
        negative_text_features (torch.Tensor): Negative text features.
    
    returns:
        loss (torch.Tensor): Loss.
    """
    # Ensure features are normalized
    image_features = F.normalize(anchor_image_features, dim=1)
    text_features = F.normalize(negative_text_features, dim=1)

    # Calculate similarity
    similarity = torch.matmul(image_features, text_features.T)

    # Compute CLIP loss
    loss = 1 - ((cross_entropy(similarity, axis=0) + cross_entropy(similarity, axis=1)) / 2)
    return loss


def custom_triplet_loss(
        anchor_image_features: torch.Tensor, 
        positive_text_features: torch.Tensor, 
        negative_text_features: torch.Tensor, 
        margin=1.0
    ) -> torch.Tensor:
    """
    Calculate the triplet loss.
    
    args:
        anchor_image_features (torch.Tensor): Anchor image features.
        positive_text_features (torch.Tensor): Positive text features.
        negative_text_features (torch.Tensor): Negative text features.
        margin (float): Margin value.

    returns:
        loss (torch.Tensor): Triplet loss.
    """
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


def calc_loss(
        anchor_image_features: torch.Tensor, 
        positive_text_features: torch.Tensor, 
        negative_text_features: torch.Tensor, 
        is_same_class: bool, 
        loss_method: Loss_method
    ) -> torch.Tensor:
    """
    Calculate the loss.

    args:
        anchor_image_features (torch.Tensor): Anchor image features.
        positive_text_features (torch.Tensor): Positive text features.
        negative_text_features (torch.Tensor): Negative text features.
        is_same_class (bool): If True, the anchor and positive features are from the same class.
        loss_method (Loss_method): Loss method.

    returns:
        loss (torch.Tensor): Loss.
    """
    if loss_method == Loss_method.DIFFRENT_SAME:
        if is_same_class:
            return custom_loss_same_class(anchor_image_features, positive_text_features)
        else:
            return custom_loss_different_class(anchor_image_features, negative_text_features)

    elif loss_method == Loss_method.CUSTOM_TRIPLET_LOSS:
        return custom_triplet_loss(anchor_image_features, positive_text_features, negative_text_features)


def create_learning_rate_fn(
    optimizer: torch.optim.Optimizer,
    train_ds_size: int,
    train_batch_size: int,
    num_train_epochs: int,
    num_warmup_steps: int,
    learning_rate: float,
    linear=False
) -> torch.optim.lr_scheduler.LambdaLR:
    """
    Returns a PyTorch learning rate scheduler.

    args:
        optimizer (torch.optim.Optimizer): Optimizer.
        train_ds_size (int): Training dataset size.
        train_batch_size (int): Batch size.
        num_train_epochs (int): Number of training epochs.
        num_warmup_steps (int): Number of warmup steps.
        learning_rate (float): Learning rate.
        linear (bool): If True, use linear learning rate decay.

    returns:
        lr_scheduler (torch.optim.lr_scheduler.LambdaLR): Learning rate scheduler.
    """
    # Calculate the number of steps per epoch
    steps_per_epoch = train_ds_size // train_batch_size
    num_train_steps = steps_per_epoch * num_train_epochs

    # Learning rate function
    def lr_lambda(current_step: int) -> float:
        """
        Learning rate function.

        args:
            current_step (int): Current step.

        returns:
            lr (float): Learning rate.
        """
        # Warmup
        if current_step < num_warmup_steps:
            # Calculate the learning rate
            return float(current_step) / float(max(1, num_warmup_steps))
        
        # Linear decay
        if linear:
            # Calculate the learning rate
            return max(
                0.0, float(num_train_steps - current_step) / float(max(1, num_train_steps - num_warmup_steps))
            )
        else:
            # Cosine decay
            return 0.5 * (1 + np.cos(np.pi * (current_step - num_warmup_steps) / (num_train_steps - num_warmup_steps)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def adaptive_grad_clip(
        parameters: List[Any], 
        clip_factor: float, 
        eps: float = 1e-3
    ):
    """
    Adaptively clip gradients to prevent exploding gradients.

    args:
        parameters (List[Any]): List of model parameters.
        clip_factor (float): Clipping factor.
        eps (float): Epsilon value.
    """
    for p in parameters:

        if p.grad is not None:
            
            # Compute the gradient norm
            grad_norm = p.grad.norm()
            max_norm = clip_factor / (eps + grad_norm)
            p.grad.data.clamp_(-max_norm, max_norm)


def train_step_batch(
        images: List[Image.Image], 
        text_feat: torch.Tensor, 
        class_labels: torch.Tensor,  # Add this to pass class labels
        batch_size: int, 
        clip_model: CLIP, 
        optimizer=None, 
        lr_scheduler=None,
        margin: float = 1.0
    ) -> float:
    """
    Train the model for one batch.

    args:
        images (List[Image.Image]): List of images.
        text_feat (torch.Tensor): Text features.
        class_labels (torch.Tensor): Class labels.
        batch_size (int): Batch size.
        clip_model (nn.Module): CLIP model.
        optimizer (torch.optim.Optimizer): Optimizer.
        lr_scheduler (torch.optim.lr_scheduler): Learning rate scheduler.
        margin (float): The margin for the triplet loss.

    returns:
        loss (float): Training loss.
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

    optimizer.zero_grad()
    total_loss.backward()

    if optimizer:
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
        optimizer=None, 
        lr_scheduler=None,
        margin: float = 1.0
    ) -> float:
    """
    Train the model for one epoch.

    args:
        epoch (int): Current epoch number.
        train_loader (DataLoader): Training DataLoader.
        clip_model (nn.Module): CLIP model.
        image_loader (ImageLoader): Image loader.
        optimizer (torch.optim.Optimizer): Optimizer.
        lr_scheduler (torch.optim.lr_scheduler): Learning rate scheduler.

    returns:
        loss (float): Training loss.
    """
    clip_model.train()
    running_loss = 0

    train_bar = tqdm(train_loader, desc=f'Fine-tuning epoch {epoch + 1}', position=1, leave=True)
    for batch in train_bar:
        optimizer.zero_grad()

        *_, image_names, _, words = batch

        images = [image_loader(image) for image in tqdm(image_names, position=2, desc='Processing Images', leave=False)]

        emb_descriptions = [gen_word_objs_embeddings(description, clip_model) for description in tqdm(words, position=2, desc='Generating Embeddings', leave=False)]
        emb_descriptions = torch.stack(emb_descriptions)

        unique_classes = set(words)  # Find the unique classes
        class_to_index = {cls: idx for idx, cls in enumerate(unique_classes)}  # Create a mapping

        class_indices = [class_to_index[cls] for cls in words]
        class_labels_tensor = torch.tensor(class_indices)

        temp_loss = train_step_batch(images, emb_descriptions, class_labels_tensor, batch_size, clip_model, optimizer, lr_scheduler, margin)

        running_loss += temp_loss

    loss = running_loss / len(train_loader)

    if optimizer:
        optimizer.step()

    if lr_scheduler:
        lr_scheduler.step()

    return loss


def validate_one_epoch(
        epoch: int, 
        val_loader: DataLoader, 
        clip_model: CLIP, 
        clip_preprocess: nn.Transformer, 
        image_loader: ImageLoader, 
        optimizer=None, 
        scheduler=None
    ) -> Tuple[float, float]:
    """
    Validate the model on the validation set.

    args:
        epoch (int): Current epoch number.
        val_loader (DataLoader): Validation DataLoader.
        clip_model (nn.Module): CLIP model.
        clip_preprocess (nn.Transformer): CLIP preprocess model.
        image_loader (ImageLoader): Image loader.
        optimizer (torch.optim.Optimizer): Optimizer.
        scheduler (torch.optim.lr_scheduler): Learning rate scheduler.

    returns:
        val_loss (float): Validation loss.
        val_accuracy (float): Validation accuracy.
    """

    global device

    clip_model.eval()
    val_loss = 0.0
    val_accuracy = 0.0
    total_samples = 0

    # Validation loop
    with torch.no_grad():
        val_bar = tqdm(val_loader, desc=f"Validation Epoch {epoch + 1}", position=1, leave=True)

        # Iterate over the validation set
        for batch in val_bar:
            *_, image_names, _, descriptions = batch

            # Preprocess images
            images = [clip_preprocess(image_loader(img_name)).unsqueeze(0).to(device) for img_name in image_names]
            images = torch.cat(images, dim=0)

            # Encode images
            images_enc = clip_model.encode_image(images)

            # Process and encode descriptions
            descriptions_enc = torch.stack([gen_word_objs_embeddings(description, clip_model) for description in descriptions]).squeeze(1)

            # Compute similarity scores between images and texts
            image_logits = images_enc @ descriptions_enc.t()
            ground_truth = torch.arange(len(image_logits)).long().to(device)

            # Calculate loss
            loss = (F.cross_entropy(image_logits, ground_truth) + F.cross_entropy(image_logits.t(), ground_truth)).div(2)
            val_loss += loss.item() * images.size(0)
            total_samples += images.size(0)

            # Calculate accuracy
            acc_i = (torch.argmax(image_logits, 1) == ground_truth).sum()
            acc_t = (torch.argmax(image_logits, 0) == ground_truth).sum()
            val_accuracy += (acc_i + acc_t).float().item() / 2

            val_bar.set_description(f"Validation Epoch {epoch + 1} Loss: {loss.item():.4f}")

        # Compute average loss and accuracy
        val_loss /= total_samples
        val_accuracy /= total_samples

    # Logging validation loss and accuracy
    print(f"Epoch {epoch + 1}, Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}")

    if optimizer:
        optimizer.step()

    if scheduler:
        scheduler.step(val_loss)

    return val_loss, val_accuracy


def main(args=None) -> Tuple[float, CLIP, nn.Transformer]:
    parser = argparse.ArgumentParser()

    # Add arguments
    parser = clip_fine_tune_argparse(parser)
    parser = dataset_argparse(parser)
    parser = early_stopper_argparse(parser)
    parser = phosc_net_argparse(parser)

    # Parse arguments
    if args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args)

    # Define patchs
    root_dir = ospj(DATA_FOLDER, "BengaliWords", "BengaliWords_CroppedVersion_Folds")
    image_loader_path = ospj(root_dir, args.split_name)

    root_model_path = ospj('models', 'fine-tuned_clip', args.split_name)
    log_file_path = ospj(root_model_path, 'training_log.log')
    model_save_path = ospj(root_model_path, 'simple', 'model.pth')

    verify_model_save_path(model_save_path)

    setup_logging(log_file_path)

    # Set up models
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
    clip_model.float()

    # Load phosc model
    phosc_model = get_phoscnet(args, device)

    # Load datasets
    train_loader, train_set = get_training_loader(args)
    validation_loader, _ = get_validation_loader(args)

    # Define image loader
    loader = ImageLoader(image_loader_path)

    # Fine-tuning
    """
    optimizer = torch.optim.RMSprop(
        clip_model.parameters(), 
        lr=args.lr,
        eps=args.eps,
        weight_decay=args.weight_decay
    ) 
    """

    optimizer = torch.optim.AdamW(
        clip_model.parameters(),
        lr=args.lr,
        eps=args.eps,
        weight_decay=args.weight_decay
    )

    # Learning rate scheduler   
    decay_lr_schedule_fn = create_learning_rate_fn(
        optimizer,
        len(train_set),
        args.batch_size,
        args.epochs,
        args.warmup_steps,
        args.lr,
        linear=False,  # set False to activate cosine annealing
    )

    # Early stopping
    early_stopping = EarlyStopping(
        save_path=ospj('models', 'fine-tuned_clip', args.split_name, args.name),
        patience=args.stop_patience,
        verbose=args.verbose,
        save_every=args.save_every,
        model_arguments=args,
        save=args.save
    )

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(
            epoch,
            train_loader,
            clip_model,
            loader,
            args.batch_size,
            optimizer,
            decay_lr_schedule_fn,
            args.margin
        )

        """
        val_loss, val_acc = validate_one_epoch(
            epoch,
            validation_loader,
            clip_model,
            clip_preprocess,
            loader,
            decay_lr_schedule_fn
        )
        """

        if early_stopping(train_loss, clip_model, epoch):
            return early_stopping.best_score, clip_model, early_stopping.best_model_path
    
    else:
        """
        print('Training completed')
        print(f'Best validation loss: {early_stopping.best_score}')
        print(f'Best model saved at: {early_stopping.best_model_path}')
        """
        
        return early_stopping.best_score, clip_model, early_stopping.best_model_path


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Ctrl-C exit')
