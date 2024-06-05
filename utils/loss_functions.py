import torch
import torch.nn.functional as F
from torch import Tensor
from utils.dbe import dbe


def compute_triplet_margin_loss(logits_per_image: Tensor, class_labels: Tensor, margin: float) -> Tensor:
    """
    Computes the triplet margin loss given logits, class labels, and a margin.

    Args:
        logits_per_image: Tensor of shape (batch_size, batch_size) representing
                          logits or similarity scores between all pairs.
        class_labels: Tensor of shape (batch_size,) representing class labels.
        margin: A float representing the margin value used in the loss calculation.

    Returns:
        A scalar tensor representing the mean triplet margin loss.
    """
    same_class_mask = class_labels.unsqueeze(1) == class_labels.unsqueeze(0)

    # Positive pairs
    positive_scores = []
    for i in range(class_labels.size(0)):
        mask = (class_labels == class_labels[i]) & (torch.arange(class_labels.size(0)) != i)

        if mask.any():
            positive_scores.append(logits_per_image[i][mask].max())
        else:
            positive_scores.append(torch.tensor(0.0, device=logits_per_image.device))

    positive_pairs = torch.stack(positive_scores) if positive_scores else torch.tensor([0.0], device=logits_per_image.device)

    # Negative pairs
    negative_mask = ~same_class_mask

    negative_mask = negative_mask.to(logits_per_image.device)

    max_negative_logits = torch.where(negative_mask, logits_per_image, torch.tensor(float('-inf')).to(logits_per_image.device))
    negative_pairs = max_negative_logits.max(dim=1)[0]

    # Compute loss
    loss = F.relu((positive_pairs - negative_pairs) + margin).mean()
    return loss


def compute_contrastive_loss(logits_per_image: Tensor, class_labels: Tensor, margin: float) -> Tensor:
    """
    Computes the contrastive loss given logits, class labels, and a margin.

    Args:
        logits_per_image: Tensor of shape (batch_size, batch_size) representing
                          logits or similarity scores between all pairs.
        class_labels: Tensor of shape (batch_size,) representing class labels.
        margin: A float representing the margin value used in the loss calculation.

    Returns:
        A scalar tensor representing the mean contrastive loss.
    """
    same_class_mask = class_labels.unsqueeze(1) == class_labels.unsqueeze(0)
    
    class_labels = class_labels.to(logits_per_image.device)
    same_class_mask = same_class_mask.to(logits_per_image.device)

    # Positive scores
    positive_scores = logits_per_image.masked_select(same_class_mask.fill_diagonal_(False))
    negative_scores = logits_per_image.masked_select(~same_class_mask)
    
    # Check if there are positive or negative scores and compute losses accordingly
    if positive_scores.numel() == 0:
        positive_loss = torch.tensor(0.0, device=logits_per_image.device, dtype=logits_per_image.dtype)
    else:
        positive_loss = F.relu(1.0 - positive_scores).mean()

    if negative_scores.numel() == 0:
        negative_loss = torch.tensor(0.0, device=logits_per_image.device, dtype=logits_per_image.dtype)
    else:
        negative_loss = F.relu(negative_scores - margin).mean()

    # Total loss
    loss = positive_loss + negative_loss

    return loss


def simple_loss(logits_per_image: Tensor) -> Tensor:
    """
    Computes the simple loss given logits. The simple loss is the mean of the top-5 logits for each image.

    Args:
        logits_per_image: Tensor of shape (batch_size, batch_size) representing
                          logits or similarity scores between all pairs.

    Returns:
        A scalar tensor representing the mean loss.
    """
    loss = logits_per_image.topk(5, dim=1).values.mean(dim=1).mean()

    return loss
