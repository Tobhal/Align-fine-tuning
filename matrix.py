import torch
import torch.nn.functional as F
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

import logging

from PIL import Image

import clip

import os
from os import PathLike
from os.path import join as ospj

from typing import List, Tuple

from timm import create_model

from tqdm import tqdm

from flags import DATA_FOLDER, device

from utils.dbe import dbe
from utils.utils import clip_text_features_from_description

from data import dataset_bengali as dset
from data.dataset_bengali import ImageLoader

from modules.utils import set_phos_version, set_phoc_version, gen_shape_description
from modules.utils.utils import split_string_into_chunks, get_phosc_description

from modules import models, residualmodels

import numpy as np
import pandas as pd

from torchvision import transforms

from transformers import AlignProcessor, AlignModel, AutoTokenizer, AutoProcessor, AlignTextModel, AlignConfig
from enum import Enum

from utils.get_dataset import get_test_loader, get_phoscnet

import argparse
from parser import phosc_net_argparse, dataset_argparse, aling_fine_tune_argparse, matrix_argparse

split = 'fold_0_t'
use_augmented = False

# align model
align_processor = AlignProcessor.from_pretrained("kakaobrain/align-base")
align_model = AlignModel.from_pretrained("kakaobrain/align-base")
align_auto_tokenizer = AutoTokenizer.from_pretrained("kakaobrain/align-base")
align_auto_processor = AutoProcessor.from_pretrained("kakaobrain/align-base")
# align_text_model = AlignTextModel.from_pretrained("kakaobrain/align-base")


# Create a new configuration with a larger maximum sequence length
# config = AlignConfig.from_pretrained("kakaobrain/align-base", max_position_embeddings=2048)

# Create a new model with the updated configuration
# align_text_model = AlignTextModel(config)

# save_path = ospj('models', 'fine-tuned_clip', split)
save_path = ospj('models', 'align-fine-tune', split)
matrix_save_path = ospj(save_path, 'matrix')


# Preprocessing for CLIP
clip_preprocess = Compose([
    Resize(224, interpolation=Image.BICUBIC),
    CenterCrop(224),
    ToTensor(),
    Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
])



def save_matrix(matrix: torch.Tensor, results: tuple, _model_save_path: PathLike, csv_filename="matrix"):
    """
    Save the given matrix as a CSV file.

    args:
        matrix: The matrix to save
        results (tuple): The results of the evaluation
        model_save_path (PathLike): The path to save the matrix
        csv_filename (str): The name of the CSV file to save
    """
    # Extract the directory from the model save path
    directory = os.path.dirname(_model_save_path)

    # Ensure the directory exists
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Create the full path for the CSV file
    csv_path = ospj(directory, f'{csv_filename}.csv')
    txt_path = ospj(directory, f'{csv_filename}.txt')

    if torch.is_tensor(matrix):
        matrix = matrix.cpu().numpy()

    # Convert the matrix to a DataFrame and save as CSV
    df = pd.DataFrame(matrix)
    df.to_csv(csv_path, index=False, header=False)

    # Save the results to a text file
    with open(txt_path, 'w') as f:
        f.write(f"Model number: {results[0]}\n")
        f.write(f"Minimum value in matrix: {results[1]}\n")
        f.write(f"Maximum value in matrix: {results[2]}\n")
        f.write(f"Mean value in matrix: {results[3]}\n")

    print(f"Matrix saved at: {csv_path}")


def calculate_cos_angle_matrix(vector: torch.Tensor) -> torch.Tensor:
    """
    Calculate the cosine angle matrix for the given vectors.

    args:
        vectors (torch.Tensor): The vectors to calculate the cosine angle matrix for

    returns:
        cos_angle_matrix (torch.Tensor): The cosine angle matrix for the given vectors
    """
    n = len(vector)
    cos_angle_matrix = torch.zeros((n, n))

    for i in range(n):
        for j in range(n):
            # Convert vectors to PyTorch tensors if they aren't already
            vec_i = vector[i]
            vec_j = vector[j]

            # Calculate the dot product of the two vectors
            try:
                dot_product = torch.matmul(vec_i, vec_j)
            except RuntimeError as e:
                dbe(vec_i.shape, vec_j.shape, e)

            # Calculate the magnitudes of the vectors
            magnitude_i = torch.norm(vec_i)
            magnitude_j = torch.norm(vec_j)

            # Calculate the cosine of the angle
            cos_theta = dot_product / (magnitude_i * magnitude_j)

            # Ensure the cosine value is within the valid range [-1, 1]
            # cos_theta = torch.clamp(cos_theta, -1, 1)

            # Assign the cosine value to the matrix
            cos_angle_matrix[i, j] = cos_theta

    return cos_angle_matrix


class ModelType(Enum):
    """
    Enum class for the type of model to evaluate.
    """
    CLIP = "CLIP"
    ALIGN = "ALIGN"


def compute_loss_and_accuracy(
        images_enc: torch.Tensor, 
        descriptions_enc: torch.Tensor, 
        image_names: List[str], 
        device: str
    ) -> Tuple[float, float]:
    """
    Compute the loss and accuracy for the given images and descriptions.

    args:
        images_enc (torch.Tensor): The encoded images
        descriptions_enc (torch.Tensor): The encoded descriptions
        image_names (List[str]): The names of the images
        device (str): The device to use for evaluation
    
    returns:
        loss (float): The loss for the batch
        accuracy (float): The accuracy for the batch
    """
    # Compute loss and accuracy for validation metrics
    image_logits = images_enc @ descriptions_enc.t()
    ground_truth = torch.arange(len(image_logits)).long().to(device)
    loss = (F.cross_entropy(image_logits, ground_truth) + F.cross_entropy(image_logits.t(), ground_truth)).div(2)

    # Calculate accuracy
    acc_i = (torch.argmax(image_logits, 1) == ground_truth).sum()
    acc_t = (torch.argmax(image_logits, 0) == ground_truth).sum()
    accuracy = (acc_i + acc_t).float() / 2 / len(image_names)

    return loss.item(), accuracy.item()


def clip_process_and_evaluate_batch(
        image_names: List[str], 
        descriptions: List[str], 
        model: nn.Module, 
        preprocess: nn.Transformer, 
        loader: ImageLoader, 
        device: str
    ) -> Tuple[List[float], List[float]]:
    """
    Process the images and descriptions in the batch and evaluate the model.

    args:
        image_names (List[str]): The names of the images in the batch
        descriptions (List[str]): The descriptions for the images in the batch
        model (nn.Module): The model to evaluate
        preprocess (nn.Transformer): The preprocessing function to apply to the images
        loader (ImageLoader): The image loader to use for loading the images
        device (str): The device to use for evaluation

    returns:
        losses (List[float]): The losses for the batch
        accuracy (List[float]): The accuracy for the batch
    """
    # Process images
    images = [preprocess(loader(img_name)).unsqueeze(0).to(device) for img_name in image_names]
    images = torch.cat(images, dim=0)

    # Precompute embeddings for all descripdtions in the batch
    descriptions_enc = torch.stack([clip_text_features_from_description(description, model) for description in descriptions]).squeeze(1)

    # Encode images using the model
    images_enc = model.encode_image(images)

    # Calculate cosine similarity between each image and text features in the batch
    similarity_matrix = torch.nn.functional.cosine_similarity(images_enc.unsqueeze(1), descriptions_enc.unsqueeze(0), dim=2)
    similarities = similarity_matrix.diag().cpu().tolist() 

    return compute_loss_and_accuracy(images_enc, descriptions_enc, image_names, device)


def align_process_and_evaluate_batch(
        image_names: List[str], 
        descriptions: List[str], 
        model: nn.Module, 
        transform: nn.Transformer, 
        loader: ImageLoader,
        device: str
    ) -> Tuple[List[float], List[float]]:
    """
    Process the images and descriptions in the batch and evaluate the model.

    args:
        image_names (List[str]): The names of the images in the batch
        descriptions (List[str]): The descriptions for the images in the batch
        model (nn.Module): The model to evaluate
        transform (nn.Transform): The transformation to apply to the images
        device (str): The device to use for evaluation

    returns:
        losses (List[float]): The losses for the batch
        accuracy (List[float]): The accuracy for the batch
    """
    images = [loader(img_name) for img_name in image_names]
    losses = []
    accuracy = []

    # Process images and descriptions
    for image, description in zip(images, descriptions):
        processor_inputs = align_auto_processor(image=image, return_tensors="pt")
        text_input = align_auto_tokenizer(description, return_tensors="pt")

        # Get image and text features
        image_features = model.get_image_features(**processor_inputs)
        text_features = model.get_text_features(**text_input)

        # Compute similarity scores between images and texts
        loss, acc = compute_loss_and_accuracy(image_features, text_features, image_names, device)

        losses.append(loss)
        accuracy.append(acc)

    return losses, accuracy


def evaluate_model_batch(
        model: nn.Module, 
        dataloader: DataLoader, 
        loader: ImageLoader,
        device: str, 
        model_type: ModelType,
    ) -> Tuple[float, float, List[float]]:
    """
    Evaluate the model on the given dataloader and return the average loss and accuracy.

    args:
        model (nn.Module): The model to evaluate
        dataloader (DataLoader): The dataloader to use for evaluation
        device (str): The device to use for evaluation
        model_type (ModelType): The type of model to evaluate

    returns:
        avg_loss (float): The average loss for the model
        avg_accuracy (float): The average accuracy for the model
        similarities (List[Float]): The cosine similarities between images and text features
    """
    model.eval()
    similarities = []
    losses = []
    accuracies = []

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Evaluate the model on the given dataloader
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            *_, image_names, _, descriptions = batch

            # Process the batch based on the model type
            if model_type == ModelType.CLIP:
                clip_process_and_evaluate_batch(image_names, descriptions, model, clip_preprocess, loader, device)
            elif model_type == ModelType.ALIGN:
                align_process_and_evaluate_batch(image_names, descriptions, model, transform, loader, device)

    # Calculate the average loss and accuracy
    avg_loss = sum(losses) / len(losses)
    avg_accuracy = sum(accuracies) / len(accuracies)

    return avg_loss, avg_accuracy, similarities


def evaluate_text_embedings(
        model: nn.Module, 
        dataloader: DataLoader, 
        loader: ImageLoader,
        # preprocess: nn.Transformer=clip_preprocess, 
        model_type: ModelType = ModelType.CLIP,
    ) -> torch.Tensor:
    """
    Evaluate the model on the given dataloader and return the text embeddings for each batch.

    args:
        model (nn.Module): The model to evaluate
        dataloader (DataLoader): The dataloader to use for evaluation
        device (str): The device to use for evaluation
        preprocess (nn.Transformer): The preprocessing function to use for images
        model_type (ModelType): The type of model to evaluate

    retunrs:
        batch_features_all (torch.Tensor): The text embeddings for each batch
    """
    global align_model, align_auto_tokenizer
    model.eval()
    similarities = []
    batch_features_all = []

    # Evaluate the model on the given dataloader
    with torch.no_grad():
        for batch in tqdm(dataloader, position=0, desc="Batch Progress"):
            # Unpacking the batch data
            *_, images, _, words = batch

            # Process the batch based on the model type
            if model_type == ModelType.CLIP:
                batch_features_all.append(clip_text_features_from_description(words, model))
            elif model_type == ModelType.ALIGN:

                # Process images and descriptions
                for image, word in zip(images, words):
                    image = loader(image)
                    description = get_phosc_description(word)

                    inputs = align_auto_tokenizer(description, padding=True, return_tensors="pt")

                    text_features = model.get_text_features(**inputs)

                    batch_features_all.append(text_features)

    batch_features_all = torch.cat(batch_features_all, dim=0)

    return batch_features_all


def print_results(results):
    for result in results:
        num, min_value, max_value, average_value = result
        print(f"Model number {num}:")
        print(f"Minimum value in matrix: {min_value}")
        print(f"Maximum value in matrix: {max_value}")
        print(f"Mean value in matrix: {average_value}")


def main(args=None, model=None, index=0) -> Tuple[float, float]:
    parser = argparse.ArgumentParser()

    parser = matrix_argparse(parser)
    parser = phosc_net_argparse(parser)
    parser = dataset_argparse(parser)
    parser = aling_fine_tune_argparse(parser)

    # Parse arguments
    if args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args)

    root_dir = ospj(DATA_FOLDER, "BengaliWords_CroppedVersion_Folds")

    phosc_model = get_phoscnet(args, device)
    test_loader, _ = get_test_loader(args, phosc_model)
    image_loader = ImageLoader(ospj(root_dir, args.split_name))

    results = []

    for num in args.nums:
        model_save_path = ospj(args.save_dir, args.name, args.split_name, str(num))

        align_fine_tuned_model = AlignModel.from_pretrained("kakaobrain/align-base")
        align_fine_tuned_model_path = ospj(model_save_path, args.checkpoint_name)
        align_fine_tuned_model.load_state_dict(torch.load(align_fine_tuned_model_path))

        if args.evaluate == 'model':
            pass
        elif args.evaluate == 'text':
            batch_features_all = evaluate_text_embedings(align_fine_tuned_model, test_loader, image_loader, ModelType.ALIGN)

        matrix = calculate_cos_angle_matrix(batch_features_all)
        min_value = torch.min(matrix).item()
        max_value = torch.max(matrix).item()
        average_value = torch.mean(matrix).item()

        res = (num, min_value, max_value, average_value)

        save_matrix(matrix, res, ospj(model_save_path, num), f'matrix_{num}')

        results.append(res)

    return results


if __name__ == '__main__':
    results = main(model=align_model)

    print_results(results)

    # similarities, batch_features_all = evaluate_model(fine_tuned_clip_model, test_loader, device, fine_tuned_clip_preprocess)
    # similarities, batch_features_all = evaluate_model(original_clip_model, test_loader, device, original_clip_preprocess)
    # batch_features_all = evaluate_text_embedings(original_clip_model, test_loader, ModelType.CLIP)
    # batch_features_all = evaluate_text_embedings(model, test_loader, ModelType.CLIP)
    # batch_features_all = evaluate_text_embedings(align_model, test_loader, align_processor, ModelType.ALIGN)

    # batch_features_all = evaluate_text_embedings(model, test_loader, ModelType.CLIP)