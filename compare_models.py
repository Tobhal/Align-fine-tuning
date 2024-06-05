import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

import logging

from PIL import Image

import torch.nn.functional as F
from torch.nn.functional import cosine_similarity

import clip
import os
from os.path import join as ospj

from timm import create_model

from tqdm import tqdm

from flags import DATA_FOLDER, device

from utils.dbe import dbe
from data import dataset_bengali as dset
from data.dataset_bengali import ImageLoader

from modules.utils import set_phos_version, set_phoc_version, gen_shape_description

from modules import models, residualmodels
from modules.utils.utils import get_phosc_description

import numpy as np

from utils.dbe import dbe
from utils.utils import clip_text_features_from_description

from torchvision import transforms
from typing import List, Tuple

from modules.utils.utils import split_string_into_chunks

# align
from transformers import AlignProcessor, AlignModel, AutoTokenizer
from enum import Enum

import argparse
from utils.get_dataset import get_test_loader, get_phoscnet
from modules.utils.utils import get_phosc_description
from parser import phosc_net_argparse, dataset_argparse

# Preprocessing for CLIP
clip_preprocess = Compose([
    Resize(224, interpolation=Image.BICUBIC),
    CenterCrop(224),
    ToTensor(),
    Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
])

align_preprocess = Compose([
    ToTensor(),
])

# align model
align_processor = AlignProcessor.from_pretrained("kakaobrain/align-base")
align_model = AlignModel.from_pretrained("kakaobrain/align-base")
align_tokenizer = AutoTokenizer.from_pretrained("kakaobrain/align-base")

align_fine_tuned_model = AlignModel.from_pretrained("kakaobrain/align-base")

"""
# Load original and fine-tuned CLIP models
original_clip_model, original_clip_preprocess = clip.load("ViT-B/32", device=device)
original_clip_model.float()

# Load fine-tuned clip model
fine_tuned_clip_model, fine_tuned_clip_preprocess = clip.load("ViT-B/32", device=device)
fine_tuned_clip_model.float()

fine_tuned_state_dict = torch.load(finetuned_model_save_path, map_location=device)
fine_tuned_clip_model.load_state_dict(fine_tuned_state_dict)
"""


class ModelType(Enum):
    CLIP = "CLIP"
    ALIGN = "ALIGN"


def calculate_cos_angle_matrix(vector_1, vector_2):
    """
    Calculate the cosine of the angle between two vectors.

    Args:
        vector_1: The first vector.
        vector_2: The second vector.

    Returns:
        cos_angle_matrix: The cosine of the angle between the two vectors.
    """
    # Ensure the vectors are PyTorch tensors and flatten them if they are 2D
    vector_1 = torch.tensor(vector_1).flatten()
    vector_2 = torch.tensor(vector_2).flatten()

    vectors = [vector_1, vector_2]
    n = len(vectors)
    cos_angle_matrix = torch.zeros((n, n))

    for i in range(n):
        for j in range(n):
            # Calculate the dot product of the two vectors
            dot_product = torch.matmul(vectors[i], vectors[j])

            # Calculate the magnitudes of the vectors
            magnitude_a = torch.norm(vectors[i])
            magnitude_b = torch.norm(vectors[j])

            # Calculate the cosine of the angle
            cos_theta = dot_product / (magnitude_a * magnitude_b)

            # Ensure the cosine value is within the valid range [-1, 1]
            cos_theta = torch.clamp(cos_theta, -1, 1)

            # Assign the cosine value to the matrix
            cos_angle_matrix[i, j] = cos_theta

    return cos_angle_matrix


def clip_preprocess_and_encode(image_names, words, clip_model, transform, loader, device):
    """
    Preprocess and encode images and descriptions using the CLIP model.

    Args:
        image_names: The list of image names.
        words: The list of words.
        clip_model: The CLIP model.
        transform: The image transformation.
        loader: The image loader.
        device: The device to use.

    Returns:
        values: The similarity values.
        indices: The indices of the descriptions.
    """

    # Preprocess and encode images
    images = [transform(loader(img_name)).unsqueeze(0).to(device) for img_name in image_names]
    # image = transform(loader(image_names)).unsqueeze(0).to(device)
    images = torch.cat(images, dim=0)
    images_features = clip_model.encode_image(images)

    # Precompute embeddings for all descriptions in the batch
    text_features = words
    # text_features = clip_text_features_from_description(words, clip_model)

    images_features /= images_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    # Compute similarity scores between images and texts
    similarity = (images_features @ text_features.t()).softmax(dim=1)

    values, indices = similarity[0].topk(5)

    # Compute cosine similarity between image and text embeddings
    # similarity_matrix = torch.nn.functional.cosine_similarity(image_logits, ground_truth, dim=0)

    return values, indices


def align_preprocess_and_encode(image_names, words, align_model, transform, loader, device):
    """
    Preprocess and encode images and descriptions using the ALIGN model.

    Args:
        image_names: The list of image names.
        words: The list of words.
        align_model: The ALIGN model.
        transform: The image transformation.
        loader: The image loader.
        device: The device to use.

    Returns:
        probs: The probabilities of the descriptions.
    """
    images = [loader(img_name) for img_name in image_names]
    descriptions = [get_phosc_description(word) for word in words]
    probs = []

    for image, description in zip(images, descriptions):
        text_input = align_tokenizer(descriptions, padding=True, return_tensors="pt")
        text_features = align_model.get_text_features(**text_input)

        image_inputs = align_processor(images=images, return_tensors="pt")
        image_features = align_model.get_image_features(**image_inputs)

        probs.append(text_features)

        """
        outputs = align_model(**inputs)

        logits = outputs.logits_per_image

        prob = logits.softmax(dim=1)
        # dbe(prob[0][0])
        probs.append(prob[0][0])
        """
    
    return probs


def evaluate_model(model, dataloader, device, loader, model_type: ModelType):
    """
    Evaluate the model on the test set.

    Args:
        model: The model to evaluate.
        dataloader: The DataLoader for the test set.
        device: The device to use for evaluation.
        loader: The image loader.
        model_type: The type of model to evaluate.

    Returns:
        avg_same_class_similarity: The average similarity score for same-class pairs.
    """
    model.eval()
    batch_similarities_values = []
    batch_similarities_indicies = []

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            *_, image_names, _, words = batch

            if model_type == ModelType.CLIP:
                w = torch.stack([clip_text_features_from_description(word, model) for word in tqdm(words, position=1, desc='Generating Embeddings', leave=False)]).squeeze(1)

                values, indicies = clip_preprocess_and_encode(image_names, w, model, transform, loader, device)
                batch_similarities_values.append(values)
                batch_similarities_indicies.append(indicies)

            elif model_type == ModelType.ALIGN:
                probs = align_preprocess_and_encode(image_names, words, model, transform, loader, device)
                batch_similarities_values.append(probs)

    # Flatten the list of similarities
    flat_similarities = batch_similarities_values

    # Compute average similarities for same and different classes
    avg_same_class_similarity = np.mean(flat_similarities) if flat_similarities else 0
    # avg_different_class_similarity = np.mean(batch_similarities_different_class) if batch_similarities_different_class else 0

    return avg_same_class_similarity


def summarize_results(*args: Tuple[str, np.floating]):
    """
    Summarize the results of the model comparisons.

    Args:
        args: A tuple containing the model name and the average similarity score.
    """
    # Extract the model names and average similarity scores
    model_names = [args[0] for arg in args]
    avg_same_class_similarities = [args[1] for arg in args]

    # Determine which model performs better for same-class pairs
    best_same_class_model = model_names[np.argmax(avg_same_class_similarities)]

    # Print the results
    print(f"Average Similarity Scores:")
    for model_name, avg_same_class_similarity in zip(model_names, avg_same_class_similarities):
        print(f"{model_name}: {avg_same_class_similarity:.5f}")

    print(f"\nBest Model for Same-Class Pairs: {best_same_class_model}")
    print(f"Worst Model for Same-Class Pairs: {model_names[1 - model_names.index(best_same_class_model)]}")


def main(args=None, fine_tune_path=None):

    parser = argparse.ArgumentParser()

    parser = phosc_net_argparse(parser)
    parser = dataset_argparse(parser)

    if args is None:
        args = parser.parse_args()
    args = parser.parse_args()

    if fine_tune_path:
        finetuned_model_save_path = ospj('models', args.split_name, 'bengali_words', '1', 'best.pt')
    else:
        finetuned_model_save_path = fine_tune_path

    # trained_model_save_path = ospj('models', 'trained_clip', split, 'super_aug', '2', 'best.pt')
    aling_fine_tuned_save_path = ospj('models', 'align-fine-tune', 'Fold0_use_50', '4', 'best.pt')

    align_fine_tuned_model.load_state_dict(torch.load(aling_fine_tuned_save_path))

    root_dir = ospj(DATA_FOLDER, "BengaliWords", "BengaliWords_CroppedVersion_Folds")
    image_loader_path = ospj(root_dir, args.split_name)

    loader = ImageLoader(image_loader_path)

    # Load phosc model
    phosc_model = get_phoscnet(args, device)

    test_loader, _ = get_test_loader(args)

    # CLIP
    # original_distances_same, original_distances_diffrent = evaluate_model(original_clip_model, test_loader, device, loader, ModelType.CLIP)
    # fine_tuned_distances_same, fine_tuned_distances_same_diffrent = evaluate_model(fine_tuned_clip_model, test_loader, device, fine_tuned_clip_model, loader)
    # summarize_results(('original', original_distances_same), ('fine_tuned', fine_tuned_distances_same))

    # ALIGN
    align_model_distances_same = evaluate_model(align_model, test_loader, device, loader, ModelType.ALIGN)
    align_fine_tuned_distances_same = evaluate_model(align_fine_tuned_model, test_loader, device, loader, ModelType.ALIGN)
    summarize_results(
        ('align_model', align_model_distances_same), 
        ('align_fine_tuned', align_fine_tuned_distances_same)
    )


if __name__ == '__main__':
    main()
