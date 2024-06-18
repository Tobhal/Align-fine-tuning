import os
import shutil
import random
import csv
import numpy as np
import argparse

import dataset_manipulation.augmentation as augmentation

from os.path import join as ospj
from utils.dbe import dbe
from flags import DATA_FOLDER

from PIL import Image
from tqdm import tqdm

from parser import prepear_data_argparse

def save_csv(data: list, folder: os.PathLike, file_name: str='data.csv'):
    with open(ospj(folder, file_name), 'w', newline='') as fp:
        writer = csv.writer(fp)
        writer.writerow(('Image', 'Word', 'Language'))

        # Sorting csv_data based on index and filename
        data.sort(key=lambda x: (int(x[0].split('/')[1]), x[0].split('/')[2]))
        
        writer.writerows(data)


def split_and_move_folder(input_folder: os.PathLike, output_folder: os.PathLike, class_file: os.PathLike, fold, percentage=0.2):
    """
    Move a percentage of classes from the input directory to the destination directory.

    Args:
        base_dir (path): The base directory where the input and destination directories are located.
        input_dir (path): The path to the input directory.
        dest_dir (path): The path to the destination directory.
        classes_file (str): The name of the file containing the class labels.
        percentage (float): The percentage of classes to move to the destination directory.
    """
    dirs = [dir for dir in os.listdir(input_folder) if os.path.isdir(os.path.join(input_folder, dir))]

    selected = random.sample(dirs, int(len(dirs)*percentage))
    
    # Read the original text file
    with open(os.path.join(input_folder, class_file), 'r') as f:
        classes = f.read().splitlines()

    # Prepare a list for the moved classes
    moved_classes = []
    remaining_classes = []

    for i, class_ in enumerate(classes):
        if f"{i}" in selected:
            moved_classes.append(class_)
        else:
            remaining_classes.append(class_)

    for dir in selected:
        shutil.move(os.path.join(input_folder, dir), os.path.join(output_folder, dir))

    # Write the moved classes into a new text file
    with open(os.path.join(output_folder, f'Val_Labels_Fold{fold}.txt'), 'w') as f:
        f.write('\n'.join(moved_classes))

    # Rewrite the original file with remaining classes
    with open(os.path.join(input_folder, class_file), 'w') as f:
        f.write('\n'.join(remaining_classes))


def gen_data_csv(folder: os.PathLike, fold: str):
    """
    Generate a CSV file from the input directory.

    Args:
        folder (path): The path to the input directory.
    """
    csv_data = []
    total_images = 0

    d = 0
    fl = 0

    phase = os.path.basename(os.path.normpath(folder))
    label_cappitalized = phase.capitalize()

    label_file = f'{label_cappitalized}_Labels_Fold{fold}.txt'

    # Get a list of all directory indices (as integers) present in the input_directory
    dir_indices = [int(dir_name) for dir_name in os.listdir(folder) if os.path.isdir(os.path.join(folder, dir_name))]
    dir_indices.sort()
    
    # initialize all labels
    labels_dict = {}
    with open(os.path.join(folder, label_file), 'r') as f:
        for i, l in enumerate(f):
            labels_dict[str(dir_indices[i])] = l.strip()

    # Walk through all subdirectories of the root folder
    for root, dirs, files in os.walk(folder):
        d += 1
        
        # Walk through all files in the current directory
        for file_name in files:
            fl += 1
            
            # Check if the file is an image
            if file_name.endswith('.jpg') or file_name.endswith('.png'):
                total_images += 1
                new_image_path = os.path.join(root, file_name)
                dir_index = os.path.basename(root)
                
                # Append the directory name to the file name to ensure uniqueness
                new_file_name = ospj(phase, dir_index, file_name)

                if dir_index in labels_dict:
                    # Add new file name to the CSV data
                    csv_data.append((new_file_name, labels_dict[dir_index], 'Bengali'))

    # Save the CSV data to a file
    save_csv(csv_data, folder)
    save_csv(csv_data, os.path.dirname(folder), file_name=f'{phase}_pairs.csv')


def rezie_images(
        folder: os.PathLike, 
        size: tuple = None, 
        down_scale_factor: float = None
    ):
    """
    Resize all images in the input directory to the specified size or keep them at max size.

    Args:
        folder (path): The path to the input directory.
        size (tuple): The new size of the images. If None, the images will be kept at max size.
        down_scale_factor (float): The factor by which to downscale the images. If None, the images will be kept at max size.
    """
    # Step 1: Find the largest width and height among all images
    max_width = 0
    max_height = 0

    for dirpath, dirnames, filenames in os.walk(folder):
        for filename in filenames:
            # Check if the file is an image
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                # Construct full file path
                file_path = os.path.join(dirpath, filename)
                # Open the image
                with Image.open(file_path) as img:
                    width, height = img.size
                    # Update max dimensions
                    max_width = max(max_width, width)
                    max_height = max(max_height, height)

    # Step 2: Resize and pad each image
    # Walk through all subdirectories of the root folder
    for dirpath, dirnames, filenames in tqdm(os.walk(folder), desc='Resizing images'):
        for filename in filenames:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                file_path = ospj(dirpath, filename)

                with Image.open(file_path) as img:
                    # Scale the image while maintaining its aspect ratio
                    # Calculate the scaling factor for both dimensions
                    scale_width = max_width / img.width
                    scale_height = max_height / img.height

                    # Use the smaller scaling factor to ensure the image fits within the constraints
                    scale_factor = min(scale_width, scale_height)

                    # New dimensions
                    new_width = int(img.width * scale_factor)
                    new_height = int(img.height * scale_factor)

                    img = img.resize((new_width, new_height), Image.LANCZOS)

                    # Pad the image
                    padded_img = Image.new("RGB", (max_width, max_height), (255, 255, 255))
                    x_offset = (max_width - new_width) // 2
                    y_offset = (max_height - new_height) // 2
                    padded_img.paste(img, (x_offset, y_offset))

                    # Resize the image to the specified size, if provided
                    if size:
                        padded_img = padded_img.resize(size, Image.LANCZOS)

                    if down_scale_factor:
                        padded_img = padded_img.resize((int(max_width * down_scale_factor), int(max_height * down_scale_factor)), Image.LANCZOS)

                    # Save the image to overwrite the original file
                    padded_img.save(file_path)


def random_factor(low, high):
    return np.random.uniform(low, high)


def augment_data(
        folder: os.PathLike,
        total_augmentations: int=20,
        noise_variability: int=60,
        max_shearx_factor: int=1,
        max_sheary_factor: int=0.05,
        max_augmentations: int=6
    ):
    # Define a dictionary that maps operation numbers to functions
    augmentations = {
        0: lambda img: augmentation.noise_image(img, noise_variability),
        1: lambda img: augmentation.shear_x(img, random_factor(-max_shearx_factor, max_shearx_factor)),
        2: lambda img: augmentation.shear_y(img, random_factor(-max_sheary_factor, max_sheary_factor)),
        3: lambda img: augmentation.random_perspective(img),
        4: lambda img: augmentation.erode(img, 1),
        5: lambda img: augmentation.dialate(img, 1),
        6: lambda img: augmentation.blur(img, random_factor(1, 2)),
        7: lambda img: augmentation.sharpness(img, random_factor(5, 10))
    }

    label = os.path.basename(os.path.normpath(folder))
    folder_parent = os.path.dirname(folder)
    augmentation_folder = ospj(folder_parent, f'{label}-aug')

    labels = dict()

    # Get labels
    with open(ospj(folder, 'data.csv'), 'r') as f:
        reader = csv.reader(f)
        next(reader)

        for img_path, l, _ in reader:
            label_id = int(img_path.split('/')[1])

            if label_id not in labels:
                labels[label_id] = l

    # Create the augmentation folder
    os.makedirs(augmentation_folder, exist_ok=True)

    csv_data = []

    for dirpath, _, filenames in tqdm(os.walk(folder), desc='Augmenting images'):
        for filename in filenames:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                file_num = os.path.basename(dirpath)
                phase = os.path.basename(os.path.dirname(dirpath))

                to_folder = ospj(augmentation_folder, file_num)

                os.makedirs(to_folder, exist_ok=True)

                # Save current image to the augmentation folder
                shutil.copy(ospj(dirpath, filename), to_folder)
                csv_data.append((ospj(f'{phase}-aug', file_num, filename), labels[int(file_num)], 'Bengali'))

                # Number of augmentations to perform on each image
                for i in range(total_augmentations):
                    file_path = ospj(dirpath, filename)

                    img = Image.open(file_path)

                    # Perform a random number of augmentations on the image
                    random_samples = random.sample(list(augmentations.keys()), random.randint(1, max_augmentations))
                    for op_n in random_samples:
                        img = augmentations[op_n](img)

                    # Save the augmented image
                    filename_no_ext, ext = os.path.splitext(filename)

                    file_name = f'{filename_no_ext}_aug{i}{ext}'

                    img.save(ospj(to_folder, file_name))

                    csv_data.append((ospj(f'{phase}-aug', file_num, file_name), labels[int(file_num)], 'Bengali'))

    # Save the CSV data to a file
    save_csv(csv_data, augmentation_folder)
    save_csv(csv_data, folder_parent, file_name=f'{label}-aug_pairs.csv')


def main(args=None):
    parser = argparse.ArgumentParser()

    parser = prepear_data_argparse(parser)

    if args:
        args = parser.parse_args(args)
    else:
        args = parser.parse_args()

    splits = [
        'train', 
        'test', 
    ]

    for fold in args.folds:
        original_data_path = ospj(args.original_data, f'fold_{fold}')   # Path to train and test data
        output_data_path = ospj(args.output_dir, f'fold_{fold}_{args.output_name}')  # Path to save the new data

        print(f'{original_data_path} -> {output_data_path}')

        # Copy data from original to output folder
        shutil.copytree(original_data_path, output_data_path)

        # Split train into train and validation
        if args.split:
            splits.append('val')

            split_and_move_folder(
                ospj(output_data_path, 'train'), 
                ospj(output_data_path, 'val'), 
                f'Train_Labels_Fold{fold}.txt', 
                fold, 
                percentage=args.split_ratio
            )

        # Generate CSV files
        for split in splits:
            gen_data_csv(ospj(output_data_path, split), fold)

        # Rezie images
        rezie_images(ospj(output_data_path), down_scale_factor=args.down_scale_factor)

        # Augment data
        if args.augmented:
            augment_data(
                ospj(output_data_path, 'train'), 
                total_augmentations=args.augmentations_per_image
            )


if __name__ == '__main__':
    main()