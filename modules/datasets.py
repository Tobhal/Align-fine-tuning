import os

import torch
from torch.utils.data import Dataset
# from skimage import io
import cv2 as cv

from PIL import Image
from torchvision.transforms import transforms
from modules.utils import generate_phoc_vector, generate_phos_vector, set_phos_version, set_phoc_version
# from utils import generate_phoc_vector, generate_phos_vector, set_phos_version, set_phoc_version

import pandas as pd
import numpy as np

from utils.dbe import dbe


def pad_to_fixed_size(image, size):
    # get the original image dimensions
    width, height = image.size

    # calculate the difference between the desired and original dimensions
    delta_width = size[0] - width 
    delta_height = size[1] - height

    # divide the difference by two to get equal padding on both sides
    pad_width = delta_width // 2 
    pad_height = delta_height // 2

    # create a padding transform with the calculated values
    padding = (pad_width, pad_height, delta_width - pad_width, delta_height - pad_height) 
    transform = transforms.Pad(padding)

    # apply the transform to the image
    return transform(image)


class phosc_dataset(Dataset):
    def __init__(self, csvfile, root_dir, language='eng', transform=None, image_size=(250, 50), image_resize=(250, 50)):
        set_phos_version(language)
        set_phoc_version(language)

        self.df_all = pd.read_csv(csvfile)
        self.root_dir = root_dir
        self.transform = transform

        self.image_size = image_size
        self.image_resize = image_resize

        words = self.df_all["Word"].values

        phos_vects = []
        phoc_vects = []
        phosc_vects = []

        for word in words:
            phos = generate_phos_vector(word)
            phoc = np.array(generate_phoc_vector(word))
            phosc = np.concatenate((phos, phoc))

            phos_vects.append(phos)
            phoc_vects.append(phoc)
            phosc_vects.append(phosc)


        self.df_all["phos"] = phos_vects
        self.df_all["phoc"] = phoc_vects
        self.df_all["phosc"] = phosc_vects

        # print(self.df_all)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.df_all['Image'][index])
        
        # Read image
        image = Image.open(img_path)
        # Scale image to `image_size` while keeping aspect ratio
        image.thumbnail(self.image_size, Image.ANTIALIAS)

        # Create a new black image with the size of `image_size`
        black_image = Image.new("RGB", self.image_size)

        # Paste the image on a the black image to turn in to a image the size we want
        black_image.paste(
            image, 
            (
                int((self.image_size[0] - image.size[0]) / 2), 
                int((self.image_size[1] - image.size[1]) / 2)
            )
        )

        # Scale down the image to a smaler size for working with.
        image = black_image.resize(self.image_resize, Image.ANTIALIAS)

        # print(image.shape)

        if self.transform:
            image = self.transform(image)
            
        word = self.df_all['Word'][index]

        phos = torch.tensor(self.df_all['phos'][index]).clone()
        phoc = torch.tensor(self.df_all['phoc'][index]).clone()
        phosc = torch.tensor(self.df_all['phosc'][index]).clone()
        
        item = {
            'image': image.float(),
            'word': word,
            'y_vectors': {
                'phos': phos.float(),
                'phoc': phoc.float(),
                'phosc': phosc.float(),
                'sim': 1
            }
        }

        return item
        # return image.float(), y.float(), self.df_all.iloc[index, 1]

    def __len__(self):
        return len(self.df_all)

class CharacterCounterDataset(Dataset):
    def __init__(self, longest_word_len, csvfile, root_dir, transform=None):
        self.df_all = pd.read_csv(csvfile)
        self.root_dir = root_dir
        self.transform = transform

        words = self.df_all["Word"].values

        targets = []

        for word in words:
            target = np.zeros((longest_word_len))
            target[len(word)-1] = 1
            targets.append(target)

        self.df_all["target"] = targets

        # print(self.df_all)

        # print(self.df_all.iloc[0, 5].shape)
        # print(self.df_all.to_string())

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.df_all.iloc[index, 0])
        image = cv.imread(img_path)

        y = torch.tensor(self.df_all.iloc[index, len(self.df_all.columns) - 1])

        if self.transform:
            image = self.transform(image)

        # returns the image, target vector and the corresponding word
        return image.float(), y.float(), self.df_all.iloc[index, 1]

    def __len__(self):
        return len(self.df_all)


if __name__ == '__main__':
    from torchvision.transforms import transforms

    # dataset = phosc_dataset('image_data/IAM_Data/IAM_valid_unseen.csv', 'image_data/IAM_Data/IAM_valid', 'nor', transform=transforms.ToTensor())
    dataset = phosc_dataset('image_data/GW_Data/cv1_valid_seen.csv', 'image_data/GW_Data/CV1_valid', 'eng', transform=transforms.ToTensor())
    # dataset = phosc_dataset('image_data/norwegian_data/train_gray_split1_word50.csv', 'image_data/norwegian_data/train_gray_split1_word50', 'nor', transform=transforms.ToTensor())
    dataloader = torch.utils.data.DataLoader(dataset, 5)
    # print(dataset.df_all)


    for batch in dataloader:
        print(batch['image'].shape)
        print(batch['y_vectors']['phos'].shape)
        print(batch['y_vectors']['phoc'].shape)
        print(batch['y_vectors']['phosc'].shape)
        print(batch['y_vectors']['sim'].shape)
        quit()

    # print(dataset.__getitem__(0))