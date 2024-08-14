from modules.utils import generate_label_for_description, generate_phoc_vector, generate_phos_vector
from utils.dbe import dbe
from typing import Union
from num2words import num2words
import numpy as np

def split_string_into_chunks(input_string, chunk_size: int):
    """
    Split the input string into chunks of 'chunk_size' characters.

    Args:
        input_string (str): The string to be split.
        chunk_size (int): The maximum number of characters in each chunk.

    Returns:
        list of str: A list containing the split substrings.
    """
    # Use a list comprehension to split the string into chunks of 'chunk_size' characters
    return [input_string[i:i+chunk_size] for i in range(0, len(input_string), chunk_size)]


def gen_phos_label_description(label, name: str = '') -> str:
    shapes = [
        'left semi circle',
        'verticle line',
        'bottom semi-circle',
        'right semi-circle',
        'left top hood',
        'diagonal line (135◦), going from right to left',
        'diagonal line (45◦), going from left to right',
        'loop within a character',
        'dot below a character',
        'loop below the character',
        'horizontal line',
        'left small semi-circle',
        'right top hood'
    ]

    shape_description = ''

    for pyramid_level_idx, pyramid_level_data in enumerate(label):
        pyramid_level_ordinal_idx = num2words(pyramid_level_idx + 1, to='ordinal')

        for split, phos in enumerate(pyramid_level_data):
            # shape_description += f'The {pyramid_level_ordinal_idx} level {split}'
            # shape_description += f'In the {pyramid_level_ordinal_idx} level'

            for idx, shape in enumerate(phos[0]):
                shape_description += f'shape {num2words(idx + 1)} is present {num2words(int(shape))} times'

                if idx == len(phos[0]) - 1:
                    shape_description += '.\n'
                else:
                    shape_description += ', '

                # shape_description.append(text)

    return shape_description


def gen_phoc_label_description(label) -> str:
    characters = list(label)
    shape_description = ''

    for index, character in enumerate(characters):
        idx = num2words(index + 1, to='ordinal')
        shape_description += f'The {idx} character is {character}'

        if index == len(characters) - 1:
            shape_description += '.\n'
        else:
            shape_description += ', '

    return shape_description


def get_phosc_description(word: str) -> str:
    phos = generate_label_for_description(word, 1)
    phoc = generate_phoc_vector(word, level=1)

    description = ''

    # description += gen_phos_label_description(phos)
    # description += '\n'
    description += gen_phoc_label_description(word)

    return description

def get_phosc_number_description(word: str) -> str:
    phos = generate_phos_vector(word)
    phoc = generate_phoc_vector(word)

    phos = np.array(phos)
    phoc = [np.concatenate(sublist) for sublist in phoc]

    # flattened_phos = [item for sublist in phos for item in sublist]
    flattened_phos = phos.flatten()
    flattened_phoc = phoc.flatten()

    phos_str = ' '.join(str(int(x)) for x in flattened_phos)
    phoc_str = ' '.join(str(x) for x in flattened_phoc)

    dbe(phos_str, phoc_str)

    return ''