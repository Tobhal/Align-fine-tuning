# From: https://github.com/anuj-rai-23/PHOSC-Zero-Shot-Word-Recognition/blob/main/phos_label_generator.py
import csv
import numpy as np
from pathlib import Path
from num2words import num2words

from typing import List

from typing import TextIO
import os

from utils.dbe import dbe


# Input: CSV file name that has shape counts for each alphabet
# Output: Number of shapes/columns

def get_number_of_columns(csv_file: TextIO) -> int:
    with open(csv_file) as file:
        reader = csv.reader(file, delimiter=',', skipinitialspace=True)
        return len(next(reader)) - 1


# Input: CSV file name that has shape counts for each alphabet
# Output: A dictionary where alphabet is key mapped to its shape count vector(np-array)

def create_alphabet_dictionary(csv_file: TextIO) -> dict:
    alphabet_dict = dict()

    with open(csv_file) as file:
        reader = csv.reader(file, delimiter=',', skipinitialspace=True)

        for index, line in enumerate(reader):
            alphabet_dict[line[0]] = index

    return alphabet_dict


def set_phos_version(version='eng'):
    global alphabet_csv, alphabet_dict, csv_num_cols, numpy_csv

    root = Path('modules/utils/')

    if version == 'eng':
        alphabet_csv = root / 'Alphabet.csv'
    elif version == 'gw':
        alphabet_csv = root / 'AlphabetGW.csv'
    elif version == 'nor':
        alphabet_csv = root / 'AlphabetNorwegian.csv'
    elif version == 'ben':
        alphabet_csv = root / 'AlphabetBengali.csv'

    alphabet_dict = create_alphabet_dictionary(alphabet_csv)
    csv_num_cols = get_number_of_columns(alphabet_csv)
    numpy_csv = np.genfromtxt(alphabet_csv, dtype=int, delimiter=",")
    numpy_csv = np.delete(numpy_csv, 0, 1)


# Input: A word segment(string)
# Output: A shape count vector for all alphabets in input word segment (np-array)

def word_vector(word: str) -> np.ndarray:
    vector = np.zeros(csv_num_cols)
    
    for letter in word:
        letter_index = alphabet_dict[letter]
        vector += numpy_csv[letter_index]

    return [vector]


# Input: A word(string)
# Output: PHOS vector
## Levels 1,2,3,4,5
def generate_label(word: str) -> np.ndarray:
    vector = word_vector(word)
    L = len(word)

    for split in range(2, 6):
        parts = L // split

        for mul in range(split - 1):
            vector = np.concatenate((vector, word_vector(word[mul * parts:mul * parts + parts])), axis=0)
        vector = np.concatenate((vector, word_vector(word[(split - 1) * parts:L])), axis=0)

    return vector


# Input: A list of words(strings)
# Output: A dictionary of PHOS vectors in which the words serve as the key
def gen_phos_label(word_list) -> dict:
    label = {}

    for word in word_list:
        label[word] = generate_label(word)

    return label


# Input: A word(string)
# Output: Each level of PHOS vector
## Levels 1,2,3,4,5
def generate_label_for_description(word: str, level=6) -> List[np.ndarray]:
    # TODO: Write what part of the word is considred. So for any layer other than 0 the word will be split into multiple parts
    vector = word_vector(word)
    L = len(word)

    return_vector = [[vector]]

    for split in range(2, level):
        parts = L // split
        vec = list()

        for mul in range(split - 1):
            vec.append(word_vector(word[mul * parts:mul * parts + parts]))

        vec.append(word_vector(word[(split - 1) * parts:L]))

        return_vector.append(vec)  # Append the temporary vector to the list

    return return_vector


# Input: A text file name that has a list of words(strings)
# Output: A dictionary of PHOS vectors in which the words serve as the key
def label_maker(word_txt: TextIO) -> dict:
    label = {}

    with open(word_txt, "r") as file:
        for word_index, line in enumerate(file):
            word = line.split()[0]
            label[word] = generate_label(word)

    return label
    # write_s_file(s_matrix_csv, s_matrix, word_list)


def gen_shape_description_simple(word: str) -> List[str]:
    single_phos = word_vector(word)

    shape_description = []

    for pyramid_data in single_phos[0]:
        shape_description.append(f'{int(pyramid_data)}')

    shape = [''.join(shape_description)]

    return shape

def flatten_vector(vector):
    """Flatten the given vector structure into a single list of integers."""
    flattened_list = []

    # Function to recursively extract arrays and flatten them
    def extract_and_flatten(item):
        if isinstance(item, list):
            for subitem in item:
                extract_and_flatten(subitem)
        elif isinstance(item, np.ndarray):
            flattened_list.extend(item.tolist())

    extract_and_flatten(vector)
    return list(map(int, flattened_list))

def gen_shape_description(word: str) -> List[str]:
    single_phos = word_vector(word)

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

    """
    test = generate_label_for_description(word, 3)

    all_arrays = np.concatenate([array for sublist in test for inner_list in sublist for array in inner_list])
    flattened_array = all_arrays.flatten()

    # Convert to string
    concatenated_string_flattened = ''.join(str(int(value)) for value in flattened_array)
    return concatenated_string_flattened
    dbe(concatenated_string_flattened, len(concatenated_string_flattened))

    # Concatenate into a single string with each float value converted to an int
    concatenated_string = ''.join(str(int(value)) for array in single_phos for value in array)
    dbe(concatenated_string)
    """

    phos = generate_label_for_description(word)

    for pyramid_level_idx, pyramid_level_data in enumerate(phos):
        pyramid_level_ordinal_idx = num2words(pyramid_level_idx + 1, to='ordinal')

        for split, phos in enumerate(pyramid_level_data):
            # shape_description += f'In the {pyramid_level_ordinal_idx} level (split {num2words(split)})'
            shape_description += f'{pyramid_level_ordinal_idx} level'

            for idx, shape in enumerate(phos[0]):
                shape_description += f', shape {num2words(idx + 1)} is present {num2words(int(shape))} times'

                if idx == len(phos[0]) - 1:
                    shape_description += '.\n'

    dbe(shape_description)
    return shape_description
"""
    for pyramid_level_idx, pyramid_level_data in enumerate(phos):
        pyramid_level_ordinal_idx = num2words(pyramid_level_idx + 1, to='ordinal')

        shape_description += f'In the {pyramid_level_ordinal_idx} level'

        for split, _phos in enumerate(pyramid_level_data):
            for idx, shape in enumerate(_phos[0]):
                shape_description += f', shape {num2words(idx + 1)} is present {num2words(int(shape))} times'

        shape_description += '.\n'

    return shape_description
"""

def gen_shape_description_from_phos(phos) -> List[str]:
    shape_description = []

    for idx, shape in enumerate(single_phos):
        ordinal_idx = num2words(idx + 1, to='ordinal')
        shape_description.append(f'the {ordinal_idx} shape is {shape}.\n')

    return shape_description


if __name__ == '__main__':
    set_phos_version('ben')

    word = 'অঐত'

    regular_phos = generate_label(word)
    
    print(regular_phos.shape)
    # new_phos = generate_label_NOR('zz')
    
    print(regular_phos)
    print(len(regular_phos))

    single_phos = word_vector(word)
    print(single_phos)
    print(len(single_phos))

    print(gen_shape_description(word))