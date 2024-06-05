# From: https://github.com/anuj-rai-23/PHOSC-Zero-Shot-Word-Recognition/blob/main/phoc_label_generator.py
"""
Module that generates 604 length PHOC vector as proposed in SPP-PHOCNet paper
Modified version from https://github.com/pinakinathc/phocnet_keras
"""

import csv
import numpy as np

def set_phoc_version(version_: str='eng'):
    global version
    version = version_


# Generates PHOC component corresponding to alphabets/digits

def generate_chars(word: str) -> np.ndarray:
    '''The vector is a binary and stands for:
    [0123456789abcdefghijklmnopqrstuvwxyz] 
    '''
    if version == 'eng' or version == 'gw':
        size = 36
    elif version == 'nor':
        size = 39
    elif version == 'ben':
        size = 80

    bengali_vec = ['অ', 'আ', 'ই', 'ঈ', 'উ', 'ঊ', 'ঋ', 'ৠ', 'ঌ', 'ৡ', 'এ', 'ঐ', 'ও', 'ঔ', 'ক', 'খ', 'গ', 'ঘ', 'ঙ', 'চ', 'ছ', 'জ', 'ঝ', 'ঞ', 'ট', 'ঠ', 'ড', 'ঢ', 'ণ', 'ত', 'থ', 'দ', 'ধ', 'ন', 'প', 'ফ', 'ব', 'ভ', 'ম', 'য়', 'ড়', 'ঢ়', 'য', 'র', 'ল', 'হ', 'শ', 'ষ', 'স', 'ৎ', 'ঽ', 'া', 'ি', 'ী', 'ু', 'ূ', 'ৃ', 'ৄ', 'ৢ', 'ৣ', 'ে', 'ৈ', 'ো', 'ৌ', 'ৗ', 'ঁ', 'ং', 'ঃ', '়', '্']

    vector = [0 for i in range(size)]
    for char in word:
        if version == 'eng':
            if char.isdigit():
                vector[ord(char) - ord('0')] = 1
            elif char.isalpha():
                vector[10+ord(char) - ord('a')] = 1
        
        elif version == 'nor':
            if char.isdigit():
                vector[ord(char) - ord('0')] = 1
            elif char.isalpha():
                if char == 'æ':
                    vector[36] = 1
                elif char == 'ø':
                    vector[37] = 1
                elif char == 'å':
                    vector[38] = 1
                else:
                    vector[10+ord(char) - ord('a')] = 1
                    
        elif version == 'ben':
            if char.isdigit():
                vector[ord(char) - ord('0')] = 1
            elif char in bengali_vec:
                vector[10 + bengali_vec.index(char)] = 1
    
    return np.array(vector)

# Generates PHOC component corresponding to 50 most frequent bi-grams of English

def generate_50(word):
    if version == 'eng' or version == 'gw':
        bigram = ['th', 'he', 'in', 'er', 'an', 're', 'es', 'on', 'st', 'nt', 'en',
                'at', 'ed', 'nd', 'to', 'or', 'ea', 'ti', 'ar', 'te', 'ng', 'al',
                'it', 'as', 'is', 'ha', 'et', 'se', 'ou', 'of', 'le', 'sa', 've',
                'ro', 'ra', 'hi', 'ne', 'me', 'de', 'co', 'ta', 'ec', 'si', 'll',
                'so', 'na', 'li', 'la', 'el', 'ma']
    elif version == 'nor':
        bigram = ['de', 'og', 'ha', 'je', 'at', 'me', 'fo', 'en', 'ti', 'er', 'mi',
                  'vi', 'so', 'sa', 'he', 'si', 'ik', 'af', 'sk', 'st', 'ma', 'be',
                  'hv', 'al', 'fr', 'va', 've', 'om', 'pa', 'et', 'se', 'di', 'da',
                  'li', 'bl', 'in', 'du', 'no', 'ko', 'an', 'væ', 'fa', 'ku', 'ka',
                  'ga', 'hu', 'ta', 're', 'ud', 'op']

    print(len(bigram))

    vector_50 = [0 for i in range(50)]
    for char in word:
        try:
            vector_50[bigram.index(char)] = 1
        except:
            continue

    return vector_50

# Input: A word(string)
# Output: PHOC vector
## Levels 1,2,3,4,5
def generate_phoc_vector(word, level=6):
    word = word.lower()
    # vector = generate_chars(word)
    vector = [generate_chars(word)]
    L = len(word)

    return_vector = [[vector]]

    for split in range(2, level): #split 3 
        parts = L//split # parts 3
        vec = list()

        for mul in range(split-1): # 0 - 2
            vec.append([generate_chars(word[mul*parts:mul*parts+parts])])

        vec.append([generate_chars(word[(split-1)*parts:L])])
     
        return_vector.append(vec)

    return return_vector


# Input: A list of words(strings)
# Output: A dictionary of PHOC vectors in which the words serve as the key

def gen_phoc_label(word_list):
    label={}
    for word in word_list:
        label[word]=generate_phoc_vector(word)
    return label

# Input: A text file name that has a list of words(strings)
# Output: A dictionary of PHOC vectors in which the words serve as the key

def label_maker(word_txt):
    label={}
    with open(word_txt, "r") as file:
        for word_index, line in enumerate(file):
            word = line.split()[0]
            label[word]=gen_phoc_label(word)
    return label
    #write_s_file(s_matrix_csv, s_matrix, word_list)


if __name__ == '__main__':
    set_phoc_version('ben')
    # print(generate_chars('1aøæÅ'.lower()))

    phoc_vector = generate_phoc_vector('চুরপুনি')
    
    print(phoc_vector)
    print(len(phoc_vector))