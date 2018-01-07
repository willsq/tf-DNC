from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pickle
import getopt
import time
import sys
import os
import random
import tensorflow as tf

np.random.seed(1)


def llprint(message):
    sys.stdout.write(message)
    sys.stdout.flush()

def load(path):
    return pickle.load(open(path, 'rb'))

def onehot(index, size):
    vec = np.zeros(size, dtype=np.float32)
    vec[int(index)] = 1.0
    return vec

def prepare_sample(sample, end_of_query_symbol, word_space_size):
    input_vec = np.array(sample['inputs'], dtype=np.float32)
    output_vec = np.array(sample['inputs'], dtype=np.float32)
    seq_len = input_vec.shape[0]
    weights_vec = np.zeros(seq_len, dtype=np.float32)

    target_mask = (input_vec == end_of_query_symbol)
    output_vec[target_mask] = sample['outputs']
    weights_vec[target_mask] = 1.0

    input_vec = np.array([onehot(code, word_space_size) for code in input_vec])
    output_vec = np.array([onehot(code, word_space_size) for code in output_vec])

    return (
        np.reshape(input_vec, (-1, 1, word_space_size)),
        np.reshape(output_vec, (-1, 1, word_space_size)),
        word_space_size,
        np.reshape(weights_vec, (-1, 1, 1))
    )

def get_sample(batch_size, data_dir):
    lexicon_dict = load(os.path.join(data_dir, 'lexicon-dict.pkl'))
    data = load(os.path.join(data_dir, 'train', 'train.pkl'))

    data_shuffled = random.shuffle(data)
    word_space_size = len(lexicon_dict)
    while(True):
        for sample in data:
            yield prepare_sample(sample, lexicon_dict['-'], word_space_size)
