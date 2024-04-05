import pandas as pd
import numpy as np


def load_data(max_seq_len, path='../dataset/prot-300.csv'):
    sequences = pd.read_csv(path)
    sequences = sequences[sequences['seq_length'] <= max_seq_len]
    return sequences['seq'].tolist()