import pandas as pd
import numpy as np
import torch


def load_train_data(path):
    sequences = pd.read_csv(path)
    return sequences['seq'].tolist()


def load_test_data(max_seq_len, test_dir, prefix='data', device='cpu'):
    parent_dir = f'{test_dir}/max_seq{max_seq_len}'
    inputs = torch.load(f'{parent_dir}/{prefix}_inputs_{max_seq_len}.pt').to(device)
    labels = torch.load(f'{parent_dir}/{prefix}_labels_{max_seq_len}.pt').to(device)
    masks = torch.load(f'{parent_dir}/{prefix}_masks_{max_seq_len}.pt').to(device)
    return inputs, labels, masks