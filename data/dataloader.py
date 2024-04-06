import random
import torch
from torch.utils.data import Dataset, DataLoader


class AminoAcidTokenizer:
    def __init__(self, max_seq_length=500):
        self.amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
        self.vocab = {aa: idx + 4 for idx, aa in enumerate(self.amino_acids)}  # Start from 4 to reserve special tokens
        self.special_tokens = {
            '[PAD]': 0,
            '[CLS]': 1,
            '[SEP]': 2,
            '[MASK]': 3
        }
        self.vocab.update(self.special_tokens)
        self.inverse_vocab = {idx: aa for aa, idx in self.vocab.items()}
        self.max_seq_length = max_seq_length

    def encode(self, sequence):
        # Initialize the attention mask with zeros
        attention_mask = [0] * self.max_seq_length
        
        encoded = [self.vocab.get('[CLS]', 1)]  # Start of sequence token
        attention_mask[0] = 1  # CLS token is always attended to
        
        # Encode sequence and update attention mask
        for i, aa in enumerate(sequence[:self.max_seq_length-2]):
            encoded.append(self.vocab[aa])
            attention_mask[i + 1] = 1  # Mark this token as attended to
        
        encoded.append(self.vocab.get('[SEP]', 2))  # End of sequence token
        attention_mask[len(encoded) - 1] = 1  # SEP token is always attended to
        
        # Pad the sequence if it's shorter than max length
        pad_length = self.max_seq_length - len(encoded)
        encoded += [self.vocab.get('[PAD]', 0)] * pad_length
        
        return encoded, attention_mask

    def decode(self, token_ids):
        tokens = [self.inverse_vocab.get(id, '') for id in token_ids if id not in [self.special_tokens['[PAD]'], self.special_tokens['[CLS]'], self.special_tokens['[SEP]']]]
        return ''.join(tokens)



class ProteinDataset(Dataset):
    def __init__(self, sequences, tokenizer, mask_probability=0.15, min_len=None, max_len=None):
        self.tokenizer = tokenizer
        self.mask_probability = mask_probability
        self.min_len = min_len
        self.max_len = max_len

        # Filter sequences based on length
        if min_len is not None or max_len is not None:
            self.sequences = [seq for seq in sequences if self.is_within_length(seq)]
        else:
            self.sequences = sequences

    def is_within_length(self, sequence):
        # Check if sequence length is within specified range
        return (self.min_len is None or len(sequence) >= self.min_len) and (self.max_len is None or len(sequence) <= self.max_len)

    def random_masking(self, encoded_sequence):
        inputs = encoded_sequence.copy()
        targets = encoded_sequence.copy()
        for i, token in enumerate(encoded_sequence):
            if token not in [self.tokenizer.special_tokens['[PAD]'], self.tokenizer.special_tokens['[CLS]'], self.tokenizer.special_tokens['[SEP]']] and random.random() < self.mask_probability:
                inputs[i] = self.tokenizer.vocab['[MASK]']
                targets[i] = token
        return inputs, targets

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        encoded_sequence, attention_mask = self.tokenizer.encode(sequence)
        inputs, targets = self.random_masking(encoded_sequence)
        return {
            "input_ids": torch.tensor(inputs, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(targets, dtype=torch.long)
        }
