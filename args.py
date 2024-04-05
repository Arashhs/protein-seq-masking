import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Masked Protein Modeling')
    
    # Data arguments
    parser.add_argument('--data_path', type=str, default='dataset/prot-300.csv', help='Path to the dataset csv file')
    parser.add_argument('--max_seq_len', type=int, default=200, help='Maximum protein sequence length in the dataset')
    parser.add_argument('--mask_probability', type=float, default=0.15, help='Probability of masking tokens in the input sequence')

    # Model arguments
    parser.add_argument('--hidden_size', type=int, default=768, help='Hidden size of the transformer model')
    parser.add_argument('--num_hidden_layers', type=int, default=12, help='Number of hidden layers in the transformer model')
    parser.add_argument('--num_attention_heads', type=int, default=12, help='Number of attention heads in the transformer model')
    parser.add_argument('--intermediate_size', type=int, default=3072, help='Intermediate size of the transformer model')


    # Training arguments
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for training')
    
    args = parser.parse_args()
    return args
