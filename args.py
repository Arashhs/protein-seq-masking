import argparse
import torch

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args():
    parser = argparse.ArgumentParser(description='Masked Protein Modeling')
    
    
    # General args
    parser.add_argument('--save_dir',
                        type=str,
                        default='./saved_results/',
                        help='Directory to save the outputs and checkpoints.')
    parser.add_argument('--rand_seed',
                        type=int,
                        default=117, # John-117 :>
                        help='Random seed.')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='dataset/splitted', help='Path to the directory containing train csv and test/val pt files')
    parser.add_argument('--max_seq_len', type=int, default=100, help='Maximum protein sequence length in the dataset')
    parser.add_argument('--mask_probability', type=float, default=0.15, help='Probability of masking tokens in the input sequence')

    # Model arguments
    parser.add_argument('--hidden_size', type=int, default=768, help='Hidden size of the transformer model')
    parser.add_argument('--num_hidden_layers', type=int, default=12, help='Number of hidden layers in the transformer model')
    parser.add_argument('--num_attention_heads', type=int, default=12, help='Number of attention heads in the transformer model')
    parser.add_argument('--intermediate_size', type=int, default=3072, help='Intermediate size of the transformer model')


    # Training arguments
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for training')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Which device to use (CUDA or CPU).')
    
    # Logging args
    parser.add_argument('--use_wandb',
                        type=str2bool,
                        default=True,
                        help='Whether use wandb to log results.')
    parser.add_argument('--wandb_entity',
                        type=str,
                        default='tlouhs',
                        help='Wandb entity.')
    parser.add_argument('--wandb_project',
                        type=str,
                        default='protein-masked-modeling',
                        help='Wandb project.')
    parser.add_argument('--run_name',
                        type=str,
                        default='BERTMLM',
                        help='The name of the run.')
    
    args = parser.parse_args()
    return args
