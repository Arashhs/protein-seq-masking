import torch
from torch.utils.data import DataLoader, TensorDataset
from data.dataloader import AminoAcidTokenizer, ProteinDataset
from data.utils import load_train_data, load_test_data


from transformers import BertForMaskedLM, BertConfig

from args import parse_args

from tqdm import tqdm
import pandas as pd



def main(args):
    # Load data
    train_sequences = load_train_data(args.data_dir + '/train.csv')
    val_sequences, val_labels, val_masks = load_test_data(args.max_seq_len, args.data_dir, prefix='val')
    test_sequences, test_labels, test_masks = load_test_data(args.max_seq_len, args.data_dir, prefix='test')
    
    # Initialize tokenizer
    tokenizer = AminoAcidTokenizer(max_seq_length=args.max_seq_len)
    
    # Initialize dataset
    train_dataset = ProteinDataset(train_sequences, tokenizer, mask_probability=args.mask_probability, max_len=args.max_seq_len)
    val_dataset = TensorDataset(val_sequences, val_masks, val_labels)
    test_dataset = TensorDataset(test_sequences, test_masks, test_labels)

    
    # Initialize dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size)

    # initialize model
    config = BertConfig(vocab_size=len(tokenizer.vocab), 
                        hidden_size=args.hidden_size, 
                        num_hidden_layers=args.num_hidden_layers, 
                        num_attention_heads=args.num_attention_heads, 
                        intermediate_size=args.intermediate_size)
    
    model = BertForMaskedLM(config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Iterate over the dataset
    for epoch in range(args.num_epochs):
        print(f'Epoch {epoch + 1}')
        total_loss = 0
        for batch in tqdm(train_dataloader):
            inputs, attention_mask, targets = batch['input_ids'], batch['attention_mask'], batch['labels']
            inputs, attention_mask, targets = inputs.to(device), attention_mask.to(device), targets.to(device)
            # Forward pass
            outputs = model(inputs, attention_mask=attention_mask, labels=targets)
            loss = outputs.loss
            total_loss += loss.item()
            # Backward pass
            loss.backward()
            print(loss)
            # Update weights
            optimizer.step()
            # Clear gradients
            optimizer.zero_grad()
        print(f'Total loss: {total_loss}')
    




if __name__ == '__main__':
    args = parse_args()
    main(args)