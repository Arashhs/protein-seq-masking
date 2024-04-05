import torch
from torch.utils.data import Dataset, DataLoader
from data.dataloader import AminoAcidTokenizer, ProteinDataset
from data.utils import load_data

from transformers import BertForMaskedLM, BertConfig

from args import parse_args

from tqdm import tqdm



def main(args):
    # Load data
    sequences = load_data(args.max_seq_len, path=args.data_path)
    
    # Initialize tokenizer
    tokenizer = AminoAcidTokenizer(max_seq_length=args.max_seq_len)
    
    # Initialize dataset
    dataset = ProteinDataset(sequences, tokenizer, mask_probability=args.mask_probability)
    
    # Initialize dataloader
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

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
        for batch in tqdm(dataloader):
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