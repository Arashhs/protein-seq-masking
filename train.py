import torch
from torch.utils.data import DataLoader, TensorDataset
from data.dataloader import AminoAcidTokenizer, ProteinDataset
from tqdm import tqdm
import pandas as pd
from datetime import datetime 
from transformers import BertForMaskedLM, BertConfig
import json

from args import parse_args
from data.utils import load_train_data, load_test_data
from utils.logger import WandbLogger





def train(train_dataloader, val_dataloader, model, args, wandb_logger=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    model.float()
    
    # put the model in training mode
    wandb_logger.watch_model(model)
    
    step = 0
    print(f'Start training on {len(train_dataloader.dataset)} samples.')
    
    for epoch in range(args.num_epochs):
        total_loss = 0

        model.train()
        for batch in tqdm(train_dataloader):
            inputs, attention_mask, targets = batch['input_ids'], batch['attention_mask'], batch['labels']
            inputs, attention_mask, targets = inputs.to(args.device), attention_mask.to(args.device), targets.to(args.device)
            
            # Clear the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs, attention_mask=attention_mask, labels=targets)

            loss = outputs.loss
            total_loss += loss.item()

            # Backward pass and step
            loss.backward()
            optimizer.step()

            # Log the loss
            step += args.batch_size
            wandb_logger.log('train/loss', loss.item(), step)
            wandb_logger.log('train/learning_rate', optimizer.param_groups[0]['lr'], step)

            total_loss += loss.item()


        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_dataloader:
                inputs, attention_mask, targets = batch
                inputs, attention_mask, targets = inputs.to(args.device), attention_mask.to(args.device), targets.to(args.device)
                
                outputs = model(inputs, attention_mask=attention_mask, labels=targets)
                loss = outputs.loss
                total_val_loss += loss.item()
                
        validation_loss = total_val_loss/ len(val_dataloader)

        wandb_logger.log('val/loss', validation_loss, step)
        print(f"Epoch {epoch+1}/{args.num_epochs}, Loss (train): {total_loss/ len(train_dataloader)}, Loss (val): {validation_loss}")




def main(args):
    args.run_name = f'{args.run_name}_{str(datetime.now().strftime("%Y-%m-%d %H:%M"))}'
    wandb_logger = WandbLogger(is_used=args.use_wandb, entity=args.wandb_entity, project=args.wandb_project, name=args.run_name)
    # log hyperparameters
    print('Args: {}'.format(json.dumps(vars(args), indent=4, sort_keys=True)))
    wandb_logger.log_hyperparams(args)
    
    # Load data
    train_sequences = load_train_data(args.data_dir + '/train.csv')
    val_sequences, val_labels, val_masks = load_test_data(args.max_seq_len, args.data_dir, prefix='val', device=args.device)
    test_sequences, test_labels, test_masks = load_test_data(args.max_seq_len, args.data_dir, prefix='test', device=args.device)
    
    # Initialize tokenizer
    tokenizer = AminoAcidTokenizer(max_seq_length=args.max_seq_len)
    
    # Initialize dataset
    train_dataset = ProteinDataset(train_sequences, tokenizer, mask_probability=args.mask_probability, max_len=args.max_seq_len)
    val_dataset = TensorDataset(val_sequences, val_masks, val_labels)
    test_dataset = TensorDataset(test_sequences, test_masks, test_labels)
    
    # print length of the datasets
    print('Length of the train dataset:', len(train_dataset))
    print('Length of the validation dataset:', len(val_dataset))
    print('Length of the test dataset:', len(test_dataset))
    
   
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
    
    model = BertForMaskedLM(config).to(args.device)
    
    # print the number of model parameters
    print('Size of the model:', (sum(p.numel() for p in model.parameters())*4)/(1024**3) , 'GB')
    print('Number of parameters:', sum(p.numel() for p in model.parameters()))
    
    train(train_dataloader, val_dataloader, model, args, wandb_logger=wandb_logger)
    
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    
    # Iterate over the dataset
    for epoch in range(args.num_epochs):
        print(f'Epoch {epoch + 1}')
        total_loss = 0
        for batch in tqdm(train_dataloader):
            inputs, attention_mask, targets = batch['input_ids'], batch['attention_mask'], batch['labels']
            inputs, attention_mask, targets = inputs.to(args.device), attention_mask.to(args.device), targets.to(args.device)
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