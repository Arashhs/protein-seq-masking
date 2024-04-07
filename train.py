import torch
from torch.utils.data import DataLoader, TensorDataset
from data.dataloader import AminoAcidTokenizer, ProteinDataset
from tqdm import tqdm
import pandas as pd
from datetime import datetime 
from transformers import BertForMaskedLM, BertConfig
import json
from pathlib import Path

from args import parse_args
from data.utils import load_train_data, load_test_data
from utils.logger import WandbLogger



class LossCheckpointer:

    def __init__(self, run_id) -> None:
        self.best_loss = None
        self.run_id = run_id

        self.best_model_path = Path('checkpoints') / run_id
        self.best_model_path.mkdir(parents=True, exist_ok=True)
        self.best_model_path = self.best_model_path / 'best_model.pt'

    def _save_checkpoint(self, model, optimizer, epoch, loss):
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, self.best_model_path )
        
    def load_best_model_checkpoint(self, model):
        checkpoint = torch.load(self.best_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model
    
    def epoch_done(self, model, optimizer, epoch, loss):
        # save the model if the loss is the best so far or the first epoch
        if self.best_loss is None or loss < self.best_loss:
            self.best_loss = loss
            self._save_checkpoint(model, optimizer, epoch, loss)
            print('Checkpoint saved at epoch', epoch)



def train(train_dataloader, val_dataloader, model, args, wandb_logger=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    model.float()
    
    loss_checkpointer = LossCheckpointer(args.run_name)
    
    # put the model in training mode
    wandb_logger.watch_model(model)
    
    step = 0
    print(f'Start training on {len(train_dataloader.dataset)} samples.')
    
    # Start time of the entire training
    training_start_time = datetime.now()
    
    for epoch in range(args.num_epochs):
        start_time = datetime.now()  # Start time of the epoch
        
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
            
            
        end_time = datetime.now()  # End time of the epoch
        epoch_duration = (end_time - start_time).total_seconds()
        total_training_duration = (end_time - training_start_time).total_seconds()
        wandb_logger.log('train/epoch_duration', epoch_duration, epoch)
        wandb_logger.log('train/total_training_duration', total_training_duration, epoch)
        
        # Log memory utilization here
        if torch.cuda.is_available():
            current_memory_gb = torch.cuda.memory_allocated() / (1024 ** 3)  # Convert bytes to GB
            peak_memory_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)  # Convert bytes to GB
            wandb_logger.log('cuda/current_memory_gb', current_memory_gb, step)
            wandb_logger.log('cuda/peak_memory_gb', peak_memory_gb, step)
            print(f"Epoch {epoch+1}: Current GPU Memory Utilization (GB): {current_memory_gb}, Peak GPU Memory Utilization (GB): {peak_memory_gb}")
            
        print('Training done, validating...')
        
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_dataloader):
                inputs, attention_mask, targets = batch
                inputs, attention_mask, targets = inputs.to(args.device), attention_mask.to(args.device), targets.to(args.device)
                
                outputs = model(inputs, attention_mask=attention_mask, labels=targets)
                loss = outputs.loss
                total_val_loss += loss.item()
                
        validation_loss = total_val_loss/ len(val_dataloader)

        wandb_logger.log('val/loss', validation_loss, step)
        print(f"Epoch {epoch+1}/{args.num_epochs}, Loss (train): {total_loss/ len(train_dataloader)}, Loss (val): {validation_loss}")
        
        loss_checkpointer.epoch_done(model, optimizer, epoch, validation_loss)
        
    print('Finished training')
    wandb_logger.log('train/total_training_time', total_training_duration, 0)
    print(f"Total training time: {total_training_duration} seconds")
    
    return loss_checkpointer.load_best_model_checkpoint(model)



def test(test_dataloader, model, args, wandb_logger=None):
    model.eval()
    total_test_loss = 0
    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            inputs, attention_mask, targets = batch
            inputs, attention_mask, targets = inputs.to(args.device), attention_mask.to(args.device), targets.to(args.device)

            outputs = model(inputs, attention_mask=attention_mask, labels=targets)
            loss = outputs.loss
            total_test_loss += loss.item()

    test_loss = total_test_loss/ len(test_dataloader)
    print(f"Test Loss: {test_loss}")
    wandb_logger.log('test/loss', test_loss, 0)
    




def main(args):
    args.run_name = f'{args.run_name}_maxSeq{args.max_seq_len}-{str(datetime.now().strftime("%Y-%m-%d %H:%M"))}'
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
    train_dataset = ProteinDataset(train_sequences, tokenizer, mask_probability=args.mask_probability)
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
    print(f'Number of parameters: {sum(p.numel() for p in model.parameters()):,}')
    
    try:    
        best_model = train(train_dataloader, val_dataloader, model, args, wandb_logger=wandb_logger)
    except KeyboardInterrupt:
        loss_checkpointer = LossCheckpointer(args.run_name)
        best_model = loss_checkpointer.load_best_model_checkpoint(model)
        print('-' * 80)
        print('Exiting training early.')
    print('Evaluation on test set...')
    test(test_dataloader, best_model, args, wandb_logger=wandb_logger)

    




if __name__ == '__main__':
    args = parse_args()
    main(args)