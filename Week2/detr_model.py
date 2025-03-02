
from typing import Union
import yaml

import numpy as np
from transformers import PreTrainedModel

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from utils import data_augmentation, get_optimizer

# Load YAML file
with open('config.yml', 'r') as file:
    data = yaml.safe_load(file)

# Accessing variables from the YAML data
dataset_dir = data['DATASET_DIR']
dataset_train = data['DATASET_TRAIN']
dataset_test = data['DATASET_TEST']

# Define a random seed for torch, numpy and cuda. This will ensure reproducibility
# and help obtain same results (wights inits, data splits, etc.). 
# This way we delete the posibility of a run obtaining better results by random chance
# rather than by hyperparameter tweaking.
seed = 49
torch.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed_all(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Selected device: --> {device}")


def run_model(
        params: dict[str, Union[int, float, str]], 
        model: PreTrainedModel,
        dataset, 
        trial_number: int, 
        num_labels: int=1) -> None:
    
    model_name = f""


    #--------------------------
    # 1) Call datasets
    #--------------------------
    train_dataset = ""
    validation_ratio = 0.2
    val_dataset = ""
    test_dataset = ""
    
    # -------------------------
    # 2) Hyperparameter Search
    # -------------------------
    learning_rate=params['lr']
    num_epochs = params['epochs']
    batch_size = params['batch_size']

    # -------------------------
    # 3) DataLoaders
    # -------------------------
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_fn)
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=collate_fn)
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=collate_fn)
    
    # -------------------------
    # 4) Model & Oprtimizer
    # -------------------------
    model.config.num_labels = num_labels
    model.to(device)
    optimizer = get_optimizer(params, model)

    # -------------------------
    # 5) Early stopping and patience parameters
    # -------------------------
    patience = 200
    min_delta = 0.001
    best_val_loss = np.Inf
    current_patience = 0

    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.1, verbose=True)

    for epoch in range(num_epochs):
        train_loss, train_accuracy = trainer.train(model, train_loader, criterion, optimizer, params, device)
        val_loss, val_accuracy = validator.validation(model, val_loader, criterion, params, device)
        # Adjust learning rate based on validation loss
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            current_patience = 0

            # Save the best model
            print("Best model. Saving weights")
            torch.save(model.state_dict(), model_name)
        else:
            current_patience += 1
            if current_patience > patience:
                print("Early stopping.")
                break

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss/Accuracy: {train_loss:.4f}/{train_accuracy:.4f} Val Loss/Accuracy: {val_loss:.4f}/{val_accuracy:.4f}')

        #Add info to wandb
        wandb.log({
            'Train Loss': train_loss,
            'Validation Loss': val_loss,
            'Train Accuracy': train_accuracy,
            'Validation Accuracy': val_accuracy,
        })
