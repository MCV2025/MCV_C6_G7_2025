
from typing import Union
import torch
import yaml
import numpy as np

from transformers import PreTrainedModel
from torch.utils.data import DataLoader
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
    # 3) Model & Oprtimizer
    # -------------------------
    model.config.num_labels = num_labels
    model.to(device)
    optimizer = get_optimizer(params, model)