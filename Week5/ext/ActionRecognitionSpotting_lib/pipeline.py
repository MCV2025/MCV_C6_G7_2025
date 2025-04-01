#!/usr/bin/env python3
"""
Main training script for T-DEED with argparse, wandb, and Optuna integration.
"""

import argparse
import os
import sys
import torch
import random
import numpy as np
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ChainedScheduler, LinearLR, CosineAnnealingLR
import wandb
import optuna
from tabulate import tabulate

from util.io import load_json, store_json
from util.eval_classification import evaluate
from datasets.datasets import get_datasets
from model.model_classification import Model

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True,
                        help="Model name; used to load config JSON (e.g., config/<model>.json)")
    parser.add_argument('--seed', type=int, default=1, help="Random seed")
    parser.add_argument('--optuna', action="store_true", help="Run hyperparameter optimization with Optuna")
    return parser.parse_args()

def update_args(args, config):
    args.frame_dir = config['frame_dir']
    args.save_dir = config['save_dir'] + '/' + args.model  # Append model name.
    args.store_dir = config['save_dir'] + '/' + "splits"
    args.labels_dir = config['labels_dir']
    args.store_mode = config['store_mode']
    args.task = config['task']
    args.batch_size = config['batch_size']
    args.clip_len = config['clip_len']
    args.dataset = config['dataset']
    args.epoch_num_frames = config['epoch_num_frames']
    args.feature_arch = config['feature_arch']
    args.learning_rate = config['learning_rate']
    args.num_classes = config['num_classes']
    args.num_epochs = config['num_epochs']
    args.warm_up_epochs = config['warm_up_epochs']
    args.only_test = config['only_test']
    args.device = config['device']
    args.num_workers = config['num_workers']
    return args

def get_lr_scheduler(args, optimizer, num_steps_per_epoch):
    cosine_epochs = args.num_epochs - args.warm_up_epochs
    print('Using Linear Warmup ({} epochs) + Cosine Annealing LR ({} epochs)'.format(
        args.warm_up_epochs, cosine_epochs))
    return ChainedScheduler([
        LinearLR(optimizer, start_factor=0.01, end_factor=1.0,
                 total_iters=args.warm_up_epochs * num_steps_per_epoch),
        CosineAnnealingLR(optimizer, num_steps_per_epoch * cosine_epochs)
    ])

def run_training(args, trial):
    # Set seeds for reproducibility.
    print('Setting seed to:', args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    execution_name = f'xavi_ft{str(trial.number)}'

    wandb.init(
        project='Week5_ft_final',
        entity='mcv-c6g7',
        name=execution_name,
        config=args, reinit=True)
    wandb.config.update({"store_dir": args.store_dir}, allow_val_change=True)
    wandb.config.update({"save_dir": args.save_dir}, allow_val_change=True)

    ckpt_dir = os.path.join(args.save_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    # Load datasets.
    classes, train_data, val_data, test_data = get_datasets(args)
    if args.store_mode == 'store':
        print('Datasets have been stored correctly! Re-run with store_mode set to "load" in the config JSON.')
        sys.exit('Datasets have been stored. Exiting.')
    else:
        print('Datasets loaded successfully.')

    # Define a worker initialization function for reproducibility.
    def worker_init_fn(worker_id):
        random.seed(worker_id)

    train_loader = DataLoader(
        train_data, shuffle=False, batch_size=args.batch_size,
        pin_memory=True, num_workers=args.num_workers,
        prefetch_factor=(2 if args.num_workers > 0 else None),
        worker_init_fn=worker_init_fn
    )
    val_loader = DataLoader(
        val_data, shuffle=False, batch_size=args.batch_size,
        pin_memory=True, num_workers=args.num_workers,
        prefetch_factor=(2 if args.num_workers > 0 else None),
        worker_init_fn=worker_init_fn
    )

    # Initialize the model.
    model = Model(args=args)
    optimizer, scaler = model.get_optimizer({'lr': args.learning_rate})

    best_metric = float('inf')
    losses = []

    if not args.only_test:
        num_steps_per_epoch = len(train_loader)
        lr_scheduler = get_lr_scheduler(args, optimizer, num_steps_per_epoch)

        print('START TRAINING EPOCHS')
        for epoch in range(args.num_epochs):
            # Train one epoch.
            train_loss = model.epoch(train_loader, optimizer, scaler, lr_scheduler=lr_scheduler)
            # Validate.
            val_loss = model.epoch(val_loader)
            wandb.log({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
            print('[Epoch {}] Train loss: {:.5f} | Val loss: {:.5f}'.format(epoch, train_loss, val_loss))
            if val_loss < best_metric:
                best_metric = val_loss
                torch.save(model.state_dict(), os.path.join(ckpt_dir, 'checkpoint_best.pt'))
                print('New best metric at epoch {}: {:.5f}'.format(epoch, val_loss))
            losses.append({'epoch': epoch, 'train': train_loss, 'val': val_loss})
            store_json(os.path.join(args.save_dir, 'loss.json'), losses, pretty=True)
    else:
        print("Only test mode; skipping training.")

    print('START INFERENCE')
    model.load(torch.load(os.path.join(ckpt_dir, 'checkpoint_best.pt')))
    ap_score = evaluate(model, test_data)
    table = []
    for i, class_name in enumerate(classes.keys()):
        table.append([class_name, f"{ap_score[i]*100:.2f}"])
    print(tabulate(table, headers=["Class", "Average Precision"], tablefmt="grid"))
    print(tabulate([["Average", f"{np.mean(ap_score)*100:.2f}"]],
                   headers=["", "Average Precision"], tablefmt="grid"))
    wandb.finish()
    print('FINISHED TRAINING AND INFERENCE')
    return best_metric

def main(args):
    # Load the configuration JSON based on the provided model name.
    config_path = os.path.join('config', args.model + '.json')
    config = load_json(config_path)
    args = update_args(args, config)

    if args.optuna:
        def objective(trial):
            # For each trial, reload the configuration and update args.
            config_trial = load_json(config_path)
            new_args = update_args(argparse.Namespace(model=args.model, seed=args.seed, optuna=False), config_trial)
            # Override hyperparameters using Optuna suggestions.
            new_args.batch_size = trial.suggest_categorical("batch_size", [2, 4, 8, 16])
            new_args.stride = trial.suggest_categorical("stride", [2, 4, 6])
            new_args.learning_rate = trial.suggest_categorical("learning_rate", [.0008, 5e-4, 1e-4, 1e-3, 1e-2]) 
            new_args.num_epochs = trial.suggest_categorical("num_epochs", [15, 20, 25, 30, 35])
            new_args.warm_up_epochs = trial.suggest_categorical("warm_up_epochs", [1, 3, 5])

    
            # Run training for this trial.
            metric = run_training(new_args, trial)
            return metric

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=50)
        print("Best trial:")
        trial = study.best_trial
        print(f"  Value: {trial.value}")
        print("  Params:")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")
    else:
        run_training(args)

if __name__ == '__main__':
    args = get_args()
    main(args)

# import os
# import sys
# import json
# import torch
# import random
# import numpy as np
# import optuna
# import wandb
# from torch.utils.data import DataLoader
# from torch.optim.lr_scheduler import ChainedScheduler, LinearLR, CosineAnnealingLR
# from tabulate import tabulate

# # Local imports (adjust the paths as needed)
# from util.io import load_json, store_json
# from util.eval_classification import evaluate
# from datasets.datasets import get_datasets
# from model.model_classification import Model

# wandb.login(key="50315889c64d6cfeba1b57dc714112418a50e134")

# def train_and_evaluate_model(config, trial):
#     """
#     Trains the model with the given configuration and pooling type, logs each epoch with wandb,
#     and returns the best validation metric (e.g., validation loss) as the objective.
#     """
#     # Initialize wandb with the current configuration.
#     execution_name = f'xavi_ft{str(trial.number)}'

#     wandb.init(
#         project='Week5_ft',
#         entity='mcv-c6g7',
#         name=execution_name,
#         config=config, reinit=True)
    
#     # Optionally, get the wandb config (which now contains trial values as well).
#     config = wandb.config

#     # Set seeds for reproducibility.
#     print('Setting seed to:', config.seed)
#     torch.manual_seed(config.seed)
#     np.random.seed(config.seed)
#     random.seed(config.seed)

#     # # If a model name is provided, update the save directory.
#     # if hasattr(config, "model"):
#     #     new_save_dir = os.path.join(config.save_dir, config.model)
#     #     wandb.config.update({"save_dir": new_save_dir}, allow_val_change=True)
#     #     config.save_dir = new_save_dir

#     # Get datasets.
#     classes, train_data, val_data, test_data = get_datasets(config)
#     if config.store_mode == 'store':
#         print('Datasets have been stored correctly! Re-run changing "store_mode" to "load" in the config JSON.')
#         sys.exit('Datasets have been stored! Stop training here and rerun with load mode.')
#     else:
#         print('Datasets have been loaded from previous versions correctly!')

#     # Worker initialization for reproducibility.
#     def worker_init_fn(worker_id):
#         random.seed(worker_id)

#     # Create DataLoaders.
#     train_loader = DataLoader(
#         train_data, shuffle=False, batch_size=config.batch_size,
#         pin_memory=True, num_workers=config.num_workers,
#         prefetch_factor=(2 if config.num_workers > 0 else None),
#         worker_init_fn=worker_init_fn
#     )
#     val_loader = DataLoader(
#         val_data, shuffle=False, batch_size=config.batch_size,
#         pin_memory=True, num_workers=config.num_workers,
#         prefetch_factor=(2 if config.num_workers > 0 else None),
#         worker_init_fn=worker_init_fn
#     )

#     # Instantiate the model.
#     model = Model(args=config)

#     # Create optimizer (and scaler if using mixed precision).
#     optimizer, scaler = model.get_optimizer({'lr': config.learning_rate})

#     best_criterion = float('inf')
#     losses = []

#     # Training and evaluation.
#     if not config.only_test:
#         num_steps_per_epoch = len(train_loader)
#         cosine_epochs = config.num_epochs - config.warm_up_epochs
#         print('Using Linear Warmup ({} epochs) + Cosine Annealing LR ({} epochs)'.format(
#             config.warm_up_epochs, cosine_epochs))
#         lr_scheduler = ChainedScheduler([
#             LinearLR(optimizer, start_factor=0.01, end_factor=1.0,
#                      total_iters=config.warm_up_epochs * num_steps_per_epoch),
#             CosineAnnealingLR(optimizer, num_steps_per_epoch * cosine_epochs)
#         ])

#         print('START TRAINING EPOCHS')
#         for epoch in range(config.num_epochs):
#             # Train for one epoch.
#             train_loss = model.epoch(train_loader, optimizer, scaler, lr_scheduler=lr_scheduler)
#             # Evaluate on validation data.
#             val_loss = model.epoch(val_loader)
            
#             # Log the metrics to wandb.
#             wandb.log({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
#             print('[Epoch {}] Train loss: {:0.5f} | Val loss: {:0.5f}'.format(epoch, train_loss, val_loss))
            
#             if val_loss < best_criterion:
#                 best_criterion = val_loss
#                 # Save the best model checkpoint.
#                 torch.save(model.state_dict(), os.path.join(ckpt_dir, 'checkpoint_best.pt'))
#                 print('New best validation metric found at epoch {}: {:0.5f}'.format(epoch, val_loss))
            
#             losses.append({'epoch': epoch, 'train': train_loss, 'val': val_loss})
#             os.makedirs(config.save_dir, exist_ok=True)
#             store_json(os.path.join(config.save_dir, 'loss.json'), losses, pretty=True)
#     else:
#         print("Only test mode; skipping training.")

#     print('START INFERENCE')
#     # Load the best model checkpoint.
#     model.load(torch.load(os.path.join(ckpt_dir, 'checkpoint_best.pt')))

#     # Run evaluation on the test set.
#     ap_score = evaluate(model, test_data)
#     table = []
#     for i, class_name in enumerate(classes.keys()):
#         table.append([class_name, f"{ap_score[i]*100:.2f}"])
#     print(tabulate(table, headers=["Class", "Average Precision"], tablefmt="grid"))
#     print(tabulate([["Average", f"{np.mean(ap_score)*100:.2f}"]],
#                    headers=["", "Average Precision"], tablefmt="grid"))
    
#     print('FINISHED TRAINING AND INFERENCE')
#     wandb.finish()

#     return best_criterion

# # Define the Optuna objective function.
# def objective(trial):
#     # Load your base configuration from JSON.
#     with open("config/config.json", "r") as f:
#         base_config = json.load(f)

#     # Update the hyperparameters with values suggested by Optuna.
#     base_config["batch_size"] = trial.suggest_categorical("batch_size", [2, 4, 8, 16])
#     base_config["stride"] = trial.suggest_categorical("stride", [2, 4, 5, 8, 10])
#     base_config["learning_rate"] = trial.suggest_categorical("learning_rate", [.0008, 5e-4, 1e-4, 1e-3, 1e-2]) 
#     base_config["num_epochs"] = trial.suggest_categorical("num_epochs", [15, 20, 25, 30, 35])
#     base_config["warm_up_epochs"] = trial.suggest_categorical("warm_up_epochs", [1, 3, 5])
    
#     # Ensure that some necessary parameters are present.
#     if "seed" not in base_config:
#         base_config["seed"] = 1
#     if "model" not in base_config:
#         base_config["model"] = "default_model_name"
#     if "only_test" not in base_config:
#         base_config["only_test"] = False

#     # Call the training and evaluation function.
#     metric = train_and_evaluate_model(base_config, trial)
#     return metric

# if __name__ == "__main__":
#     study = optuna.create_study(direction="minimize")
#     study.optimize(objective, n_trials=50)

#     print("Best trial:")
#     trial = study.best_trial
#     print(f"  Value: {trial.value}")
#     print("  Params:")
#     for key, value in trial.params.items():
#         print(f"    {key}: {value}")
