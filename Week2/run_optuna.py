import random
import wandb
import optuna
from transformers import DetrForObjectDetection, DetrFeatureExtractor

from detr_model import run_model

wandb.login(key="50315889c64d6cfeba1b57dc714112418a50e134")

# Simulate training.
epochs = 10

def objective_model_cv(trial):
    params = {
        # 'batch_size': trial.suggest_categorical('batch_size', [8,16,32,64]),  # 8,16,32,64
        # 'img_size': trial.suggest_categorical('img_size', [224]),  # 8,16,32,64,128,224,256
        # 'lr': trial.suggest_categorical('lr', [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3]),  # 0.0001, 0.001, 0.01, 0.1, 0.2, 0.3
        # 'optimizer': trial.suggest_categorical('optimizer', ['adadelta', 'adam', 'sgd', 'RMSprop']),  # adadelta, adam, sgd, RMSprop
        # 'unfroze': trial.suggest_int('unfroze', 10, 30, step=5),

        # 'rot': trial.suggest_categorical('rot', [0, 20]),
        # 'sr': trial.suggest_categorical('sr', [0, 0.2]),
        # 'zr': trial.suggest_categorical('zr', [0, 0.2]),
        # 'hf': trial.suggest_categorical('hf', [0, 0.2]),

        # 'margin': trial.suggest_float('margin', 0.1, 2.0),

        # 'momentum': trial.suggest_float('momentum', 0.95, 0.95),
        # 'dropout': trial.suggest_categorical('dropout', ['0']),
        # 'epochs': trial.suggest_int('epochs', 100, 100),
        # 'output': trial.suggest_int('output', 8, 8),
        'batch_size': trial.suggest_categorical('batch_size', [16]),  # 8,16,32,64
        'img_size': trial.suggest_categorical('img_size', [224]),  # 8,16,32,64,128,224,256
        'lr': trial.suggest_categorical('lr', [0.3]),  # 0.0001, 0.001, 0.01, 0.1, 0.2, 0.3
        'optimizer': trial.suggest_categorical('optimizer', ['adadelta']),  # adadelta, adam, sgd, RMSprop
        'unfroze': trial.suggest_categorical('unfroze', [20]),

        'rot': trial.suggest_categorical('rot', [20]),
        'sr': trial.suggest_categorical('sr', [0]),
        'zr': trial.suggest_categorical('zr', [0.2]),
        'hf': trial.suggest_categorical('hf', [0.2]),

        'margin': trial.suggest_float('margin', 1.0, 1.0),

        'momentum': trial.suggest_float('momentum', 0.95, 0.95),
        'dropout': trial.suggest_categorical('dropout', ['0']),
        'epochs': trial.suggest_int('epochs', 100, 100),
        'output': trial.suggest_categorical('output', [2]),

        # 'loss': trial.suggest_categorical('loss', ['CF', 'NCA', 'CL', 'OCL', 'TR', 'OTR'])
        # 'miner': trial.suggest_categorical('miner', ['beasyhard'])

    }

    print(params)

    config = dict(trial.params)
    config['trial.number'] = trial.number

    execution_name = f'Detr_ft_xavi{str(trial.number)}'

    wandb.init(
        project='Detr_W1_ft',
        entity='mcv_c6g7',
        name=execution_name,
        config=config,
        reinit=True,
    )

    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
    trainer = SiameseTrainer()
    validator = SiameseValidator()
    dataset = SiameseMITDataset # pass type not object - init within the fun
    criterion = ContrastiveLoss(margin=params['margin'])

    # model = TripletNet(BaseNet())
    # trainer = TripletTrainer()
    # validator = TripletValidator()
    # dataset = TripletMITDataset # pass type not object - init within the fun
    # criterion = TripletLoss(margin=params['margin'])

    ratio = run_model(params, model, trainer, validator, dataset, criterion, trial.number)
    return ratio


study = optuna.create_study(direction="maximize", study_name='c6-Week1')
study.optimize(objective_model_cv, n_trials=1)