
import configparser
import torch
import torch.optim as optim
import torch.nn as nn

OPTIMIZERS = {
    "Adam": optim.Adam,
    "AdamW": optim.AdamW,
    "SGD": optim.SGD,
    "RMSprop": optim.RMSprop
}

LOSS = {
    "CrossEntropyLoss": nn.CrossEntropyLoss,
    "MSELoss": nn.MSELoss,
    "L1Loss": nn.L1Loss,
    "BCEWithLogitsLoss": nn.BCEWithLogitsLoss,
    "NLLLoss": nn.NLLLoss
}


def read_config():
    config = configparser.ConfigParser()
    config.read('config.ini')

    # Access values from the configuration file

    # General
    debug_mode = config.getboolean('General', 'debug')
    log_level = config.get('General', 'log_level')

    # Hardware (GPU or CPU)
    device = config.get('Hardware', 'device')

    if device.lower() == 'cuda':
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print("\033[92m[INFO] CUDA is available. GPU enabled.\033[0m")
        else:
            device = torch.device('cpu')
            print("\033[91m[WARNING] CUDA is not available. Device switched to CPU.\033[0m")
    else:
        device = torch.device('cpu')
        print("\033[92m[INFO] CPU selected.\033[0m")

    # Optimizer and loss
    optimizer_name = config.get('Optim and loss', 'optimizer')
    loss_name = config.get('Optim and loss', 'loss')

    optimizer = OPTIMIZERS[optimizer_name]
    loss = LOSS[loss_name]

    # Hyperparameters
    seed = config.getint('Hyperparameters', 'seed')
    target_rate = config.getint('Hyperparameters', 'target_rate')
    learning_rate = config.getfloat('Hyperparameters', 'learning_rate')
    n_classes = config.getint('Hyperparameters', 'n_classes')
    n_best_rag = config.getint('Hyperparameters', 'n_best_rag')
    frames_per_second = config.getint('Hyperparameters', 'frames_per_secon')
    audio_seconds = config.getint('Hyperparameters', 'audio_seconds')

    raw_train_datas_path = config.get('Datas', 'raw_train_datas_path')
    raw_test_datas_path = config.get('Datas', 'raw_test_datas_path')
    train_metadata_path = config.get('Datas', 'train_metadata_path')
    eval_metadata_path = config.get('Datas', 'eval_metadata_path')

    # Scheduler
    scheduler = config.get('Scheduler', 'scheduler')

    # Store retrieved values in a dictionary
    config_values = {
        'debug_mode': debug_mode,
        'log_level': log_level,

        'device': device,

        'optimizer': optimizer,
        'loss': loss,

        'seed': seed,
        'target_rate': target_rate,
        'learning_rate': learning_rate,
        'n_classes': n_classes,
        'n_best_rag': n_best_rag,
        'frames_per_second': frames_per_second,
        'audio_seconds': audio_seconds,

        'scheduler': scheduler,
        
        'raw_train_datas_path': raw_train_datas_path,
        'raw_test_datas_path': raw_test_datas_path,
        'train_metadata_path': train_metadata_path,
        'test_metadata_path': eval_metadata_path
    }

    return config_values
