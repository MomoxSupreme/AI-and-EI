import configparser
"""
run create_config() to return to the default config, else you should prefere to modify config.ini directly
"""

def create_config():
    config = configparser.ConfigParser()
    # Add sections and key-value pairs
    config['General'] = {
        'debug': 'True',
        'log_level': 'info'
    }

    config['Hardware'] = {
        'DEVICE': 'cuda',
    }

    config['Optim and loss'] = {
        'OPTIMIZER': 'Adam',
        'LOSS': 'BCEWithLogitsLoss'
    }

    config['Hyperparameters'] = {
        'SEED': '2026',
        'TARGET_RATE': '16000',
        'LEARNING_RATE': '0.001',
        'N_CLASSES': '10',
        'N_BEST_RAG': '3',
        'FRAMES_PER_SECOND': '50',
        'AUDIO_SECONDS': '10',
    }

    config['Scheduler'] = {
        'SCHEDULER': 'Sbzkz'
    }

    config['Datas'] = {
        'raw_train_datas_path': "./Data/raw/audio/train",
        'raw_test_datas_path': "./Data/raw/audio/eval",
        'train_metadata_path': "./Data/raw/metadata/train/public_weak.tsv",
        'eval_metadata_path': "./Data/raw/metadata/eval/public_weak.tsv"
    }

    # Write the configuration to a file
    with open('config.ini', 'w') as configfile:
        config.write(configfile)

if __name__ == "__main__":
    create_config()