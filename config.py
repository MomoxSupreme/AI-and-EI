import configparser

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
        'LOSS': 'BCELogits'
    }

    config['Hyperparameters'] = {
        'SEED': '2026',
        'TARGET_RATE': '16000',
        'LEARNING_RATE': '0.001',
        'N_CLASSES': '10',
        'N_BEST_RAG': '3',
        'FRAMES_PER_SECON': '50',
        'AUDIO_SECONDS': '10',
    }

    config['Scheduler'] = {
        'SCHEDULER': 'Sbzkz'
    }

    # Write the configuration to a file
    with open('config.ini', 'w') as configfile:
        config.write(configfile)

if __name__ == "__main__":
    create_config()