# AI-and-EI-Scratch

This project provides a modular codebase to train, evaluate, and compare multiple neural network architectures on a single benchmark task—audio classification using the DESED (Domestic Environment Sound Event Detection) dataset. Beyond standard predictive performance metrics, this tool focuses on Green AI by tracking and comparing the environmental footprint (computational cost, power consumption, energy per gflops...) of each model throughout the training and inference phases. 

## Project structure

```text
AI-and-EI-Scratch/
├── Models/
│   └── models.py           # implements the various nn models in pytorch from modular components
├── Utils/
│   ├── config_reader.py    # Reading and parsing of config.ini
│   ├── datasets.py         # Data preprocessing and formatting for each model
│   ├── metrics.py          # Implements the various metrics for evaluation
│   ├── scheduler.py        
│   ├── stats.py            # Utils functions to get some stats on the models (example : number of parameters)
│   └── visualize.py        # Utils functions to visualize the results obtained
├── evaluate/
│   └── evaluate.py         # Main script to launch in order to get the graphics and results
├── config.ini              # Hyperparameters tuning (epochs, learning rate, patience...). Change this file to change the hyperparameters
├── config.py               # Generates the config.ini files. Launch it to retrieve the default config.
├── requirements.txt        # Python dependances used in the project
├── test.py                 # Does the inference phase on the models
├── training.py             # Trains the different models
└── LICENSE                 
