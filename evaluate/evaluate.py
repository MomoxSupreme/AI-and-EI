import pandas as pd
from training import train_m5
from Utils.config_reader import read_config
from Utils.stats import *

"""
this file will construct the csv file containing the datas usefull for the evaluations
we use codecarbon to measure environmental impact (notably energy consumption that has also an economic impact)
and we use native pytorch functions to get the params and flops
"""

def main():

    #dict for the datas

    measures = {
        "model_name": ["M5", "CRNN", "SVM"],
        "total_params": [0,0,0],
        "trainable_params": [0,0,0],
        "flops_per_passage": [0,0,0],
        "training_consumption_kwh": [0,0,0],
        "best_val_f1": [0,0,0],
        "test_f1": [0,0,0],
        "test_consumption_kwh_per_passage": [0,0,0]
    }

    measures_df = pd.DataFrame(measures)
    measures_df.to_csv('nom_ficher.csv', index=False)

    pass

if __name__ == "__main__":
    main()