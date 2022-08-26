import os
import argparse
import time
import csv
import sys
import json
import random
import numpy as np
import pprint
import yaml

import torch
import torch.multiprocessing as mp

import ray
from ray import tune

#print("Ver:", ray.__version__)

from matdeeplearn import models, process, training

from sklearn.metrics import r2_score
import pandas as pd
from matplotlib import pyplot as plt
import scipy as sp
import math
import scipy.stats as stats
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

import warnings

import os
import shutil

def fxn():
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()

################################################################################
#
################################################################################
#  MatDeepLearn code
################################################################################
#
################################################################################
def main():
    start_time = time.time()
    #print("Starting...")
    #print("GPU is available:",torch.cuda.is_available(), ", Quantity: ", torch.cuda.device_count(),)

    parser = argparse.ArgumentParser(description="MatDeepLearn inputs")
    ###Job arguments
    parser.add_argument(
        "--config_path",
        default="config.yml",
        type=str,
        help="Location of config file (default: config.yml)",
    )
    parser.add_argument(
        "--run_mode",
        default=None,
        type=str,
        help="run modes: Training, Predict, Repeat, CV, Hyperparameter, Ensemble, Analysis",
    )
    parser.add_argument(
        "--job_name",
        default=None,
        type=str,
        help="name of your job and output files/folders",
    )
    parser.add_argument(
        "--model",
        default=None,
        type=str,
        help="CGCNN_demo, MPNN_demo, SchNet_demo, MEGNet_demo, GCN_demo, SOAP_demo, SM_demo",
    )
    parser.add_argument(
        "--seed",
        default=None,
        type=int,
        help="seed for data split, 0=random",
    )
    parser.add_argument(
        "--model_path",
        default=None,
        type=str,
        help="path of the model .pth file",
    )
    parser.add_argument(
        "--save_model",
        default=None,
        type=str,
        help="Save model",
    )
    parser.add_argument(
        "--load_model",
        default=None,
        type=str,
        help="Load model",
    )
    parser.add_argument(
        "--write_output",
        default=None,
        type=str,
        help="Write outputs to csv",
    )
    parser.add_argument(
        "--parallel",
        default=None,
        type=str,
        help="Use parallel mode (ddp) if available",
    )
    parser.add_argument(
        "--reprocess",
        default=None,
        type=str,
        help="Reprocess data since last run",
    )
    ###Processing arguments
    parser.add_argument(
        "--data_path",
        default=None,
        type=str,
        help="Location of data containing structures (json or any other valid format) and accompanying files",
    )
    parser.add_argument("--format", default=None, type=str, help="format of input data")
    ###Training arguments
    parser.add_argument("--train_ratio", default=None, type=float, help="train ratio")
    parser.add_argument(
        "--val_ratio", default=None, type=float, help="validation ratio"
    )
    parser.add_argument("--test_ratio", default=None, type=float, help="test ratio")
    parser.add_argument(
        "--verbosity", default=None, type=int, help="prints errors every x epochs"
    )
    parser.add_argument(
        "--target_index",
        default=None,
        type=int,
        help="which column to use as target property in the target file",
    )
    ###Model arguments
    parser.add_argument(
        "--epochs",
        default=None,
        type=int,
        help="number of total epochs to run",
    )
    parser.add_argument("--batch_size", default=None, type=int, help="batch size")
    parser.add_argument("--lr", default=None, type=float, help="learning rate")

    ##Get arguments from command line
    args = parser.parse_args(sys.argv[1:])

    ##Open provided config file
    assert os.path.exists(args.config_path), (
        "Config file not found in " + args.config_path
    )
    with open(args.config_path, "r") as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.FullLoader)

    ##Update config values from command line
    if args.run_mode != None:
        config["Job"]["run_mode"] = args.run_mode
    run_mode = config["Job"].get("run_mode")
    config["Job"] = config["Job"].get(run_mode)
    if config["Job"] == None:
        print("Invalid run mode")
        sys.exit()

    if args.job_name != None:
        config["Job"]["job_name"] = args.job_name
    if args.model != None:
        config["Job"]["model"] = args.model
    if args.seed != None:
        config["Job"]["seed"] = args.seed
    if args.model_path != None:
        config["Job"]["model_path"] = args.model_path
    if args.load_model != None:
        config["Job"]["load_model"] = args.load_model
    if args.save_model != None:
        config["Job"]["save_model"] = args.save_model
    if args.write_output != None:
        config["Job"]["write_output"] = args.write_output
    if args.parallel != None:
        config["Job"]["parallel"] = args.parallel
    if args.reprocess != None:
        config["Job"]["reprocess"] = args.reprocess

    if args.data_path != None:
        config["Processing"]["data_path"] = args.data_path
    if args.format != None:
        config["Processing"]["data_format"] = args.format

    if args.train_ratio != None:
        config["Training"]["train_ratio"] = args.train_ratio
    if args.val_ratio != None:
        config["Training"]["val_ratio"] = args.val_ratio
    if args.test_ratio != None:
        config["Training"]["test_ratio"] = args.test_ratio
    if args.verbosity != None:
        config["Training"]["verbosity"] = args.verbosity
    if args.target_index != None:
        config["Training"]["target_index"] = args.target_index

    for key in config["Models"]:
        if args.epochs != None:
            config["Models"][key]["epochs"] = args.epochs
        if args.batch_size != None:
            config["Models"][key]["batch_size"] = args.batch_size
        if args.lr != None:
            config["Models"][key]["lr"] = args.lr

    if run_mode == "Predict":
        config["Models"] = {}
    else:
        config["Models"] = config["Models"].get(config["Job"]["model"])

    if config["Job"]["seed"] == 0:
        config["Job"]["seed"] = np.random.randint(1, 1e6)

    ################################################################################
    #  Begin processing
    ################################################################################

    if run_mode != "Hyperparameter":

        process_start_time = time.time()

        dataset = process.get_dataset(
            config["Processing"]["data_path"],
            config["Training"]["target_index"],
            config["Job"]["reprocess"],
            config["Processing"],
        )

    ################################################################################
    #  Training begins
    ################################################################################

    ##Regular training
    if run_mode == "Training":
        world_size = torch.cuda.device_count()
        if world_size == 0:
            training.train_regular(
                "cpu",
                world_size,
                config["Processing"]["data_path"],
                config["Job"],
                config["Training"],
                config["Models"],
            )

        elif world_size > 0:
            if config["Job"]["parallel"] == "True":
                mp.spawn(
                    training.train_regular,
                    args=(
                        world_size,
                        config["Processing"]["data_path"],
                        config["Job"],
                        config["Training"],
                        config["Models"],
                    ),
                    nprocs=world_size,
                    join=True,
                )
            if config["Job"]["parallel"] == "False":
                training.train_regular(
                    "cuda",
                    world_size,
                    config["Processing"]["data_path"],
                    config["Job"],
                    config["Training"],
                    config["Models"],
                )

    ##Predicting from a trained model
    elif run_mode == "Predict":

        train_error = training.predict(
            dataset, config["Training"]["loss"], config["Job"]
        )

    ##Hyperparameter optimization
    elif run_mode == "Hyperparameter":

        #print("Starting hyperparameter optimization")
        #print("running for " + str(config["Models"]["epochs"])+ " epochs"+ " on "+ str(config["Job"]["model"])+ " model")

        ##Reprocess here if not reprocessing between trials
        if config["Job"]["reprocess"] == "False":
            process_start_time = time.time()

            dataset = process.get_dataset(
                config["Processing"]["data_path"],
                config["Training"]["target_index"],
                config["Job"]["reprocess"],
                config["Processing"],
            )

            #print("Dataset used:", dataset)
            #print(dataset[0])

            if config["Training"]["target_index"] == -1:
                config["Models"]["output_dim"] = len(dataset[0].y[0])
            # print(len(dataset[0].y))

            print(
                #"--- %s seconds for processing ---" % (time.time() - process_start_time)
            )

        ##Set up search space for each model; these can subject to change
        hyper_args = {}
        dim1 = [x * 10 for x in range(1, 20)]
        dim2 = [x * 10 for x in range(1, 20)]
        dim3 = [x * 10 for x in range(1, 20)]
        batch = [x * 10 for x in range(1, 20)]
        hyper_args["SchNet_demo"] = {
            "dim1": tune.choice(dim1),
            "dim2": tune.choice(dim2),
            "dim3": tune.choice(dim3),
            "gnn_count": tune.choice([1, 2, 3, 4, 5, 6, 7, 8, 9]),
            "post_fc_count": tune.choice([1, 2, 3, 4, 5, 6]),
            "pool": tune.choice(
                ["global_mean_pool", "global_add_pool", "global_max_pool", "set2set"]
            ),
            "lr": tune.loguniform(1e-4, 0.05),
            "batch_size": tune.choice(batch),
            "cutoff": config["Processing"]["graph_max_radius"],
        }
        hyper_args["CGCNN_demo"] = {
            "dim1": tune.choice(dim1),
            "dim2": tune.choice(dim2),
            "gnn_count": tune.choice([1, 2, 3, 4, 5, 6, 7, 8, 9]),
            "post_fc_count": tune.choice([1, 2, 3, 4, 5, 6]),
            "pool": tune.choice(
                ["global_mean_pool", "global_add_pool", "global_max_pool", "set2set"]
            ),
            "lr": tune.loguniform(1e-4, 0.05),
            "batch_size": tune.choice(batch),
        }
        hyper_args["MPNN_demo"] = {
            "dim1": tune.choice(dim1),
            "dim2": tune.choice(dim2),
            "dim3": tune.choice(dim3),
            "gnn_count": tune.choice([1, 2, 3, 4, 5, 6, 7, 8, 9]),
            "post_fc_count": tune.choice([1, 2, 3, 4, 5, 6]),
            "pool": tune.choice(
                ["global_mean_pool", "global_add_pool", "global_max_pool", "set2set"]
            ),
            "lr": tune.loguniform(1e-4, 0.05),
            "batch_size": tune.choice(batch),
        }
        hyper_args["GCN_demo"] = {
            "dim1": tune.choice(dim1),
            "dim2": tune.choice(dim2),
            "gnn_count": tune.choice([1, 2, 3, 4, 5, 6, 7, 8, 9]),
            "post_fc_count": tune.choice([1, 2, 3, 4, 5, 6]),
            "pool": tune.choice(
                ["global_mean_pool", "global_add_pool", "global_max_pool", "set2set"]
            ),
            "lr": tune.loguniform(1e-4, 0.05),
            "batch_size": tune.choice(batch),
        }
        
        ##Run tune setup and trials
        best_trial = training.tune_setup(
            hyper_args[config["Job"]["model"]],
            config["Job"],
            config["Processing"],
            config["Training"],
            config["Models"],
        )

        ##Write hyperparameters to file
        hyperparameters = best_trial.config["hyper_args"]
        hyperparameters = {
            k: round(v, 6) if isinstance(v, float) else v
            for k, v in hyperparameters.items()
        }
        with open(
            config["Job"]["job_name"] + "_optimized_hyperparameters.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(hyperparameters, f, ensure_ascii=False, indent=4)
    else:
        print("No valid mode selected, try again")

def plot():
    df = pd.read_csv('plot.csv')
    plot = df.to_numpy()
    epochs = range(1, len(plot[:,0]) + 1)
    plt.plot(epochs, plot[:,0], 'r', label='Training Error')
    plt.plot(epochs, plot[:, 1], 'b', label='validation Error')
    plt.title('Training and Validation Error')
    plt.xlabel('Epochs')
    plt.ylabel('Error')
    plt.legend()
    plt.savefig('C:/Users/Nick/Documents/MatDeepLearn/error.png')

mode = [0, 0]

if mode[0] == 0:
    n = 200

    r2_test, r2_val, r2_train = 0.000, 0.000, 0.000
    MAE_test, MAE_val, MAE_train = 5.000, 5.000, 5.000
    RMSE_test, RMSE_val, RMSE_train = 5.000, 5.000, 5.000

    for i in range(n):
        print('\nIteration #', i+1, '\n')
        if __name__ == "__main__":
            main()
        
        df = pd.read_csv('C:/Users/Nick/Documents/MatDeepLearn/test_test_outputs.csv')
        df = df.drop(['ids'], axis = 1)
        df = df.dropna()
        df = df.dropna()
        y_test = df['target']
        y_pred = df['prediction']

        df_train = pd.read_csv('C:/Users/Nick/Documents/MatDeepLearn/test_train_outputs.csv')
        df_train = df_train.drop(['ids'], axis = 1)
        df_train = df_train.dropna()
        y_train = df_train['target']
        y_pred_train = df_train['prediction']

        df_valid = pd.read_csv('C:/Users/Nick/Documents/MatDeepLearn/test_val_outputs.csv')
        df_valid = df_valid.drop(['ids'], axis = 1)
        df_valid = df_valid.dropna()
        y_valid = df_valid['target']
        y_pred_valid = df_valid['prediction']
        
        if mode[1] == 0:
            if((r2_score(y_test, y_pred) > 0 and r2_score(y_valid, y_pred_valid) > 0 and r2_score(y_train, y_pred_train) > 0) and (mean_absolute_error(y_test, y_pred) + 0.5*mean_absolute_error(y_valid, y_pred_valid) + 0.8*mean_absolute_error(y_train, y_pred_train)) < (MAE_test + 0.5*MAE_val + 0.8*MAE_train)):
                r2_test, r2_val, r2_train = r2_score(y_test, y_pred), r2_score(y_valid, y_pred_valid), r2_score(y_train, y_pred_train)
                MAE_test, MAE_val, MAE_train = mean_absolute_error(y_test, y_pred), mean_absolute_error(y_valid, y_pred_valid), mean_absolute_error(y_train, y_pred_train)
                RMSE_test, RMSE_val, RMSE_train = math.sqrt(mean_squared_error(y_test, y_pred)), math.sqrt(mean_squared_error(y_valid, y_pred_valid)), math.sqrt(mean_squared_error(y_train, y_pred_train))
                os.replace("C:/Users/Nick/Documents/MatDeepLearn/my_model.pth", "C:/Users/Nick/Documents/MatDeepLearn/ModelSavePath/my_model.pth")
                plot()
        if mode[1] == 1:  
            if((r2_score(y_test, y_pred) > 0 and r2_score(y_valid, y_pred_valid) > 0 and r2_score(y_train, y_pred_train) > 0) and (r2_score(y_test, y_pred) + 0.5*r2_score(y_valid, y_pred_valid) + 0.8*r2_score(y_train, y_pred_train)) > (r2_test + 0.5*r2_val + 0.8*r2_train)):
                r2_test, r2_val, r2_train = r2_score(y_test, y_pred), r2_score(y_valid, y_pred_valid), r2_score(y_train, y_pred_train)
                MAE_test, MAE_val, MAE_train = mean_absolute_error(y_test, y_pred), mean_absolute_error(y_valid, y_pred_valid), mean_absolute_error(y_train, y_pred_train)
                RMSE_test, RMSE_val, RMSE_train = math.sqrt(mean_squared_error(y_test, y_pred)), math.sqrt(mean_squared_error(y_valid, y_pred_valid)), math.sqrt(mean_squared_error(y_train, y_pred_train))
                os.replace("C:/Users/Nick/Documents/MatDeepLearn/my_model.pth", "C:/Users/Nick/Documents/MatDeepLearn/ModelSavePath/my_model.pth")
                plot()
            
        print('\n')
        print('-----------------------------------------------------------------------')
        print('|           Metric            |', ' Training  |  Validation  |  Testing  |')
        print('-----------------------------------------------------------------------')
        print('|          R² Score           |   ', '{:.3f}'.format(round(r2_train, 5)), '  |    ', '{:.3f}'.format(round(r2_val, 5)), '   |  ', '{:.3f}'.format(round(r2_test, 5)), '  |' )
        print('-----------------------------------------------------------------------')
        print('|     Mean Absolute Error     |   ', '{:.3f}'.format(round(MAE_train, 5)), '  |    ', '{:.3f}'.format(round(MAE_val, 5)), '   |  ', '{:.3f}'.format(round(MAE_test, 5)), '  |' )
        print('-----------------------------------------------------------------------')
        print('|   Root Mean Squared Error   |   ', '{:.3f}'.format(round(RMSE_train, 5)), '  |    ', '{:.3f}'.format(round(RMSE_val, 5)), '   |  ', '{:.3f}'.format(round(RMSE_test, 5)), '  |' )
        print('-----------------------------------------------------------------------')
        
else __name__ == "__main__":
        main()