import argparse
import itertools
import time
import os

import pandas as pd
import numpy as np
import tensorflow as tf

from preprocess.dataset import CollaborativeFilteringDataset
from model.als import train_and_predict_alternating_least_squares
from model.ncf import train_and_predict_ncf_model
from model.autoenc import train_and_predict_autoencoder

#cil_dataset = CollaborativeFilteringDataset("~/datasets/cil-collaborative-filtering-2022", apply_z_trafo=True, normalize_by_col=False)

# Current best approach
#train_and_predict_autoencoder(cil_dataset, width=6, depth=1, n=1, epochs=100, dropout_rate=0.5, strategy="standard", generate_plot=True)

def autoencoder_grid_search(args, n_repeats=1):
    cols = ["Depth", "Width", "Strategy", "Dropout_rate"] + [f"score_{i}" for i in range(n_repeats)]
    output_data = []
    for config in itertools.product(*[args.aenc_depth, args.aenc_width, args.aenc_strategy, args.aenc_dropout_rate]):
        depth, width, strategy, dropout_rate = config
        timestamp = time.ctime()
        row = [depth, width, strategy, dropout_rate]
        for i in range(n_repeats):
            dataset = CollaborativeFilteringDataset(args.data_path, val_split=args.val_split)
            dense_predictions = train_and_predict_autoencoder(dataset, width, depth, n=args.aenc_Nbag, epochs=args.aenc_epochs, dropout_rate=dropout_rate, strategy=strategy, generate_plot=args.convergence_plot, restore_best_weights=args.restore_best_weights)
            score = dataset.compute_val_score_from_dense(dense_predictions)
            row += [float(score)]
        output_data.append(row)
    df = pd.DataFrame(output_data, columns=cols)
    df.to_csv(os.path.join("scores", f"{timestamp}.csv"), index=False)
        


def autoencoder_predict(args):
    dataset = CollaborativeFilteringDataset(args.data_path, val_split=args.val_split)
    for config in itertools.product(*[args.aenc_depth, args.aenc_width, args.aenc_strategy, args.aenc_dropout_rate]):
        depth, width, strategy, dropout_rate = config
        dense_predictions = train_and_predict_autoencoder(dataset, width, depth, n=args.aenc_Nbag, epochs=args.aenc_epochs, dropout_rate=dropout_rate, strategy=strategy, generate_plot=args.convergence_plot, restore_best_weights=args.restore_best_weights)
        # submit
        dataset.create_submission_from_dense(dense_predictions)
        

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    # Args for the dataset
    parser.add_argument(
        "--data_path",
        type=str,
        default="~/datasets/cil-collaborative-filtering-2022"
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.05
    )
    # Args for the autoencoder model
    parser.add_argument(
        "--grid_search",
        default=False,
        action="store_true"
    )
    parser.add_argument(
        "--predict_aenc",
        default=False,
        action="store_true"
    )
    parser.add_argument(
        "--convergence_plot",
        default=False,
        action="store_true"
    )
    parser.add_argument(
        "--restore_best_weights",
        default=False,
        action="store_true"
    )
    # Hyperparameters for autoenc
    parser.add_argument(
        "--aenc_Nbag",
        type=int,
        default=1
    )
    parser.add_argument(
        "--aenc_depth",
        type=int,
        nargs="+",
        default=[1]
    )
    parser.add_argument(
        "--aenc_width",
        type=int,
        nargs="+",
        default=[8]
    )
    parser.add_argument(
        "--aenc_strategy",
        nargs="+",
        default=["standard"],
        choices=["standard", "effective", "renormalize"]
    )
    parser.add_argument(
        "--aenc_dropout_rate",
        type=float,
        nargs="+",
        default=[0.5]
    )
    parser.add_argument(
        "--aenc_epochs",
        type=int,
        default=1000
    )

    args = parser.parse_args()
    print(args)

    if args.grid_search: autoencoder_grid_search(args)


