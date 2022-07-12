import argparse
from codecs import latin_1_decode
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
from model.svd import train_and_predict_low_rank_approx
from model.slim import train_and_predict_SLIM

#cil_dataset = CollaborativeFilteringDataset("~/datasets/cil-collaborative-filtering-2022", apply_z_trafo=True, normalize_by_col=False)

# Current best approach
#train_and_predict_autoencoder(cil_dataset, width=6, depth=1, n=1, epochs=100, dropout_rate=0.5, strategy="standard", generate_plot=True)

def autoencoder_grid_search(args):
    cols = ["Depth", "Width", "N_bag", "epochs", "Strategy", "Dropout_rate"] + [f"score_{i}" for i in range(args.n_repeats)]
    output_data = []
    timestamp = time.ctime()
    for config in itertools.product(*[args.aenc_depth, args.aenc_width, args.aenc_Nbag, args.aenc_epochs, args.aenc_strategy, args.aenc_dropout_rate]):
        depth, width, Nbag, epochs, strategy, dropout_rate = config
        row = [depth, width, Nbag, epochs, strategy, dropout_rate]
        for i in range(args.n_repeats):
            dataset = CollaborativeFilteringDataset(args.data_path, val_split=args.val_split)
            dense_predictions = train_and_predict_autoencoder(dataset, width, depth, n=Nbag, epochs=epochs, dropout_rate=dropout_rate, strategy=strategy, generate_plot=args.convergence_plot, restore_best_weights=args.restore_best_weights)
            score = dataset.compute_val_score_from_dense(dense_predictions)
            row += [float(score)]
        output_data.append(row)
    df = pd.DataFrame(output_data, columns=cols)
    df.to_csv(os.path.join("scores", f"AUTOENC-{timestamp}.csv"), index=False)
        
def autoencoder_predict(args):
    dataset = CollaborativeFilteringDataset(args.data_path, val_split=args.val_split)
    for config in itertools.product(*[args.aenc_depth, args.aenc_width, args.aenc_Nbag, args.aenc_epochs, args.aenc_strategy, args.aenc_dropout_rate]):
        depth, width, Nbag, epochs, strategy, dropout_rate = config
        dense_predictions = train_and_predict_autoencoder(dataset, width, depth, n=Nbag, epochs=epochs, dropout_rate=dropout_rate, strategy=strategy, generate_plot=args.convergence_plot, restore_best_weights=args.restore_best_weights)
        # submit
        dataset.create_submission_from_dense(dense_predictions)

def als_grid_search(args):
    cols = ["rank", "l2"] + [f"score_{i}" for i in range(args.n_repeats)]
    output_data = []
    timestamp = time.ctime()
    for (rank, l2) in itertools.product(*[args.baseline_als_rank, args.baseline_als_l2]):
        row = [rank, l2]
        for i in range(args.n_repeats):
            dataset = CollaborativeFilteringDataset(args.data_path, val_split=args.val_split, normalize_by_col=True)
            dense_predictions = train_and_predict_alternating_least_squares(dataset, k=rank, lamb=l2, iters=args.baseline_als_iters)
            score = dataset.compute_val_score_from_dense(dense_predictions)
            row += [float(score)]
        output_data.append(row)
    df = pd.DataFrame(output_data, columns=cols)
    df.to_csv(os.path.join("scores", f"ALS-{timestamp}.csv"), index=False)
        
def ncf_grid_search(args):
    cols = ["model_type", "factors", "epochs"] + [f"score_{i}" for i in range(args.n_repeats)]
    output_data = []
    timestamp = time.ctime()
    for (model_type, factors, epochs) in itertools.product(*[args.baseline_ncf_model_type, args.baseline_ncf_factors, args.baseline_ncf_epochs]):
        row = [model_type, factors, epochs]
        for i in range(args.n_repeats):
            dataset = CollaborativeFilteringDataset(args.data_path, val_split=args.val_split)
            model = train_and_predict_ncf_model(dataset, n_latent=factors, model_type=model_type, epochs=epochs)
            locations = dataset.get_val_locations()
            predictions = model.predict(locations, batch_size=1<<10)
            score = dataset.compute_val_score(locations, predictions)
            row += [float(score)]
        output_data.append(row)
    df = pd.DataFrame(output_data, columns=cols)
    df.to_csv(os.path.join("scores", f"NCF-{timestamp}.csv"), index=False)

def slim_grid_search(args):
    cols = ["l1"] + [f"score_{i}" for i in range(args.n_repeats)]
    output_data = []
    timestamp = time.ctime()
    for l1 in args.baseline_SLIM_l1:
        row = [l1]
        for i in range(args.n_repeats):
            dataset = CollaborativeFilteringDataset(args.data_path, val_split=args.val_split)
            dense_predictions = train_and_predict_SLIM(dataset, l1=l1)
            score = dataset.compute_val_score_from_dense(dense_predictions)
            row += [float(score)]
        output_data.append(row)
    df = pd.DataFrame(output_data, columns=cols)
    df.to_csv(os.path.join("scores", f"SLIM-{timestamp}.csv"), index=False)

def svd_grid_search(args):
    cols = ["rank"] + [f"score_{i}" for i in range(args.n_repeats)]
    output_data = []
    timestamp = time.ctime()
    for rank in args.baseline_SVD_rank:
        row = [rank]
        for i in range(args.n_repeats):
            dataset = CollaborativeFilteringDataset(args.data_path, val_split=args.val_split, normalize_by_col=True)
            dense_predictions = train_and_predict_low_rank_approx(dataset, rank)
            score = dataset.compute_val_score_from_dense(dense_predictions)
            row += [float(score)]
        output_data.append(row)
    df = pd.DataFrame(output_data, columns=cols)
    df.to_csv(os.path.join("scores", f"SVD-{timestamp}.csv"), index=False)

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
        default=0.1
    )
    parser.add_argument(
        "--n_repeats",
        type=int,
        default=5
    )
    # Args for the autoencoder model
    parser.add_argument(
        "--grid_search",
        default=False,
        action="store_true"
    )
    parser.add_argument(
        "--aenc_predict",
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
        nargs="+",
        default=[1]
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
        nargs="+",
        default=[1000]
    )
    # Args for the ALS baseline
    parser.add_argument(
        "--baseline_als_grid_search",
        default=False,
        action="store_true"
    )
    parser.add_argument(
        "--baseline_als_rank",
        type=int,
        nargs="+",
        default=[3]
    )
    parser.add_argument(
        "--baseline_als_l2",
        type=float,
        nargs="+",
        default=[0.1]
    )
    parser.add_argument(
        "--baseline_als_iters",
        type=int,
        default=20
    )
    # Args for the NCF baseline
    parser.add_argument(
        "--baseline_ncf_grid_search",
        default=False,
        action="store_true"
    )
    parser.add_argument(
        "--baseline_ncf_model_type",
        nargs="+",
        default="ncf",
        choices=["gmf", "mlp", "ncf"]
    )
    parser.add_argument(
        "--baseline_ncf_factors",
        type=int,
        nargs="+",
        default=[16]
    )
    parser.add_argument(
        "--baseline_ncf_epochs",
        type=int,
        nargs="+",
        default=[20]
    )
    # Args for SLIM baseline
    parser.add_argument(
        "--baseline_SLIM_grid_search",
        default=False,
        action="store_true"
    )
    parser.add_argument(
        "--baseline_SLIM_l1",
        type=float,
        nargs="+",
        default=[0.1],
    )
    # Args for SVD baseline
    parser.add_argument(
        "--baseline_SVD_grid_search",
        default=False,
        action="store_true"
    )
    parser.add_argument(
        "--baseline_SVD_rank",
        type=int,
        nargs="+",
        default=[3]
    )


    args = parser.parse_args()

    if args.grid_search: autoencoder_grid_search(args)
    if args.aenc_predict: autoencoder_predict(args)
    if args.baseline_als_grid_search: als_grid_search(args)
    if args.baseline_ncf_grid_search: ncf_grid_search(args)
    if args.baseline_SVD_grid_search: svd_grid_search(args)
    if args.baseline_SLIM_grid_search: slim_grid_search(args)


