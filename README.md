# CIL Collaborative Filtering 2022


## Abstract 

Highly accurate Recommender Systems, including
Collaborative Filtering, lie at the heart of a satisfactory
customer experience and continuous user engagement for a
plethora of large-scale online platforms. While Matrix Factorization is the most widely studied and 
applied Collaborative Filtering approach, there is evidence to suggest that linear techniques lack the complexity to sufficiently capture
the underlying relationship between users and items. The
use of neural networks like Autoencoders offers a potential
remedy and may more accurately represent this relationship.
In this work, we propose our Denoising Autoencoder Model
(DÆM) for highly accurate Collaborative Filtering and show
improvement over four evaluated state-of-the-art models.

## Setup

This repository uses numpy and Tensorflow 2 along with other common data science libraries. Use `requirements.txt` to install the correct versions. Be sure to dowload the CIL Collaborative Filtering 2022 dataset and supply the directory as command line argument.

## Instructions
To print the possible command line arguments use:
```
python main.py --help
```

### Reproducing the kaggle results

To make a prediction using our denoising autoencoder model, use the following command with selected hyperparameters:
```
python main.py --data_path <PATH> --val_split 0.0 --aenc_predict --aenc_Nbag <N> --aenc_depth <D> --aenc_width <W> --aenc_dropout_rate <R> --aenc_epochs <EPOCHS>
```

Our kaggle submissions are annotated with the command that was used to generate them. So they should have a comment that contains a command like the above. Predictions are stored in the appropriate submission format under the `predictions` directory.

### Grid search and baseline results
In the report, we employ basic grid search to find good hyperparameters for both our model and the baseline models we used for comparison. The following commands perform this grid search, compute the score on a holdout set, then save the results in the `scores` directory.

Run Grid Search for the autoencoder model:
```
python main.py --data_path <PATH> --aenc_grid_search --aenc_Nbag <N1,N2,...> --aenc_depth <D1,D2,...> --aenc_width <W1,W2,...> --aenc_dropout_rate <R1,R2,...> --aenc_epochs <E1,E2,...> --aenc_strategy standard effective renormalize
```

Run Grid Search for baseline model SVD:
```
python main.py --data_path <PATH> --baseline_SVD_grid_search --baseline_SVD_rank <R1,R2,...>
```

Run Grid Search for baseline model ALS:
```
python main.py --data_path <PATH> --baseline_als_grid_search --baseline_als_rank <R1,R2,...> --baseline_als_l2 <X1,X2,...>
```

Run Grid Search for baseline model SLIM:
```
python main.py --data_path <PATH> --baseline_SLIM_grid_search --baseline_SLIM_l1 <X1,X2,...>
```

Run Grid Search for baseline model [NCF](https://arxiv.org/abs/1708.05031?context=cs):
```
python main.py --data_path <PATH> --baseline_ncf_grid_search --baseline_ncf_model_type gmf mlp ncf --baseline_ncf_factors <F1,F2,...> --baseline_ncf_epochs <E1,E2,...>
```

## Plotting
The `plotting` directory contains the code used to generate the plots in the report along with grid search results used to generate them. These results should be reproducible using the above instructions.

