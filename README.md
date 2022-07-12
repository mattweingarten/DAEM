# CIL Collaborative Filtering 2022

## Instructions
Make a prediction using our denoising autoencoder model:
```
python main.py --data_path <PATH> --val_split 0.0 --aenc_predict --aenc_Nbag <N> --aenc_depth <D> --aenc_width <W> --aenc_dropout_rate <R> --aenc_epochs <EPOCHS>
```

Run Grid Search for the autoencoder model:
```
python main.py --data_path <PATH> --val_split 0.0 --aenc_grid_search --aenc_Nbag <N1,N2,...> --aenc_depth <D1,D2,...> --aenc_width <W1,W2,...> --aenc_dropout_rate <R1,R2,...> --aenc_epochs <E1,E2,...> --aenc_strategy standard effective renormalize
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

