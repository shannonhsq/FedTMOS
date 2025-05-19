# FedTMOS: Efficient One-Shot Federated Learning with Tsetlin Machine

This repository is the implementation of [FedTMOS: Efficient One-Shot Federated Learning with Tsetlin Machine (ICLR'25)](https://openreview.net/forum?id=44hcrfzydU&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DICLR.cc%2F2025%2FConference%2FAuthors%23your-submissions))


<img width="1530" alt="image" src="src/iclrposter_shannon.pdf">


## Requirements

- To install pyTsetlinMachineParallel_vote, ensure that you have gcc 
- cd into this directory then call:
  
  
```
make
```

## Running the code:

```
python oneshotfedtmos.py --epochs 1 --dirichlet 2 --local_epochs 30  --n_clauses 200 --T 1000 --s 5 --patch_dim 5 --dataset 'F-MNIST'  --data_dir '/data/dir/' --num_models 1 --k 10 --num_clients 10 --c 3 --seed 20 --load_pretrain 0 --dir_type 'cls' 

python oneshotfedtmos.py --epochs 1 --dirichlet 0.3 --local_epochs 30  --n_clauses 200 --T 1000 --s 5 --patch_dim 5 --dataset 'F-MNIST'  --data_dir '/data/dir/' --num_models 1 --k 30 --num_clients 10 --c 3 --seed 20 --load_pretrain 0 

```

Configurations:
- `--dataset`: the name of the datasets (eg. `MNIST`, `F-MNIST`,`SVHN`,`CIFAR-10`).
- `--num_clients`: the number of total clients
- `--local_epochs`: the number of epochs for local training
- `--epochs`: the number of communication rounds
- `--num_models`: the number of local models
- `--load_pretrain`: whether to load pretrain local models for each client
- `--data_dir`: the location of the datasets 
- `--seed`: random seed (note that the seed corresponds to the data distribution split. for reproducibility, we used seed = {0,20,24} in all experiments)
- `--dir_type`:
  - `dir`: split the data using a Dirichlet distribution with the concentration parameter defined by --dirichlet.
  - `cls`: allocate data by assigning each client a fixed number of classes, specified by --dirichlet.
- `--dirichlet`:
  - If `--dir_type=dir`: this is the concentration parameter (β) for the Dirichlet distribution used to split data non-IID across clients, amaller values result in more heterogeneous (non-IID) splits.
  - If `--dir_type=cls`: this indicates the number of classes assigned to each client, data is split such that each client receives samples from this many classes.  
- `--n_clauses`: the number of clauses of the model
- `--T`: the feedback threshold of the model
- `--s`: the learning sensitivity of the model
- `--c`: the number of server models
- `--k`: the number of clusters




```
@inproceedings{
qi2025fedtmos,
title={Fed{TMOS}: Efficient One-Shot Federated Learning with Tsetlin Machine},
author={Shannon How Shi Qi and Jagmohan Chauhan and Geoff V. Merrett and Jonathon Hare},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
url={https://openreview.net/forum?id=44hcrfzydU}
}

```
