# FedTMOS: Efficient One-Shot Federated Learning with Tsetlin Machine (ICLR'25)




## Requirements

- To install pyTsetlinMachineParallel_vote, ensure that you have gcc 
- cd into this directory then call:
  
  
```
make
```

## Running the code:

```
python oneshotfedtmos.py --epochs 1 --dirichlet 2 --local_epochs 30  --n_clauses 200 --T 1000 --s 5 --patch_dim 5 --dataset 'F-MNIST'  --dir '/data/dir/' --num_models 1 --k 10 --num_clients 10 --c 3 --seed 20 --load_pretrain 0 --dir_type 'cls' 

python oneshotfedtmos.py --epochs 1 --dirichlet 0.3 --local_epochs 30  --n_clauses 200 --T 1000 --s 5 --patch_dim 5 --dataset 'F-MNIST'  --dir '/data/dir/' --num_models 1 --k 30 --num_clients 10 --c 3 --seed 20 --load_pretrain 0 

```

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
