# Single-Node Attack For Fooling Graph Neural Networks

This repository is the official implementation of Strategic Classification with Graph Neural Networks.

## Requirements
This project is based on PyTorch 1.8.0 and the PyTorch Geometric library.

First, install PyTorch from the official website: https://pytorch.org/.
Then install PyTorch Geometric: https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html
(PyTorch Geometric must be installed according to the instructions there).

## Experiments

1. **synthetic_dataset_main**
Returns the non-strategic, naive and robust accuracies for a single experiment, performed on our synthetic dataset.

The available input arguments are:
* `--alpha`: The influence of the graph.

* `--cost_lim`: The maximal moving distance.

* `--train_iterations`: The limit on the number of train iterations. If not set, then we iterate until we converge.

* `--test_iterations`: The limit on the number of test iterations. If not set, then we iterate until we converge.

* `--seed`: A seed for reproducability.

2. **real_datasets_main**
Returns the non-strategic, naive and robust accuracies for a single experiment, performed on a real dataset (Cora, CiteSeer or PubMed).


The available input arguments are:
* `--dataset`: Name of the real dataset, all caps.

* `--num_layers`: The number of layers in the SGC model.

* `--cost_lim`: The maximal moving distance.

* `--train_iterations`: The limit on the number of train iterations. If not set, then we iterate until we converge.

* `--temp`: The sigmoid temperature.

* `--lr`: The learning rate.

* `--epochs`: The number of training epochs.

* `--decay`: The weight decay.

* `--seed`: A seed for reproducability.