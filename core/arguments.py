from argparse import ArgumentParser

from utils.basic_classes import DataSet


def get_synthetic_dataset_parser() -> ArgumentParser:
    parser = ArgumentParser()

    # synthetic dataset settings
    parser.add_argument("--alpha", dest='alpha', type=float, default=0.5, required=False)

    # StrategicModelParameters
    parser.add_argument("--cost_lim", dest='cost_lim', type=float, default=2.0, required=False)
    parser.add_argument("--train_iterations", dest='train_iterations', type=int, default=None, required=False)
    parser.add_argument("--test_iterations", dest='test_iterations', type=int, default=None, required=False)

    # result reproduction
    parser.add_argument("--seed", dest="seed", type=int, default=0, required=False)
    return parser


def get_real_datasets_parser() -> ArgumentParser:
    parser = ArgumentParser()

    # real datasets settings
    parser.add_argument("--dataset", dest="dataset_name", default=DataSet.CORA,
                        type=DataSet.from_string, choices=list(DataSet), required=False)
    parser.add_argument("--num_layers", dest='num_layers', type=int, default=1, required=False)

    # StrategicModelParameters
    parser.add_argument("--cost_lim", dest='cost_lim', type=float, default=0.25, required=False)
    parser.add_argument("--train_iterations", dest='train_iterations', type=int, default=3, required=False)
    parser.add_argument("--temp", dest="temp", type=float, default=0.05, required=False)

    # optimization parameters
    parser.add_argument("--lr", dest="lr", type=float, default=0.2, required=False)
    parser.add_argument("--epochs", dest="epochs", type=int, default=20, required=False)
    parser.add_argument("--decay", dest="decay", type=float, default=1.3e-5, required=False)

    # result reproduction
    parser.add_argument("--seed", dest="seed", type=int, default=0, required=False)
    return parser
