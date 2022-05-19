from enum import Enum, auto
import os
import torch
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from typing import NamedTuple, Dict, Optional, List

from utils.general_helpers import get_git_path
from utils.preprocessing import binarize_dataset, direct_edge_index, make_data_inductive
from utils.synthetic_dataset import get_synthetic_dataset


class SplitRealDatasetParameters(NamedTuple):
    """
        An object for the parameters that dictate the split of our real datasets
    """
    negative_classes: int
    positive_classes: int
    num_layers: int


class StrategicModelParameters(NamedTuple):
    """
        An object for the parameters of a strategic model
    """
    cost_lim: float
    train_iterations: Optional[int] = None
    test_iterations: Optional[int] = None
    temp: Optional[float] = None


class DataSet(Enum):
    """
        An object for our datasets
    """
    SYNTHETIC = auto()

    CORA = auto()
    CITESEER = auto()
    PUBMED = auto()

    @staticmethod
    def from_string(s: str):
        try:
            return DataSet[s]
        except KeyError:
            raise ValueError()

    def string(self) -> str:
        return DataSet(self).name

    def get_dataset(self, num_layers: int) -> Data:
        """
        A get function for the dataset

        :param num_layers: int
        :return: data: Data
        """
        dataset_path = os.path.join(get_git_path(), 'datasets')
        if not os.path.exists(dataset_path):
            os.mkdir(dataset_path)
        dataset_name = self.string()

        if self is DataSet.SYNTHETIC:
            path = os.path.join(dataset_path, dataset_name + '.pt')
            if os.path.exists(path):
                data = torch.load(path)
            else:
                data = get_synthetic_dataset()
                torch.save(data, path)
        else:
            dataset = Planetoid(dataset_path, dataset_name)
            data = dataset[0]

            # remove validation set
            data.train_mask += data.val_mask
            delattr(data, 'val_mask')

            # preprocess real dataset
            if data.y.max().item() + 1 != 2:
                data = binarize_dataset(data=data, class_split=self.get_predetermined_class_split())
                setattr(data, 'num_classes', 2)

            if not data.is_directed():
                data.edge_index = direct_edge_index(edge_index=data.edge_index)

            data = make_data_inductive(data=data, k=num_layers)
        return data

    def get_predetermined_class_split(self) -> Dict[str, List[int]]:
        """
        The predetermined class split for each dataset

        :return: split: Dict[str, List[int]]
        """
        if self is DataSet.CORA:
            return {'negative': [0, 2, 3], 'positive': [1, 4, 5, 6]}
        elif self is DataSet.CITESEER:
            return {'negative': [0, 2, 3], 'positive': [1, 4, 5]}
        elif self is DataSet.PUBMED:
            return {'negative': [1, 2], 'positive': [0]}
        else:
            exit("Irrelevant Dataset")
