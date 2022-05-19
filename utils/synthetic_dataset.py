import torch
import numpy as np
from torch_geometric.data import Data

NUM_NODES = 10000
FEATURES_MEAN = 1
FEATURES_STD = 1.0

IN_CLUSTER_EDGES_PER_NODE = 5
CROSS_CLUSTER_EDGES_PER_NODE = 3


class SYNTHETIC_SET(object):
    """
    Generates a synthetic dataset for a single split (train or test)
    """
    def __init__(self):
        super(SYNTHETIC_SET, self).__init__()

        # negative cluster
        neg_x = torch.normal(mean=-FEATURES_MEAN * torch.ones(NUM_NODES,), std=FEATURES_STD)
        neg_in_edges = SYNTHETIC_SET._create_in_cluster_edges()
        neg_cross_edges = SYNTHETIC_SET._create_cross_cluster_edges(negative_to_positive_direction=True)
        neg_y = torch.zeros(size=(NUM_NODES,), dtype=torch.float)

        # positive cluster
        pos_x = torch.normal(mean=FEATURES_MEAN * torch.ones(NUM_NODES, ), std=FEATURES_STD)
        pos_in_edges = SYNTHETIC_SET._create_in_cluster_edges() + NUM_NODES
        pos_cross_edges = SYNTHETIC_SET._create_cross_cluster_edges(negative_to_positive_direction=False)
        pos_y = torch.ones(size=(NUM_NODES,), dtype=torch.float)

        # merge clusters
        self.x = torch.cat([neg_x, pos_x], dim=0).unsqueeze(1)
        self.edge_index = torch.cat([neg_in_edges, neg_cross_edges,
                                     pos_in_edges, pos_cross_edges], dim=1)
        self.y = torch.cat([neg_y, pos_y], dim=0)

    @staticmethod
    def _create_in_cluster_edges() -> torch.Tensor:
        """
        Creates edges inside a cluster

        :return: edge_index: torch.Tensor
        """
        from_node, to_node = [], []
        for node_idx in range(NUM_NODES):
            possible_neighbors = np.concatenate((np.arange(node_idx), np.arange(node_idx + 1, NUM_NODES)), axis=0)
            neighbors = np.random.choice(possible_neighbors, size=IN_CLUSTER_EDGES_PER_NODE,
                                         replace=False)
            from_node += neighbors.tolist()
            to_node += [node_idx] * IN_CLUSTER_EDGES_PER_NODE
        return torch.tensor([from_node, to_node])

    @staticmethod
    def _create_cross_cluster_edges(negative_to_positive_direction: bool) -> torch.Tensor:
        """
        Creates edges in between clusters

        :param negative_to_positive_direction: bool
        :return: edge_index: torch.Tensor
        """
        if negative_to_positive_direction:
            node_idx_shift_from = 0
            node_idx_shift_to = NUM_NODES
        else:
            node_idx_shift_from = NUM_NODES
            node_idx_shift_to = 0

        edge_origin_list, edge_destination_list = [], []
        for node_idx in range(NUM_NODES):
            destinations = np.random.choice(np.arange(NUM_NODES), size=CROSS_CLUSTER_EDGES_PER_NODE,
                                            replace=False) + node_idx_shift_from
            edge_origin_list += destinations.tolist()

            edge_destination_list += [node_idx + node_idx_shift_to] * CROSS_CLUSTER_EDGES_PER_NODE
        return torch.tensor([edge_origin_list, edge_destination_list])


def get_synthetic_dataset() -> Data:
    """
    A get function for a synthetic dataset with all splits included

    :return: synthetic_data: Data
    """
    train_set = SYNTHETIC_SET()
    test_set = SYNTHETIC_SET()

    # merge splits
    x = torch.cat((train_set.x, test_set.x), dim=0)
    edge_index = torch.cat((train_set.edge_index,
                            test_set.edge_index + train_set.x.shape[0]), dim=1)
    y = torch.cat((train_set.y, test_set.y), dim=0)

    data = Data(x=x, edge_index=edge_index, y=y)

    # create masks
    test_mask = torch.ones(x.shape[0], dtype=torch.bool)
    test_mask[:train_set.x.shape[0]] = False
    setattr(data, 'train_mask', ~test_mask)
    setattr(data, 'test_mask', test_mask)
    return data
