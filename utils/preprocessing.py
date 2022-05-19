import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj
from typing import List, Tuple, Dict

MINIMAL_TEST_RATIO = 0.01


def get_mask_for_class(y: torch.Tensor, classes: List[int]) -> torch.Tensor:
    """
        Returns a mask of True values for the classes that are stated

    :param y: torch.Tensor
    :param classes: List[int]
    :return: mask: torch.Tensor
    """
    mask = (y == classes[0])
    for sole_class in classes[1:]:
        mask = mask | (y == sole_class)
    return mask


def binarize_dataset(data: Data, class_split: Dict[str, List[int]]) -> Data:
    """
        Binarizes the dataset

    :param data: Data
    :param class_split: Dict[str, List[int]]
        A dictionary where the keys are the strings 'negative' or 'positive' and
        the values are Lists of the classes which correspond to the split which is specified in the key
    :return: data: Data
    """
    num_classes = data.y.max().item() + 1
    list_of_negative_classes, list_of_positive_classes = class_split['negative'], class_split['positive']
    list_of_classes = list_of_negative_classes + list_of_positive_classes

    assert len(set(list_of_negative_classes) & set(list_of_positive_classes)) == 0, \
        "Positive classes can't have any overlap with the negative classes"
    assert set(list_of_classes).issubset(set(range(num_classes))), \
        "Positive classes and negative classes must be in the range of the number of classes for the dataset"

    # get positive and negative masks
    negative_mask = get_mask_for_class(y=data.y, classes=list_of_negative_classes)
    positive_mask = get_mask_for_class(y=data.y, classes=list_of_positive_classes)
    all_masked = negative_mask | positive_mask

    # remove nodes that are not included in any class
    data.train_mask[~all_masked] = False
    data.test_mask[~all_masked] = False

    # binarize
    data.y[negative_mask] = 0
    data.y[positive_mask] = 1
    data.y = data.y.type(torch.FloatTensor)
    return data


def direct_edge_index(edge_index: torch.Tensor) -> torch.Tensor:
    """
        Directs a given edge_index tensor from larger degree nodes to smaller degree nodes

    :param edge_index: torch.Tensor
    :return: edge_index: torch.Tensor
    """
    adj = to_dense_adj(edge_index=edge_index).squeeze(0)
    final_adj = adj.clone()
    degree = adj.sum(dim=1)

    # direct each edge from larger degree nodes to smaller degree nodes
    for from_node, to_node in edge_index.T:
        if degree[from_node] > degree[to_node]:
            final_adj[to_node, from_node] = 0
        elif degree[from_node] < degree[to_node]:
            final_adj[from_node, to_node] = 0
        elif from_node > to_node:
            final_adj[from_node, to_node] = 0
        else:
            final_adj[to_node, from_node] = 0
    return final_adj.nonzero().T


def make_data_inductive(data: Data, k: int) -> Data:
    """
        Removes nodes from the test set which influence the train set

    :param data: Data
    :param k: int
        Number of layers for the model
    :return: inductive_data: Data
    """
    n = data.x.shape[0]
    seen_nodes, nodes_not_seen = [], data.train_mask.nonzero().flatten().tolist()

    # collects all nodes in the k-hop vicinity of the train set
    for _ in range(k):
        neighbor_nodes = []
        for node_idx in nodes_not_seen:
            neighbor_nodes += data.edge_index[1, data.edge_index[0] == node_idx].tolist()

        nodes_not_seen = list(set(neighbor_nodes) - set(seen_nodes))

        # record nodes not seen before
        seen_nodes += nodes_not_seen

    # create a k-hop vicinity train set mask
    train_mask_k_hop_vicinity = torch.zeros(size=(n,), dtype=torch.bool)
    train_mask_k_hop_vicinity[seen_nodes] = True

    # remove nodes form m the test mask which are in the k-hop vicinity train set mask
    data.test_mask[train_mask_k_hop_vicinity] = False
    return data
