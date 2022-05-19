import torch
from torch.nn import BCELoss, Flatten, Sequential, Sigmoid
from torch_geometric.data.data import Data
from torch_geometric.nn import MessagePassing
from typing import List

SIGMOID = Sequential(Sigmoid(), Flatten(start_dim=0))
LOSS = BCELoss()
BIASES = (-1 * torch.arange(-2, 5, 0.05)).tolist()


def train(data: Data, model: MessagePassing, optimizer, epochs: int):
    """
    A training method over multiple epochs

    :param data: Data
    :param model: MessagePassing
    :param optimizer:
    :param epochs: int
    """
    for epoch in range(0, epochs):
        _train_per_epoch(data=data, model=model, optimizer=optimizer)


def _train_per_epoch(data: Data, model: MessagePassing, optimizer):
    """
    Trains for a single epoch

    :param data: Data
    :param model: MessagePassing
    :param optimizer:
    """
    model.train()

    optimizer.zero_grad()
    probabilities = SIGMOID(model(x=data.x, edge_index=data.edge_index))
    LOSS(probabilities[data.train_mask], data.y[data.train_mask]).backward()
    optimizer.step()


@torch.no_grad()
def test(data: Data, model: MessagePassing) -> List[float]:
    """
    A testing method

    :param data: Data
    :param model: MessagePassing
    :return: acc: List[float]
    """
    model.eval()
    logits, accs = model(x=data.x, edge_index=data.edge_index), []
    for _, mask in data('train_mask', 'test_mask'):
        preds = (logits[mask] > 0).long().flatten()
        num_correct_nodes = preds.eq(data.y[mask]).sum().item()
        num_nodes = mask.sum().item()
        accs.append(100 * num_correct_nodes / num_nodes)
    return accs


@torch.no_grad()
def line_search(data: Data, model: MessagePassing) -> MessagePassing:
    """
    Performs line search to find the best bias

    :param data: Data
    :param model: MessagePassing
    :return: model: MessagePassing
    """
    best_acc, best_bias = -1, BIASES[0]
    model.lin.weight[0][0] = 1
    for bias in BIASES:
        model.lin.bias[0] = bias
        acc = test(data=data, model=model)[0]
        if acc > best_acc:
            best_acc = acc
            best_bias = bias

    model.lin.bias[0] = best_bias
    return model
