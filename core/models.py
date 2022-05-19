from torch_geometric.nn import SGConv
from torch_geometric.typing import Adj, OptTensor
from typing import Tuple, Optional
import torch
from torch import Tensor
from torch_sparse import SparseTensor
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_scatter import scatter_add

from utils.basic_classes import StrategicModelParameters
from core.simulate_strategic_movement import simulate_strategic_movement


class ExtendedSGConv(SGConv):
    """
    An expanded SGC model which includes alpha
    """
    def __init__(self, in_channels: int, out_channels: int, K: int = 1,
                 add_self_loops: bool = True, bias: bool = True,
                 alpha: Optional[float] = None):
        super(ExtendedSGConv, self).__init__(in_channels=in_channels, out_channels=out_channels, K=K,
                                             add_self_loops=add_self_loops, bias=bias)
        self.alpha = alpha

    def activate_gcn_norm(self, x: Tensor, edge_index: Adj,
                          edge_weight: OptTensor = None) -> Tuple[Tensor, Tensor]:
        """
        Taken off the SGConv implementation

        :param x: Tensor
        :param edge_index: Adj
        :param edge_weight: OptTensor = None
        :return: edge_index: Tensor
        :return: edge_weight: Tensor
        """
        if isinstance(edge_index, Tensor):
            edge_index, edge_weight = gcn_norm(  # yapf: disable
                edge_index, edge_weight, x.size(self.node_dim), False,
                self.add_self_loops, dtype=x.dtype)
        elif isinstance(edge_index, SparseTensor):
            edge_index = gcn_norm(  # yapf: disable
                edge_index, edge_weight, x.size(self.node_dim), False,
                self.add_self_loops, dtype=x.dtype)
        return edge_index, edge_weight

    def get_self_weights_per_node(self, x: Tensor, edge_index: Adj, edge_weight: OptTensor = None)\
            -> Tensor:
        """
        A get function which returns the self weights of each nodes

        :param x: Tensor
        :param edge_index: Adj
        :param edge_weight: OptTensor = None
        :return: self_weights: Tensor
        """
        if self.alpha is None:
            edge_index, edge_weight = self.activate_gcn_norm(x=x, edge_index=edge_index,
                                                             edge_weight=edge_weight)
            return edge_weight[edge_index[0] == edge_index[1]]
        else:
            return (1 - self.alpha) * torch.ones(size=(x.shape[0],), device=edge_index.device)

    def _normalize_graph_topology(self, x: Tensor, edge_index: Adj,
                                  edge_weight: OptTensor = None) -> Tuple[Tensor, Tensor]:
        """
        Normalizes the edge_index/adjacency matrix

        :param x: Tensor
        :param edge_index: Adj
        :param edge_weight: OptTensor = None
        :return: edge_index: Tensor
        :return: edge_weight: Tensor
        """
        if self.alpha is None:
            edge_index, edge_weight = \
                self.activate_gcn_norm(x=x, edge_index=edge_index, edge_weight=edge_weight)
        else:
            n, device = x.shape[0], x.device

            # removes self loops
            edge_index = edge_index[:, edge_index[0] != edge_index[1]]

            # normalization
            deg = scatter_add(src=torch.ones(edge_index.shape[1], device=device, dtype=torch.float),
                              index=edge_index[1], dim=0, dim_size=n)
            edge_weight = deg[edge_index[1]].pow_(-1)

            # EDGE CASE: normalization for nodes without incoming edges
            nodes_without_incoming_edges = torch.tensor(list(set(range(n)) - set(edge_index[1].tolist())),
                                                        device=device, dtype=torch.long)
            edge_index_for_no_incoming_edges = torch.stack((nodes_without_incoming_edges,
                                                            nodes_without_incoming_edges), dim=0)
            edge_weight_for_no_incoming_edges = torch.ones(len(nodes_without_incoming_edges), device=device)
            edge_index = torch.cat((edge_index, edge_index_for_no_incoming_edges), dim=1)
            edge_weight = torch.cat((edge_weight, edge_weight_for_no_incoming_edges), dim=0)

        return edge_index, edge_weight

    def embed_graph(self, x: Tensor, edge_index: Adj,
                    edge_weight: OptTensor = None) -> Tensor:
        """
        Propagates and aggregates the node embeddings

        :param x: Tensor
        :param edge_index: Adj
        :param edge_weight: OptTensor = None
        :return:
        """
        original_x = x.clone()
        cache = self._cached_x

        if cache is None:
            edge_index, edge_weight =\
                self._normalize_graph_topology(x=x, edge_index=edge_index, edge_weight=edge_weight)

            for k in range(self.K):
                # propagate_type: (x: Tensor, edge_weight: OptTensor)
                x = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                                   size=None)
                if self.cached:
                    self._cached_x = x
        else:
            x = cache.detach()

        if self.alpha is not None:
            # Adds self loops
            x = self.alpha * x + (1 - self.alpha) * original_x
        return x

    def non_strategic_forward(self, x: Tensor, edge_index: Adj,
                              edge_weight: OptTensor = None) -> Tensor:
        """
        A forward function that does not take strategic movement into consideration

        :param x: Tensor
        :param edge_index: Adj
        :param edge_weight: OptTensor = None
        :return: Tensor
        """
        embedded_x = self.embed_graph(x=x, edge_index=edge_index, edge_weight=edge_weight)
        return self.lin(embedded_x)

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        return self.non_strategic_forward(x=x, edge_index=edge_index, edge_weight=edge_weight)


class StrategicSGConv(ExtendedSGConv):
    """
    Our model which defends against strategic movement
    """
    def __init__(self, in_channels: int, out_channels: int, K: int,
                 strategic_model_parameters: StrategicModelParameters,
                 add_self_loops: bool = True, bias: bool = True,
                 alpha: Optional[float] = None):
        super(StrategicSGConv, self).__init__(in_channels=in_channels, out_channels=out_channels,
                                              K=K, add_self_loops=add_self_loops,
                                              bias=bias, alpha=alpha)
        self.strategic_model_parameters = strategic_model_parameters

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        """
        A forward function that DOES take strategic movement into consideration

        :param x: Tensor
        :param edge_index: Adj
        :param edge_weight: OptTensor = None
        :return: Tensor
        """
        x = simulate_strategic_movement(x_init=x, edge_index=edge_index, model=self,
                                        strategic_model_parameters=self.strategic_model_parameters,
                                        exact_movement=not self.training)
        return self.non_strategic_forward(x=x, edge_index=edge_index)
