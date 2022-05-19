import torch
from torch import Tensor
from torch.nn import Sigmoid
from typing import Optional

from utils.basic_classes import StrategicModelParameters

L_NORM = 2
SIGMOID = Sigmoid()
TEST_TOLERANCE = 1e-4
ASSERT_TOL = 1e-3


def cost_function(x: torch.Tensor, y: torch.Tensor):
    return torch.linalg.norm(y - x, ord=L_NORM, dim=1, keepdim=True)


def stop_criteria_function(iterations: Optional[int], cost_lim: bool):
    """
    Returns the condition/function by which we use to stop (or not) the strategic movement

    :param iterations:
    :param cost_lim:
    """
    if not (cost_lim > 0):
        return lambda idx, moved: False
    elif iterations is None:
        return lambda idx, moved: moved.sum().item()  # up to a convergence
    elif iterations > 0:
        return lambda idx, moved: (idx < iterations)  # up to a number of iterations
    else:
        return lambda idx, moved: False


def simulate_strategic_movement(x_init: Tensor, edge_index: Tensor, model,
                                strategic_model_parameters: StrategicModelParameters,
                                exact_movement: bool) -> Tensor:
    """
    Simulates the exact/approximate strategic movement

    :param x_init: Tensor
    :param edge_index: Tensor
    :param model:
    :param strategic_model_parameters: StrategicModelParameters
    :param exact_movement: bool
        Whether the strategic movement is exact or approximated
    :return: strategic_x: Tensor
    """
    # constants
    cost_lim, train_iterations, test_iterations, temp = strategic_model_parameters
    x, n = x_init.clone(), x_init.shape[0]
    w = model.lin.weight.T.squeeze(dim=1)  # shape (f,)
    tolerance = TEST_TOLERANCE if exact_movement else 0
    iterations = test_iterations if exact_movement else train_iterations

    # changed parameters
    idx = 1
    nodes_previously_moved = torch.zeros(size=(n, 1), dtype=torch.bool, device=x.device)
    iteration_criteria = (cost_lim > 0) if iterations is None else (iterations > 0) and (cost_lim > 0)
    kappa = cost_function(x, x)

    while iteration_criteria:
        model_output = model.non_strategic_forward(x=x, edge_index=edge_index)  # shape (n, 1)
        negatively_classified_nodes = (model_output < 0)  # shape (n, 1)

        assert not exact_movement or ((nodes_previously_moved * model_output).min().item() >= 0), \
            "Our lagrangian multipliers are incorrect"

        # the strategic movement with no limits
        self_weights = model.get_self_weights_per_node(x=x, edge_index=edge_index)  # shape (n,)
        normalization = (w ** 2).sum(dim=0) * self_weights  # shape (n,)
        normalized_model_outputs = (model_output - tolerance) / normalization.unsqueeze(dim=1)
        proj_x = x - negatively_classified_nodes * normalized_model_outputs * w  # shape (n, f)

        # limiting the movement
        cost = cost_function(proj_x, x_init)  # shape (n, 1)
        assert torch.allclose(cost, cost_function(proj_x, x) + kappa, rtol=ASSERT_TOL), \
            "A Quadratic Cost function should hold the equality in the triangle inequality"
        cost_left = (cost_lim - cost)  # shape (n, 1)

        # limited movement
        moved = (cost_left > 0) & negatively_classified_nodes  # shape (n, 1)
        if exact_movement:
            next_x = x + moved.float() * (proj_x - x)
            assert (nodes_previously_moved & moved).sum() == 0, "Nodes move multiple times"
        else:
            next_x = x + (proj_x - x) * SIGMOID(cost_left / temp)  # shape (n, f)

        # check criteria
        if iterations is None:
            iteration_criteria = moved.sum().item()  # up to a convergence
        else:
            iteration_criteria = (idx < iterations)

        # preparing for the next iterations
        nodes_previously_moved = nodes_previously_moved | moved
        idx += 1
        kappa += cost_function(next_x, x)
        x = next_x
    return x
