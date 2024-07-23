"""
Copyright (C) [2023] [Tharuka Kasthuriarachchige]

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

Paper: [FeDABoost: AdaBoost Enhanced Federated Learning]
Published in: 
"""

import torch

from typing import List

def fedAvg(global_model: torch.nn.Module, local_models: List[torch.nn.Module]) -> torch.nn.Module:
    """
    Federated averaging algorithm. Returns the updated global model.

    Parameters:
    ------------
    global_model: torch.nn.Module object;
        Global model.
    local_models: list;
        List of local models.

    Returns:
    ------------
    global_model: torch.nn.Module object;
        Updated global model.
    """
    # update global model parameters here
    state_dicts = [model.state_dict() for model in local_models]
    for key in global_model.track_layers.keys():
        global_model.track_layers[key].weight.data = torch.stack(
            [item[str(key) + ".weight"] for item in state_dicts]
        ).mean(dim=0)
        global_model.track_layers[key].bias.data = torch.stack(
            [item[str(key) + ".bias"] for item in state_dicts]
        ).mean(dim=0)
        # info here - https://discuss.pytorch.org/t/how-to-change-weights-and-bias-nn-module-layers/93065/2
    return global_model

def fedProx(global_model: torch.nn.Module, local_models: List[torch.nn.Module], mu: float) -> torch.nn.Module:
    """
    Federated Proximal algorithm. Returns the updated global model.

    Parameters:
    ------------
    global_model: torch.nn.Module object
        Global model.
    local_models: list
        List of local models.
    mu: float
        Proximal term coefficient.

    Returns:
    ------------
    global_model: torch.nn.Module object
        Updated global model.
    """
    state_dicts = [model.state_dict() for model in local_models]
    global_state_dict = global_model.state_dict()

    for key in global_state_dict.keys():
        
        stacked_tensors = torch.stack([state_dict[key] for state_dict in state_dicts], dim=0) # Stack the parameters from each local model
        averaged_tensor = torch.mean(stacked_tensors, dim=0) # Compute the mean of the stacked parameters
        
        global_state_dict[key] = averaged_tensor

    global_model.load_state_dict(global_state_dict)
    
    # Apply the proximal term adjustment
    for key in global_state_dict.keys():
        global_state_dict[key] -= mu * (global_state_dict[key] - global_model.state_dict()[key])
    
    global_model.load_state_dict(global_state_dict)
    
    return global_model

def fedAdaBoost(global_model: torch.nn.Module, local_models: List[torch.nn.Module], weights: List[float]):
    """
    FeDABoost algorithm. Returns the updated global model.
    
    Parameters:
    ------------
    global_model: torch.nn.Module object
        Global model.
    local_models: list
        List of local models.
    weights: list
        List of weights for each local model.
    
    Returns:
    ------------
    global_model: torch.nn.Module object
        Updated global model.
    """
    state_dicts = [model.state_dict() for model in local_models]
    weights = [round(weight,3) for weight in weights]

    for key in global_model.track_layers.keys():

        stacked_weights = torch.stack(
            [item[str(key) + ".weight"] * weight for item, weight in zip(state_dicts, weights)]
        )

        global_model.track_layers[key].weight.data = stacked_weights.sum(dim=0) / sum(weights)

        stacked_biases = torch.stack(
            [item[str(key) + ".bias"] * weight for item, weight in zip(state_dicts, weights)]
        )

        global_model.track_layers[key].bias.data = stacked_biases.sum(dim=0) / sum(weights)
            
    return global_model