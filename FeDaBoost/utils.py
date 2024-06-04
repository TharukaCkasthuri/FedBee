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
Published in: [Journal/Conference Name]
"""

import torch
import itertools

import numpy as np
import pandas as pd

from torch.utils.data import Dataset, DataLoader


def get_device() -> torch.device:
    """
    Returns the device to be used for training.

    Parameters:
    --------
    None

    Returns:
    --------
    device: torch.device object
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_all_possible_pairs(client_ids: list) -> list:
    """
    Returns all possible pairs of client ids.

    Parameters:
    --------
    client_ids: list; list of client ids

    Returns:
    --------
    pairs: list; list of all possible pairs of client ids
    """

    pairs = list(itertools.combinations(client_ids, 2))

    return pairs

class Client:
    """
    Client class for federated learning.

    Parameters:
    ------------
    client_id: str; client id
    train_dataset: torch.utils.data.Dataset object; training dataset
    test_dataset: torch.utils.data.Dataset object; validation dataset
    batch_size: int; batch size
    """

    def __init__(
        self,
        client_id: str,
        train_dataset: object,
        test_dataset: object,
        batch_size: int,
    ) -> None:
        self.client_id: str = client_id
        self.train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
        self.test_dataloader = DataLoader(test_dataset, batch_size, shuffle=True)

    def train(self, model, loss_fn, optimizer, epoch) -> tuple:
        """
        Training the model.

        Parameters:
        ------------
        model: torch.nn.Module object; model to be trained
        loss_fn: torch.nn.Module object; loss function
        optimizer: torch.optim object; optimizer
        epoch: int; epoch number

        Returns:
        ------------
        model: torch.nn.Module object; trained model
        loss_avg: float; average loss
        """
        batch_loss = []
        for batch_idx, (x, y) in enumerate(self.train_dataloader):
            outputs = model(x)
            loss = loss_fn(outputs, y)
            model.zero_grad()
            loss.backward()
            optimizer.step()
            batch_loss.append(loss.item())

        loss_avg = sum(batch_loss) / len(batch_loss)
        print(f"Epoch: {epoch + 1} \tClient ID: {self.client_id} \tLoss: {loss.item():.6f}")
        return model, loss_avg

    def eval(self, model, loss_fn) -> float:
        """
        Evaluate the model with validation dataset.

        Parameters:
        ------------
        model: torch.nn.Module object; model to be evaluated
        loss_fn: torch.nn.Module object; loss function

        Returns:
        ------------
        loss_avg: float; average loss
        """
        batch_loss = []
        for _, (x, y) in enumerate(self.test_dataloader):
            outputs = model(x)
            loss = loss_fn(outputs, y)
            batch_loss.append(loss.item())
        loss_avg = sum(batch_loss) / len(batch_loss)

        return loss_avg


def min_max_normalization(data, new_min=0, new_max=1):
    """
    Min-max normalization for a list or numpy array.

    Parameters:
    - data: List or numpy array of numeric values.
    - new_min: Minimum value of the normalized range.
    - new_max: Maximum value of the normalized range.

    Returns:
    - Numpy array of normalized values.
    """
    # Convert the data to a numpy array if it's not already
    data = np.array(data)

    # Calculate the min and max of the original data
    original_min = np.min(data)
    original_max = np.max(data)

    # Perform min-max normalization
    normalized_data = new_min + (data - original_min) * (1 - 0) / (original_max - original_min)

    return normalized_data


def calculate_weights(init_weights, top_eigens):
    """
    Calculate normalized weights for a list of weak classifiers in AdaBoost.

    Parameters:
    - init_weights: Initial weights (numpy array or list).
    - top_eigens: List of top eigenvalues from weak classifiers.

    Returns:
    - List of normalized weights for each weak classifier.
    """
    top_eigens = [w / np.sum(top_eigens) for w in top_eigens]
    num_classifiers = len(init_weights)
    
    weights = np.array(init_weights)  # Initialize weights

    alpha = []
    for i in range(num_classifiers):
        # Avoid division by zero
        if top_eigens[i] != 0 and top_eigens[i]!=1:
            alpha.append(0.5 * np.log((1 - top_eigens[i]) / top_eigens[i]))
        elif (top_eigens[i]==1):
            alpha.append(0.5 * np.log((1 - 0.99999999) / 0.99999999))
        else:
            alpha.append(0)
    
    weights = [round(w * np.exp(-a),5) for w, a in zip(weights, alpha)]
    weights = [w / np.sum(weights) for w in weights]

    return weights