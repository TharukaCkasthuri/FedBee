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

import os
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
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def get_client_ids(folder_path):
    client_ids = []
    try:
        files = os.listdir(folder_path)
        for file in files:
            if file.endswith('.pt'):
                client_id = file.split('.')[0]
                client_ids.append(client_id)
        return client_ids
    except Exception as e:
        raise e
    
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


def get_alpha_emp(error_fun: list, num_classes: int = 10, emphasis_factor: float = 3.0):
    """
    Calculate adjusted weights for a list of weak classifiers in AdaBoost, 
    giving higher weights to classifiers with lower errors.

    Parameters:
    - error_fun: List of errors for each weak classifier.
    - num_classes: Number of classes (default is 10).
    - emphasis_factor: A factor to increase the weight given to better-performing classifiers (default is 2.0).

    Returns:
    - List of adjusted weights for each weak classifier.
    """
    # Normalize the errors
    errs = [w / np.sum(error_fun) for w in error_fun]
    num_classifiers = len(errs)

    alpha = []
    for i in range(num_classifiers):
        # Avoid division by zero and log of zero
        if errs[i] != 0 and errs[i] != 1:
            # Increase the weight of better-performing classifiers using the emphasis factor
            alpha_value = emphasis_factor * (0.5 * np.log((1 - errs[i]) / errs[i]) + np.log(num_classes - 1))
        elif errs[i] == 1:
            alpha_value = emphasis_factor * (0.5 * np.log((1 - 0.99999999) / 0.99999999) + np.log(num_classes - 1))
        else:
            alpha_value = 0
        alpha.append(alpha_value)
    
    # Normalize the alpha values to make them sum to 1
    alpha = [w / np.sum(alpha) for w in alpha]

    return alpha


def get_alpha(error_fun: dict, num_classes: int = 10):
    """
    Calculate adjusted weights for a set of weak classifiers in AdaBoost, 
    giving higher weights to classifiers with lower errors.

    Parameters:
    - error_fun: Dictionary where keys are classifier identifiers and values are the errors for each weak classifier.
    - num_classes: Number of classes (default is 10).

    Returns:
    - Dictionary with classifier identifiers as keys and their adjusted weights (alpha) as values.
    """
    # Extract errors from the dictionary and normalize them
    errs = {key: error / np.sum(list(error_fun.values())) for key, error in error_fun.items()}

    # Initialize the dictionary to hold the alpha values
    alpha = {}
    
    # Calculate the alpha values
    for key, error in errs.items():
        if error != 0 and error != 1:
            alpha_value = 0.5 * np.log((1 - error) / error) + np.log(num_classes - 1)
        elif error == 1:
            alpha_value = 0.5 * np.log((1 - 0.99999999) / 0.99999999) + np.log(num_classes - 1)
        else:
            alpha_value = 0
        alpha[key] = alpha_value
    
    # Normalize alpha values
    alpha_sum = np.sum(list(alpha.values()))
    #alpha = {key: value / alpha_sum for key, value in alpha.items()}
    return alpha


def get_weights(alpha: dict, prev_weights: dict):
    """
    Calculate adjusted weights for a set of weak classifiers in AdaBoost, 
    giving higher weights to classifiers with lower errors, and adjusting 
    for previous weights.

    Parameters:
    - alpha: Dictionary of alpha values for each weak classifier.
    - prev_weights: Dictionary of previous weights for each classifier.

    Returns:
    - Dictionary of updated weights for each weak classifier.
    """
    new_weights = {}
    for key in prev_weights.keys():
        new_weights[key] = prev_weights[key] * np.exp(-alpha[key])

    # Normalize the weights
    total_weight = np.sum(list(new_weights.values()))
    new_weights = {key: value / total_weight for key, value in new_weights.items()}

    return new_weights


def influence_alpha(lambda_param:float, alphas:dict)->dict:
    """
    Calculate the influence of alpha values in AdaBoost based on local and global performance.

    Parameters:
    - lambda_param: Influence factor (lambda), a value between 0 and 1.
    - alphas: Dictionary of initial alpha values, {client_id: alpha_j_initial}.

    Returns:
    - A dictionary with the adjusted alpha values, {client_id: adjusted_alpha_j}.
    """
    k = len(alphas)
  
    final_alphas = {}  
    
    for client, alpha_initial in alphas.items():
        other_alphas_avg = sum(alpha for other_client, alpha in alphas.items() if other_client != client) / (k - 1)
        
        adjusted_alpha = lambda_param * alpha_initial + (1 - lambda_param) * other_alphas_avg
        final_alphas[client] = adjusted_alpha
    
    return final_alphas
    

def adjust_epochs(client_weights, e_base=5, beta=10):
    """
    Adjust local training epochs for each client based on their weights.
    
    Parameters:
    - client_weights: A dictionary with client IDs as keys and weights as values, {client_id: weight}.
    - E_base: Minimum number of training epochs for any client.
    - beta: Scaling factor to control how much weight influences the epoch count.
    
    Returns:
    - A dictionary with adjusted epochs per client, {client_id: adjusted_epochs}.
    """
    # Normalize weights to ensure they sum to 1
    total_weight = sum(client_weights.values())
    normalized_weights = {client: weight / total_weight for client, weight in client_weights.items()}

    adjusted_epochs = {}
    for client, weight in normalized_weights.items():
        calculated_epochs = e_base + beta * weight
        epochs = max(1, int(calculated_epochs))  
        print(f"Client: {client}, Weight: {weight}, Calculated Epochs: {calculated_epochs}, Adjusted Epochs: {epochs}")
        adjusted_epochs[client] = epochs
    
    return adjusted_epochs


def adjust_local_epochs(current_loss, previous_loss, e_base=5, gamma=20):
    """
    Adjust the number of local training epochs for all clients based on the loss reduction.

    Parameters:
    - current_loss: The current global validation loss.
    - previous_loss: The previous global validation loss.
    - e_base: Minimum number of training epochs for any client.
    - gamma: Scaling factor to control how much the loss reduction influences the epoch count.

    Returns:
    - Adjusted number of training epochs for each client.
    """
    print(f"Previous Loss: {previous_loss}, Current Loss: {current_loss}")
    loss_reduction = previous_loss - current_loss  # Positive if loss has decreased, negative if increased

    if loss_reduction > 0:  # If there is a positive loss reduction (improvement)
        # Decrease local epochs to avoid overfitting
        adjusted_epochs = max(1, int(e_base - gamma * loss_reduction))
    else:  # If loss reduction is zero or negative (no improvement or deterioration)
        # Increase local epochs to allow more training
        adjusted_epochs = e_base + abs(int(gamma * loss_reduction))
    print(f"Loss Reduction: {loss_reduction}, Adjusted Epochs: {adjusted_epochs}")
    return adjusted_epochs

def z_scores(weight_dict):
    """
    Calculate the z-scores for all values in a dictionary of weights.

    :param weight_dict: Dictionary where keys are clients and values are weights
    :return: Dictionary of clients and their corresponding z-scores
    """
    values = np.array(list(weight_dict.values()))
    mean = np.mean(values)
    std_dev = np.std(values)
    
    # Handle the case where standard deviation is zero
    if std_dev == 0:
        raise ValueError("Standard deviation cannot be zero.")
    
    z_scores = {key: (value - mean) / std_dev for key, value in weight_dict.items()}
    
    return z_scores



