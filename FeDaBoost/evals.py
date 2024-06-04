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
import numpy as np


from sklearn.metrics import mean_squared_error, mean_absolute_error

def evaluate(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
) -> tuple:
    """
    Evaluate the model with validation dataset.Returns the average loss, mean squared error and mean absolute error.

    Parameters:
    -------------
    model: torch.nn.Module object;
        Model to be evaluated.
    dataloader: torch.utils.data.DataLoader object;
        Validation dataset.
    loss_fn: torch.nn.Module object;
        Loss function.

    Returns:
    -------------
    loss: float;
      Average loss.
    mse: float;
        Average mean squared error.
    mae: float;
        Average mean absolute error.

    """
    model.eval()
    loss, mse, mae = [], [], []

    for _, (x, y) in enumerate(dataloader):
        predicts = model(x)
        batch_loss = loss_fn(predicts, y)

        loss.append(batch_loss.item())
        batch_mse = mean_squared_error(y, np.squeeze(predicts.detach().numpy()))
        mse.append(batch_mse)
        batch_mae = mean_absolute_error(y, np.squeeze(predicts.detach().numpy()))
        mae.append(batch_mae)

    return sum(loss) / len(loss), sum(mse) / len(mse), sum(mae) / len(mae)


def evaluate_mae_with_confidence(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    num_bootstrap_samples: int = 250,
    confidence_level: float = 0.95,
) -> tuple:
    """
    Evaluate the model with validation dataset and calculate confidence intervals. Returns the average mean absolute error and confidence intervals.

    Parameters:
    -------------
    model: torch.nn.Module object;
        Model to be evaluated.
    dataloader: torch.utils.data.DataLoader object;
        Validation dataset.
    num_bootstrap_samples: int;
        Number of bootstrap samples.
    confidence_level: float;
        Confidence level. Default is 0.95.

    Returns:
    -------------
    avg_mae: float;
        Average mean absolute error.
    (lower_mae, upper_mae): tuple;
        Lower and upper bounds of the confidence interval for mean absolute error.
    """
    model.eval()
    mae_values = []

    for _, (x, y) in enumerate(dataloader):
        predicts = model(x)
        batch_mae = mean_absolute_error(y, predicts.detach().cpu().numpy())
        mae_values.append(batch_mae)

    avg_mae = np.mean(mae_values)

    bootstrap_mae = []

    for _ in range(num_bootstrap_samples):
        bootstrap_sample_indices = np.random.choice(
            len(dataloader.dataset), size=len(dataloader.dataset), replace=True
        )
        bootstrap_dataloader = torch.utils.data.DataLoader(
            dataloader.dataset,
            batch_size=dataloader.batch_size,
            sampler=torch.utils.data.SubsetRandomSampler(bootstrap_sample_indices),
        )

        mae_values = []

        for _, (x, y) in enumerate(bootstrap_dataloader):
            predicts = model(x)
            batch_mae = mean_absolute_error(y, predicts.detach().cpu().numpy())
            mae_values.append(batch_mae)

        bootstrap_mae.append(np.mean(mae_values))

    # Calculate confidence intervals
    confidence_interval = (1 - confidence_level) / 2
    sorted_mae = np.sort(bootstrap_mae)

    lower_mae = sorted_mae[int(confidence_interval * num_bootstrap_samples)]
    upper_mae = sorted_mae[int((1 - confidence_interval) * num_bootstrap_samples)]

    bootstrap_mae_std = np.std(bootstrap_mae)

    return avg_mae, (lower_mae, upper_mae), bootstrap_mae_std