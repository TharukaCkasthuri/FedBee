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
import torch.nn as nn
import torch.nn.functional as F


from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datasets.femnist.preprocess import FEMNISTDataset
from datasets.mnist.preprocess import MNISTDataset

def evaluate(
    model: torch.nn.Module,
    test_data: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    batch_size: int = 16,
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
    testdl = DataLoader(test_data, batch_size, shuffle=False, drop_last=True)
    batch_loss = []
    for _, (x, y) in enumerate(testdl):
        outputs = model(x)
        if isinstance(loss_fn, torch.nn.CrossEntropyLoss) and isinstance(test_data, FEMNISTDataset):
            y = y.view(-1)
        elif isinstance(test_data, MNISTDataset):
            y = torch.argmax(y, dim=1)
        else:
            y = y.view(-1, 1)
        loss = loss_fn(outputs, y)
        batch_loss.append(loss.item())
    
    loss = np.mean(batch_loss)
    return loss

def evaluate_classification(
    model: torch.nn.Module,
    test_data: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    bath_size: int = 16,
) -> tuple:
    """
    Evaluate the model with validation dataset. Returns the average loss, accuracy, precision, recall, and F1 score.

    Parameters:
    -------------
    model: torch.nn.Module object;
        Model to be evaluated.
    test_data: torch.utils.data.DataLoader object;
        Validation dataset.
    loss_fn: torch.nn.Module object;
        Loss function, such as FocalLoss.

    Returns:
    -------------
    avg_loss: float;
      Average loss.
    accuracy: float;
        Accuracy of the model.
    precision: float;
        Precision of the model.
    recall: float;
        Recall of the model.
    f1_score: float;
        F1 score of the model.
    """
    model.eval()
    testdl = DataLoader(test_data, batch_size=bath_size, shuffle=False, drop_last=True)
    
    batch_loss = []
    all_targets = []
    all_preds = []
    
    for x, y in testdl:
        outputs = model(x)
        
        # Assuming outputs are logits, apply softmax for prediction probabilities
        y_pred = torch.softmax(outputs, dim=1)
        
        # Get predictions and convert targets to appropriate format
        _, predicted_classes = torch.max(y_pred, 1)
        if isinstance(y, torch.Tensor) and y.dim() == 1:
            y_true = y
        else:
            y_true = torch.argmax(y, dim=1)
        
        loss = loss_fn(y_pred, torch.nn.functional.one_hot(y_true, num_classes=y_pred.size(1)).float())
        batch_loss.append(loss.item())
        
        all_targets.append(y_true)
        all_preds.append(predicted_classes)
    
    # Concatenate all targets and predictions
    all_targets = torch.cat(all_targets).cpu().numpy()
    all_preds = torch.cat(all_preds).cpu().numpy()
    
    # Calculate average loss
    avg_loss = np.mean(batch_loss)
    
    # Calculate classification metrics
    accuracy = accuracy_score(all_targets, all_preds)
    precision, recall, f1_score, _ = precision_recall_fscore_support(all_targets, all_preds, average='weighted')
    
    return avg_loss, accuracy, precision, recall, f1_score


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

class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        """
        Focal Loss implementation.

        Parameters:
        ------------
        alpha: float; balancing factor for class imbalance (default=1)
        gamma: float; focusing parameter to adjust the rate at which easy examples are down-weighted (default=2)
        reduction: str; reduction method to apply to output ('mean', 'sum', or 'none')
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        if inputs.dim() > 2:
            inputs = inputs.view(inputs.size(0), inputs.size(1), -1)  # N,C,H,W -> N,C,H*W
            inputs = inputs.permute(0, 2, 1)  # N,C,H*W -> N,H*W,C
            inputs = inputs.contiguous().view(-1, inputs.size(-1))  # N,H*W,C -> N*H*W,C
        targets = targets.view(-1)

        log_pt = F.log_softmax(inputs, dim=-1)
        pt = torch.exp(log_pt)  # Get probabilities
        log_pt = log_pt.gather(1, targets.view(-1, 1)).squeeze()  # Select log probabilities of true class
        pt = pt.gather(1, targets.view(-1, 1)).squeeze()  # Select probabilities of true class

        focal_loss = -self.alpha * (1 - pt) ** self.gamma * log_pt

        # Apply reduction method
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
        

class HybridLoss(nn.Module):
    def __init__(self, focal_alpha=1, focal_gamma=2, focal_weight=0.5):
        """
        A hybrid loss combining Cross-Entropy and Focal Loss.

        Parameters:
        ------------
        focal_alpha: float; alpha parameter for Focal Loss (default=1)
        focal_gamma: float; gamma parameter for Focal Loss (default=2)
        focal_weight: float; weighting factor to balance Cross-Entropy and Focal Loss (default=0.5)
        """
        super(HybridLoss, self).__init__()
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.focal_weight = focal_weight

    def focal_loss(self, inputs, targets):
        log_pt = F.log_softmax(inputs, dim=-1)
        pt = torch.exp(log_pt)
        log_pt = log_pt.gather(1, targets.view(-1, 1)).squeeze()
        pt = pt.gather(1, targets.view(-1, 1)).squeeze()
        focal_loss = -self.focal_alpha * (1 - pt) ** self.focal_gamma * log_pt
        return focal_loss.mean()

    def forward(self, inputs, targets):
        ce_loss = self.cross_entropy_loss(inputs, targets)
        focal_loss = self.focal_loss(inputs, targets)
        hybrid_loss = ce_loss * (1 - self.focal_weight) + focal_loss * self.focal_weight
        return hybrid_loss