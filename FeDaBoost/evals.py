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


from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error
from datasets.femnist.preprocess import FEMNISTDataset
from datasets.mnist.preprocess import MNISTDataset

def evaluate(
    model: torch.nn.Module,
    test_data: torch.utils.data.DataLoader,
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
    testdl = DataLoader(test_data, 32, shuffle=False, drop_last=True)
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

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def evaluate_classification(
    model: torch.nn.Module,
    test_data: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
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
    testdl = DataLoader(test_data, batch_size=32, shuffle=False, drop_last=True)
    
    batch_loss = []
    all_targets = []
    all_preds = []
    
    with torch.no_grad():
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


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable



class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class FocalCosineLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, xent=.1):
        super(FocalCosineLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

        self.xent = xent

        self.y = torch.Tensor([1]).cuda()

    def forward(self, input, target, reduction="mean"):
        cosine_loss = F.cosine_embedding_loss(input, F.one_hot(target, num_classes=input.size(-1)), self.y, reduction=reduction)

        cent_loss = F.cross_entropy(F.normalize(input), target, reduce=False)
        pt = torch.exp(-cent_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * cent_loss

        if reduction == "mean":
            focal_loss = torch.mean(focal_loss)

        return cosine_loss + self.xent * focal_loss
