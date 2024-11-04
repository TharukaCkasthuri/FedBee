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
import math
import logging
import numpy as np

from torch.utils.data import DataLoader
from utils import get_device
from datasets.mnist.preprocess import MNISTDataset
from datasets.femnist.preprocess import FEMNISTDataset
from sklearn.metrics import f1_score

class Client:
    """
    Client class for federated learning.

    Parameters:
    ------------
    client_id: str; client id
    train_dataset: torch.utils.data.Dataset object; training dataset
    test_dataset: torch.utils.data.Dataset object; validation dataset
    loss_fn: torch.nn.Module object; loss function
    batch_size: int; batch size
    learning_rate: float; learning rate
    weight_decay: float; weight decay
    local_model: torch.nn.Module object; model
    """

    def __init__(
        self,
        client_id: str,
        train_dataset: object,
        test_dataset: object,
        loss_fn: torch.nn.Module,
        batch_size: int,
        learning_rate: float,
        weight_decay: float,
        local_model: object = None,
    ) -> None:

        self.client_id: str = client_id
        self.loss_fn = loss_fn
        self.batch_size = batch_size

        self.traindl = DataLoader(
            train_dataset, batch_size, shuffle=True, drop_last=True
        )

        self.valdl = DataLoader(test_dataset, 8, shuffle=False, drop_last=True)
        self.optimizer = torch.optim.SGD(
            local_model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        self.device = get_device()
        self.local_model = local_model
        self.local_model = local_model.to(self.device)
        self.train_dataset = train_dataset
        self.datapoints = len(train_dataset)

    def get_num_datapoints(self) -> int:
        """
        Get the number of samples in the training dataset.
        """
        return self.datapoints

    def set_model(self, model_weights) -> None:
        """
        Set the model for the client.

        Parameters:
        ------------
        model_weights: dict; state dictionary of model weights
        """
        self.local_model.load_state_dict(model_weights)

    def get_model(self) -> object:
        """
        Get the model of the client.

        Parameters:
        ------------
        None

        Returns:
        ------------
        model: torch.nn.Module object; model
        """
        return self.local_model

    def train(self, global_round:int, max_local_round:int, threshold:float, patience:int) -> tuple:
        """
        Training the model, using the fedaboost-optima strategy.

        Parameters:
        ------------
        model: torch.nn.Module object; model to be trained.
        loss_fn: torch.nn.Module object; loss function.

        Returns:
        ------------
        model: torch.nn.Module object; trained model.
        """
        previous_loss_avg = float('inf')  
        no_improvement_rounds = 0  

        print(f"Client: {self.client_id} \tTraining...")
        logging.info(f"Client: {self.client_id} \tTraining...")

        for epoch in range(max_local_round):
            print("\n")
            batch_loss = []
            for batch_idx, (x, y) in enumerate(self.traindl):
                x, y = x.to(self.device), y.to(self.device)
                outputs = self.local_model(x)

                if isinstance(self.loss_fn, torch.nn.CrossEntropyLoss) and isinstance(self.train_dataset, FEMNISTDataset):
                    y = y.view(-1)
                elif isinstance(self.train_dataset, MNISTDataset):
                    y = torch.argmax(y, dim=1)
                else:
                    y = y.view(-1, 1)

                loss = self.loss_fn(outputs, y)
                self.local_model.zero_grad()
                loss.backward()
                self.optimizer.step()
    
                batch_loss.append(loss.item())
    
            loss_avg = sum(batch_loss) / len(batch_loss)    
            print(f"Client: {self.client_id} \tEpoch: {epoch + 1} \tAverage Training Loss: {loss_avg} \tGlobal Round: {global_round}")
            logging.info(f"Client: {self.client_id} \tEpoch: {epoch + 1} \tAverage Training Loss: {loss_avg} \tGlobal Round: {global_round}")

            # Dynamic loss reduction evaluation.
            loss_reduction = previous_loss_avg - loss_avg
            if loss_reduction < threshold:
                no_improvement_rounds += 1
                print(f"Loss reduction below threshold ({loss_reduction:.6f}). No improvement rounds: {no_improvement_rounds}")
                logging.info(f"Loss reduction below threshold ({loss_reduction:.6f}). No improvement rounds: {no_improvement_rounds}")
            else:
                pass

            if no_improvement_rounds >= patience:
                print(f"Stopping early at local epoch {epoch + 1} due to no significant improvement.")
                logging.info(f"Stopping early at local epoch {epoch + 1} due to no significant improvement.")
                break

            previous_loss_avg = loss_avg

        return self.local_model

    def evaluate(self) -> tuple:
        """
        Evaluate the model with validation dataset.

        Parameters:
        ------------
        model: torch.nn.Module object; model to be evaluated
        loss_fn: torch.nn.Module object; loss function

        Returns:
        ------------
        loss_avg: float; average loss
        f1_avg: float; average F1 score
        """
        batch_loss = []
        all_preds = []
        all_labels = []

        for _, (x, y) in enumerate(self.valdl):
            x, y = x.to(self.device), y.to(self.device)
            outputs = self.local_model(x)
            if isinstance(self.loss_fn, torch.nn.CrossEntropyLoss) and isinstance(self.train_dataset, FEMNISTDataset):
                y = y.view(-1)
            elif isinstance(self.train_dataset, MNISTDataset):
                y = torch.argmax(y, dim=1)
            else:
                y = y.view(-1, 1)
            
            loss = self.loss_fn(outputs, y)
            batch_loss.append(loss.item())
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
        
        loss_avg = sum(batch_loss) / len(batch_loss)
        f1_avg = f1_score(all_labels, all_preds, average='macro') 
        
        return loss_avg, f1_avg

class BoostingClient(Client):
    """
    Client class for federated learning.
    
    Parameters:
    ------------
    client_id: str; client id
    train_dataset: torch.utils.data.Dataset object; training dataset
    test_dataset: torch.utils.data.Dataset object; validation dataset
    loss_fn: torch.nn.Module object; loss function
    batch_size: int; batch size
    learning_rate: float; learning rate
    weight_decay: float; weight decay
    local_model: torch.nn.Module object; model
    local_round: int; number of local rounds
    """

    def __init__(
        self,
        client_id: str,
        train_dataset: object,
        test_dataset: object,
        loss_fn: torch.nn.Module,
        batch_size: int,
        learning_rate: float,
        weight_decay: float,
        local_model: object = None,
    ) -> None:
        
        super().__init__(
            client_id, 
            train_dataset, 
            test_dataset, 
            loss_fn, 
            batch_size, 
            learning_rate, 
            weight_decay, 
            local_model
        )

        self.num_classes = train_dataset.num_classes()
        self.alpha_range = self.__get_alpha_range(self.num_classes)
        self.eta = 1
        self.error_threshold = 0.5

    def train(self, global_round, max_local_round, threshold=0.01, patience=2, weight:float=1) -> tuple:
        """
        Training the model, using the fedaboost-optima strategy.

        Parameters:
        ------------
        model: torch.nn.Module object; model to be trained.
        loss_fn: torch.nn.Module object; loss function.

        Returns:
        ------------
        model: torch.nn.Module object; trained model.
        """

        alpha = self.get_alpha()
        error_rate = self.__get_error_rate()
        self.weight =self.update_weight(alpha, performance_indicator = (error_rate > self.error_threshold))

        print(f"The client training is boosted by: {self.weight}")

        self.loss_fn.gamma = 1 + math.exp(self.weight)
        self.loss_fn.alpha = 1
        logging.info(f"Client focal loss gamma: {self.loss_fn.gamma}")  
        previous_loss_avg = float('inf')  
        no_improvement_rounds = 0  

        print(f"Client: {self.client_id} \tTraining...")
        logging.info(f"Client: {self.client_id} \tTraining...")

        for epoch in range(max_local_round):
            print("\n")
            batch_loss = []
            for batch_idx, (x, y) in enumerate(self.traindl):
                x, y = x.to(self.device), y.to(self.device)
                outputs = self.local_model(x)

                if isinstance(self.loss_fn, torch.nn.CrossEntropyLoss) and isinstance(self.train_dataset, FEMNISTDataset):
                    y = y.view(-1)
                elif isinstance(self.train_dataset, MNISTDataset):
                    y = torch.argmax(y, dim=1)
                else:
                    y = y.view(-1, 1)

                loss = self.loss_fn(outputs, y)
                self.local_model.zero_grad()
                loss.backward()
                self.optimizer.step()
    
                batch_loss.append(loss.item())
    
            loss_avg = sum(batch_loss) / len(batch_loss)    
            print(f"Client: {self.client_id} \tEpoch: {epoch + 1} \tAverage Training Loss: {loss_avg} \tGlobal Round: {global_round} {self.loss_fn.gamma} {alpha}")
            logging.info(f"Client: {self.client_id} \tEpoch: {epoch + 1} \tAverage Training Loss: {loss_avg} \tGlobal Round: {global_round} {self.loss_fn.gamma} {self.loss_fn.alpha}")

            # Dynamic loss reduction evaluation.
            loss_reduction = previous_loss_avg - loss_avg
            if loss_reduction < threshold:
                no_improvement_rounds += 1
                print(f"Loss reduction below threshold ({loss_reduction:.6f}). No improvement rounds: {no_improvement_rounds}")
                logging.info(f"Loss reduction below threshold ({loss_reduction:.6f}). No improvement rounds: {no_improvement_rounds}")
            else:
                pass

            if no_improvement_rounds >= patience:
                print(f"Stopping early at local epoch {epoch + 1} due to no significant improvement.")
                logging.info(f"Stopping early at local epoch {epoch + 1} due to no significant improvement.")
                break

            previous_loss_avg = loss_avg

        return self.local_model

    def __get_error_rate(self) -> float:
        """
        Evaluate the model on the validation dataset and return the error rate.

        Returns:
        ------------
        error_rate: float; error rate (proportion of incorrect predictions)
        """
        incorrect_preds = 0
        total_samples = 0

        for _, (x, y) in enumerate(self.traindl):
            x, y = x.to(self.device), y.to(self.device)
            outputs = self.local_model(x)

            if isinstance(self.train_dataset, FEMNISTDataset):
                y = y.view(-1)
            elif isinstance(self.train_dataset, MNISTDataset):
                y = torch.argmax(y, dim=1)
            else:
                y = y.view(-1, 1)

            preds = torch.argmax(outputs, dim=1)

            incorrect_preds += (preds != y).sum().item()  
            total_samples += y.size(0)  

        # Error rate as the proportion of incorrect predictions
        error_rate = incorrect_preds / total_samples if total_samples > 0 else 0

        logging.info(f"Client: {self.client_id} \tError Rate: {error_rate}")
        return error_rate
    
    def get_alpha(self) -> float:
        """
        Calculate adjusted weights for client in the FL setting, 
        giving higher weights to clients with lower errors with the global model.

        Parameters:
        - error: Error value for the client validation data with the global model.

        Returns:
        - Adjusted weight (alpha) for the client.
        """
        error_rate = self.__get_error_rate()
        logging.info(f"Client {self.client_id} Error rate: {error_rate}")

        if 0 < error_rate < 1:
            alpha = np.log((1 - error_rate) / (error_rate)) + np.log(10 - 1)
        elif error_rate == 1:
            alpha = np.log((1 - (1-1e-6)) / (1-1e-6)) + np.log(10 - 1)
        elif error_rate == 0:
            alpha = np.log((1 - 1e-6) / (1e-6)) + np.log(10 - 1)
        else:
            raise ValueError("Error value must be in the range [0, 1].")

        return alpha
    
    def __get_alpha_range(self, num_classes=10) -> tuple:
        """
        This function calculates the range of alpha for a given number of classes
        using the calculate_alpha function.
        """
        
        # Calculate alpha for error rates close to 0 and close to 1
        alpha_min = np.log((1 - (1-1e-6)) / (1-1e-6))   # Error rate close to 1
        alpha_max = np.log((1 - 1e-6) / (1e-6)) + np.log(62 - 1)    # Error rate close to 0
        
        return (alpha_min, alpha_max)

    def map_alpha(self, alpha, original_range, target_min=0, target_max=3):
        """
        Maps the alpha value from the original range [original_min, original_max] 
        to the target range [target_min, target_max].
        """

        original_min, original_max = original_range
        original_min = -original_min
        original_max = -original_max
        
        # Linear transformation to map alpha to the target range
        mapped_alpha = ((alpha - original_min) * (target_max - target_min)) / (original_max - original_min) + target_min
        return mapped_alpha
    
    def set_weight(self, weight:float) -> None:
        """
        Set the weight for the client.

        Parameters:
        ------------
        weight: float; weight
        """
        self.weight = weight
        return self.weight

    def update_weight(self, alpha, performance_indicator=1) -> None:
        """
        Update the weights for the client.

        Parameters:
        ------------
        weight: float; weight
        """
        self.weight = self.weight * math.exp(float(self.eta) * float(alpha) * int(performance_indicator))
        return self.weight
