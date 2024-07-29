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
import flwr as fl

from torch.utils.data import DataLoader
from utils import get_device
from datasets.mnist.preprocess import MNISTDataset
from datasets.femnist.preprocess import FEMNISTDataset

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
        loss_fn: torch.nn.Module,
        batch_size: int,
        learning_rate: float,
        weight_decay: float,
        local_rounds: int,
        local_model: object = None,
    ) -> None:

        self.client_id: str = client_id
        self.loss_fn = loss_fn
        self.batch_size = batch_size
        self.local_round = local_rounds

        self.traindl = DataLoader(
            train_dataset, batch_size, shuffle=True, drop_last=True
        )
        self.valdl = DataLoader(test_dataset, batch_size, shuffle=True, drop_last=True)
        self.optimizer = torch.optim.SGD(
            local_model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        self.device = get_device()
        self.local_model = local_model
        self.local_model = local_model.to(self.device)
        self.train_dataset = train_dataset

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

    def train(self, global_round) -> tuple:
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
        train_losses = []
        for epoch in range(self.local_round):
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
            train_losses.append(loss_avg)
    
            print(
                f"Client: {self.client_id} \tEpoch: {epoch + 1} \tAverage Training Loss: {loss_avg} \tGlobal Round: {global_round}"
            )
        return self.local_model, train_losses 

    def evaluate(self) -> float:
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
        loss_avg = sum(batch_loss) / len(batch_loss)

        return loss_avg

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, client: Client):
        self.client = client

    def get_parameters(self):
        return [val.cpu().numpy() for val in self.client.get_model().parameters()]

    def set_parameters(self, parameters):
        params_dict = zip(self.client.get_model().state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.client.get_model().load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        loss_fn = torch.nn.CrossEntropyLoss()
        model, train_losses = self.client.train(loss_fn, epochs=config["epochs"])
        return self.get_parameters(), len(self.client.traindl.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = self.client.eval(loss_fn)
        return float(loss), len(self.client.valdl.dataset), {}