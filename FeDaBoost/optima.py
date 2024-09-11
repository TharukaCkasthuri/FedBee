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
import os
import pandas as pd

from server import Server

from aggregators import fedAvg, fedProx, weighted_avg
from utils import get_alpha,get_weights, influence_alpha, adjust_local_epochs

from torch.utils.tensorboard import SummaryWriter

class OptimaServer(Server):

    """
    The federated learning server class for fedaboost-optima.

    Parameters:
    ----------------
    rounds: int;
        Number of global rounds
        stratergy: callable;
        Averaging stratergy for federated learning.

    Methods:
    ----------------
    __aggregate(self, weights = []) -> None:
        Aggregate the models of the clients.
    train(self) -> None:
        Train the model using federated fedaboost-optima algorithm.
    
    """

    def __init__(self, rounds: int, strategy: callable) -> None:
        super().__init__(rounds, strategy)

    def __aggregate(self, weights = None) -> None:
        """
        Aggregate the models of the clients.

        Parameters:
        ----------------
        weights: list or None
            List of weights for the weighted average strategy. If None, standard strategies are used.

        Returns:
        ----------------
        model: torch.nn.Module object
            Aggregated model.
        updated: bool
            Indicates whether the global model was updated.
        """

        prev_params = [p.clone() for p in self.global_model.parameters()]
        client_models = [client.get_model() for client in self.client_dict.values()]
        self.global_model = weighted_avg(self.global_model, client_models, weights)
        updated_params = [p.clone() for p in self.global_model.parameters()]
        updated = self._check_model_update(prev_params, updated_params)

        return self.global_model, updated



    def train(self):
        """
        Train the model using federated learning.

        Parameters:
        ----------------
        model: torch.nn.Module object;
            Model to be trained

        Returns:
        ----------------
        model: torch.nn.Module object;
            Trained model
        """
        consecutive_no_update_rounds = 0
        consecutive_loss_change_rounds = 0
        prev_global_loss = 999999999
        stats = []

        weights_dict = {client: (1 / len(self.client_dict)) for client in self.client_dict}
        print(f"Initial Weights: {weights_dict}")

        for round in range(1,self.rounds+1):
            update_status = False
            print(f"\n | Global Training Round : {round} |\n")
            local_loss = {}
            for client in self.client_dict.values():
                prev_client_loss, _ = client.evaluate()
                client.set_model(self.global_model.state_dict())
                updated_client_loss, _ = client.evaluate()
                epoch = adjust_local_epochs(prev_client_loss, updated_client_loss)
                local_loss[client.client_id] = abs(updated_client_loss)#*weights_dict[client.client_id]

                # Training the client model for the specified number of local rounds, using the local data.
                _ = client.train(round, epoch, weights_dict[client.client_id])

                self._receive(client)

            print(f"Local Loss: {local_loss}")

            alphas = get_alpha(local_loss)
            final_alpha = influence_alpha(0.5, alphas)
            weights_dict = get_weights(final_alpha, weights_dict)
            self.global_model, update_status = self.__aggregate(alphas.values())

            print(f"Model Updated: {update_status}")

            if consecutive_loss_change_rounds == 3:
                print("The global model parameters have not been updated for 3 consecutive rounds, so the training has converged.")
                break

            self._broadcast(self.global_model)
            
            if not update_status:
                consecutive_no_update_rounds += 1
                print("The global model parameters have not been updated, so the training has converged.")
            else:
                consecutive_no_update_rounds = 0

            if consecutive_no_update_rounds == 3:
                print("The global model parameters have not been updated for 5 consecutive rounds, so the training has converged.")
                break

        if self.stratergy == "fedaboost":
            file_name = "stats/fedaboost_stats_mnist.csv"
            self._save_stats(stats, file_name)
            print(f"Saved the training statistics to {file_name}")

        # Close the TensorBoard writer
        self.writer.close()

        return self.global_model
    
