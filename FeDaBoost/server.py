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
import random
import os
import logging

import pandas as pd
import numpy as np

from clients import Client
from aggregators import fedAvg, fedProx, weighted_avg
from utils import get_alpha,get_weights, influence_alpha, adjust_local_epochs, get_device, z_scores

from datasets.femnist.preprocess import FEMNISTDataset
from datasets.mnist.preprocess import MNISTDataset

from sklearn.metrics import f1_score

from torch.utils.tensorboard import SummaryWriter

class Server:

    """
    The federated learning server class. 
    
    Parameters:
    ----------------
    rounds: int;
        Number of global rounds
    stratergy: callable;
        Averaging stratergy for federated learning.
    
    Methods:
    ----------------
    init_model(self, model: torch.nn.Module) -> None:
        Initialize the model for federated learning.
    connect_client(self, client: Client) -> None:
        Add a client for federated learning setup.
    __aggregate(self, weights = []) -> None:
        Aggregate the models of the clients.
    _broadcast(self, model: torch.nn.Module) -> None:
        Broadcast the model to the clients.
    _receive(self, client:callable) -> list:
        Receive the models from the clients.
    train(self) -> None:
        Train the model using federated learning.
    _collect_stats(self, local_loss: list, weights: list) -> dict:
        Collect training statistics.
    _save_stats(self, stats: list, path: str) -> None:
        Save training statistics to a CSV file.
    _check_model_update(self, prev_params: list, updated_params: list) -> bool:
        Check if the model parameters have been updated.
    """

    def __init__(self,rounds:int, stratergy:callable, checkpt_path:str=None, log_dir:str = 'runs') -> None:
        self.rounds = rounds
        self.stratergy = stratergy
        self.client_dict = {}
        self.checkpoint_path = checkpt_path
        print(f"Checkpoint Path: {self.checkpoint_path}")

        # Initialize TensorBoard writer
        self.writer = SummaryWriter(log_dir=log_dir)


    def init_model(self, model: torch.nn.Module) -> None:
        """
        Initialize the model for federated learning.

        Parameters:
        ----------------
        model: torch.nn.Module object;
            Model to be trained

        Returns:
        ----------------
        None
        """
        self.global_model = model
        self.global_model.train()
        self.global_model.to("mps:0")


    def connect_client(self, client: Client) -> None:
        """
        Add a client for federated learning setup.

        Parameters:
        ----------------
        client_id: str;
            Client id

        Returns:
        ----------------
        None
        """
        
        client_id = client.client_id
        self.client_dict[client_id] = client


    def sample_clients(self, num_clients: int) -> dict:
        """
        Sample clients from the client dictionary.

        Parameters:
        ----------------
        num_clients: int;
            Number of clients to sample

        Returns:
        ----------------
        list
            Dict of sampled clients
        """
        sampled_client_ids = np.random.choice(list(self.client_dict.keys()), num_clients, replace=False)
        sampled_clients = {client_id: self.client_dict[client_id] for client_id in sampled_client_ids}
  
        return sampled_clients


    def _aggregate(self,trained_clients, weights = None) -> None:
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
        client_models = [client.get_model() for client in trained_clients.values()]

        aggregation_functions = {
        "fedavg": weighted_avg,
        "fedprox": fedProx,
        "fedaboost-optima": weighted_avg,
        "fedaboost-ranker": weighted_avg,
        "fedaboost-concord": weighted_avg,
        }

        aggregation_function = aggregation_functions.get(self.stratergy)

        if aggregation_function is None:
            raise ValueError(f"Unsupported aggregation strategy: {self.stratergy}")
        
        self.global_model = aggregation_function(self.global_model, client_models, weights)

        updated_params = [p.clone() for p in self.global_model.parameters()]

        updated = self._check_model_update(prev_params, updated_params)

        return self.global_model, updated


    def _broadcast(self, model: torch.nn.Module) -> None:
        """
        Broadcast the model to the clients.

        Parameters:
        ----------------
        model: torch.nn.Module object;
            Model to be broadcasted
        """
        model_state_dict = model.state_dict()
        for client_id, client in self.client_dict.items():
            client.set_model(model_state_dict)
            self.client_dict[client_id] = client
            print(f"Broadcasted model to client {client.client_id}")


    def _receive(self, client:callable) -> list:
        """
        Receive the models from the clients.

        Returns:
        ----------------
        models: list;
            List of models
        """
        print(f"Received model from client {client.client_id}")
        self.client_dict[client.client_id] = client


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
        stats = []

        for round in range(1,self.rounds+1):
            update_status = False
            print(f"\n | Global Training Round : {round} |\n")
            logging.info(f"\n | Global Training Round : {round} |\n")
            loss_item = 0
            num_data_points = {}

            train_clients = self.sample_clients(random.randint(10, 20))
            logging.info(f"Selected Clients: {train_clients.keys()}")

            for client in train_clients.values():
                client.set_model(self.global_model.state_dict())
                loss, f1 = client.evaluate()
                loss_item += loss
                _ = client.train(round,10)
                num_data_points[client.client_id] = client.get_num_datapoints()
                self._receive(client)

            total_data_points = sum(num_data_points[client.client_id] for client in train_clients.values())
            weights = [num_data_points[client.client_id] / total_data_points for client in train_clients.values()]

            self.global_model, update_status = self._aggregate(train_clients, weights=weights)

            self.save_checkpt(self.global_model, f"{self.checkpoint_path}/checkpoints/ckpt_{round}.pt")

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
        
        # Close the TensorBoard writer
        self.writer.close()

        return self.global_model
    
    def __check_convergence(self, global_loss: float, prev_global_loss: float) -> bool:
        """
        Check if the training has converged.

        Parameters:
        ----------------
        global_loss: float
            Current global loss
        prev_global_loss: float
            Previous global loss

        Returns:
        ----------------
        bool
            True if converged, False otherwise
        """
        if abs(global_loss - prev_global_loss) < 0.0001:
            self.consecutive_loss_change_rounds += 1
            if self.consecutive_loss_change_rounds == 3:
                print("The global model parameters have not changed significantly for 3 consecutive rounds, training has converged.")
                return True
        else:
            self.consecutive_loss_change_rounds = 0
        return False
    
    def _collect_stats(self, local_loss: list, weights: list) -> dict:
        """
        Collect training statistics.

        Parameters:
        ----------------
        local_loss: list
            List of client losses
        weights: list
            List of weights for clients

        Returns:
        ----------------
        dict
            Dictionary containing training statistics
        """
        stats = {f"{client.client_id}-loss": local_loss[i] for i, client in enumerate(self.client_dict.values())}
        stats.update({f"{client.client_id}-weight": weights[i] for i, client in enumerate(self.client_dict.values())})
        return stats
    
    def _save_stats(self, stats: list, path: str) -> None:
        """
        Save training statistics to a CSV file.

        Parameters:
        ----------------
        stats: list
            List of training statistics
        path: str
            Path to save the CSV file

        Returns:
        ----------------
        None
        """
        stats_df = pd.DataFrame(stats)
        directory = os.path.dirname(path)
        
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        stats_df.to_csv(path, index=False)

    def _check_model_update(self, prev_params: list, updated_params: list) -> bool:
        """
        Check if the model parameters have been updated.

        Parameters:
        ----------------
        prev_params: list
            List of previous model parameters
        updated_params: list
            List of updated model parameters

        Returns:
        ----------------
        bool
            True if updated, False otherwise
        """
        for prev_param, updated_param in zip(prev_params, updated_params):
            if not torch.equal(prev_param, updated_param):
                return True
        return False
    
    def save_checkpt(self, checkpoint: torch.nn.Module, ckptpath: str) -> None:
        """
        Saving the checkpoints.

        Parameters:
        ----------------
        checkpoint:
            Model at a specific checkpoint.
        ckptpath: str;
            Path to save the checkpoint. Default is None.

        Returns:
        ----------------
        None
        """
        if os.path.exists(ckptpath):
            torch.save(
                checkpoint.state_dict(),
                ckptpath,
            )
        else:
            os.makedirs(os.path.dirname(ckptpath), exist_ok=True)
            torch.save(
                checkpoint.state_dict(),
                ckptpath,
            )

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

    def __init__(self, rounds: int, strategy: callable, checkpt_path:str=None,  log_dir:str = 'runs', max_local_round = 10) -> None:
        super().__init__(rounds, strategy, checkpt_path, log_dir)
        self.max_local_round = max_local_round
        logging.info("FedaBoost-Optima Boosting Server Initialized")


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

        weights_dict = {client: (1 / len(self.client_dict)) for client in self.client_dict}
        boost_dict = {client: 0 for client in self.client_dict}

        logging.info(f"Initial Weights: {weights_dict}")
        for round in range(1,self.rounds+1):
            update_status = False
            print(f"\n | Global Training Round : {round} |\n")
            logging.info(f"\n | Global Training Round : {round} |\n")
            local_err = {}

            for client in self.client_dict.values():
                client.set_model(self.global_model.state_dict())
                error_rate = client.get_error_rate()
                local_err[client.client_id] = error_rate
                logging.info(f"Client {client.client_id} Error rate: {error_rate}")

                _ = client.train(global_round= round, max_local_round = self.max_local_round, weight = boost_dict[client.client_id])
                self._receive(client)

            print(f"Local Error Rate: {local_err}")
            logging.info(f"Local Error Rate: {local_err}")
            alphas = get_alpha(local_err)
            logging.info(f"Initial Alpha: {alphas}")
            final_alpha = influence_alpha(0.5, alphas)
            logging.info(f"Final Alpha: {final_alpha}")
            weights_dict = get_weights(final_alpha, weights_dict)
            logging.info(f"Updated Weights using final alpha: {weights_dict}")
            boost_dict = z_scores(weights_dict)
            logging.info(f"Boosting values - z-scores of weights: {boost_dict}")

            self.global_model, update_status = self.__aggregate(final_alpha.values())

            self.save_checkpt(self.global_model, f"{self.checkpoint_path}/checkpoints/ckpt_{round}.pt")

            print(f"Model Updated: {update_status}")
            logging.info(f"Model Updated Successfully using Fedaboost-Optima: {update_status}")

            self._broadcast(self.global_model)
            
            if not update_status:
                consecutive_no_update_rounds += 1
                print("The global model parameters have not been updated, so the training has converged.")
                logging.info("The global model parameters have not been updated, so the training has converged. Might be some issue with the model aggregation.")
            else:
                consecutive_no_update_rounds = 0

            if consecutive_no_update_rounds == 3:
                print("The global model parameters have not been updated for 5 consecutive rounds, so the training has converged.")
                logging.info("The global model parameters have not been updated for 3 consecutive rounds. Hence, stopping the training.")
                break

        return self.global_model
    
class RankServer(Server):
    """
    The federated learning server class for fedaboost-ranker.

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
        Train the model using federated fedaboost-ranker algorithm.
    
    """

    def __init__(self, rounds: int, strategy: callable, checkpt_path:str=None,  log_dir:str = 'runs', test_dataset = "None", loss_fn:torch.nn.Module=None) -> None:
        super().__init__(rounds, strategy, checkpt_path, log_dir)
        self.test_dataset = test_dataset
        self.test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True)
        self.device = get_device()
        self.loss_fn = loss_fn

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
        client_losses = {}

        for client in self.client_dict.values():
            client_losses[client.client_id] = self.evaluate(client.get_model())

        sorted_clients = sorted(client_losses.items(), key=lambda x: x[1])
        client_ranks = {client_id: rank+1 for rank, (client_id, _) in enumerate(sorted_clients)}
        alphas = get_alpha(client_losses)
        final_alpha = {client_id: alphas[client_id] / client_ranks[client_id] for client_id in client_losses}

        weights = get_weights(final_alpha, weights)


        self.global_model = weighted_avg(self.global_model, client_models, alphas.values())

        updated_params = [p.clone() for p in self.global_model.parameters()]
        updated = self._check_model_update(prev_params, updated_params)

        return self.global_model, weights, updated

    
    
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

            for client in self.client_dict.values():
                client.set_model(self.global_model.state_dict())
                updated_client_loss, _ = client.evaluate()
                
                # Training the client model for the specified number of local rounds, using the local data.
                _ = client.train(round, 5, weights_dict[client.client_id])
                self._receive(client)


            self.global_model, weights_dict, update_status = self.__aggregate(weights_dict)

            self.save_checkpt(self.global_model, f"{self.checkpoint_path}/checkpoints/ckpt_{round}.pt")


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

    

    def evaluate(self, model: torch.nn.Module) -> tuple:
        """
        Evaluate the receieved local model using the test dataset.

        Parameters:
        ----------------
        model: torch.nn.Module object;
            Model to be evaluated

        Returns:
        ----------------
        loss: float
            Loss of the model
        """
        batch_loss = []

        for _, (x, y) in enumerate(self.test_dataloader):
            x, y = x.to(self.device), y.to(self.device)
            x = x.clone().detach().float()  
            y = y.clone().detach().long()   

            outputs = model(x)
            if isinstance(self.loss_fn, torch.nn.CrossEntropyLoss) and isinstance(self.test_dataset, FEMNISTDataset):
                y = y.view(-1)
            elif isinstance(self.test_dataset, MNISTDataset):
                y = torch.argmax(y, dim=1)
            else:
                y = y.view(-1, 1)
            
            loss = self.loss_fn(outputs, y)
            batch_loss.append(loss.item())            
        
        loss_avg = sum(batch_loss) / len(batch_loss)
        return loss_avg 