import torch
import os
import pandas as pd

from clients import Client
from aggregators import fedAvg, fedProx, weighted_avg
from utils import calculate_weights

class Server:
    def __init__(self,rounds:int, stratergy:callable) -> None:
        self.rounds = rounds
        self.stratergy = stratergy
        self.client_dict = {}

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
        

    def __aggregate(self, weights = []) -> None:
        """
        Aggregate the models of the clients.

        Parameters:
        ----------------
        models: list;
            List of models

        Returns:
        ----------------
        model: torch.nn.Module object;
            Aggregated model
        """
        client_models = [client.get_model() for client in self.client_dict.values()]
        if self.stratergy == "fedavg":
            self.global_model = fedAvg(self.global_model, client_models)
        elif self.stratergy == "fedprox":
            self.global_model = fedProx(self.global_model, client_models)
        elif self.stratergy == "fedaboost":
            self.global_model = weighted_avg(self.global_model, client_models, weights)
        return self.global_model


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
        prev_global_loss = 999999999
        stats = []

        if self.stratergy == "fedaboost":
            weights = [(1/len(self.client_dict)) for client in self.client_dict]

        for round in range(1,self.rounds+1):
            updated = False
            print(f"\n | Global Training Round : {round+1} |\n")

            local_loss = []
            for client in self.client_dict.values():
                client.set_model(self.global_model.state_dict())
                
                client_loss = client.evaluate()
                local_loss.append(client_loss)

                client_model, client_loss = client.train(round)
                self.__receive(client)

            prev_params = [p.clone() for p in self.global_model.parameters()]

            if self.stratergy == "fedaboost":
                self.global_model = self.__aggregate(weights)
            else:
                self.global_model = self.__aggregate()

            updated_params = [p.clone() for p in self.global_model.parameters()]

            updated = self.__check_model_update(prev_params, updated_params)
            print(f"Model Updated: {updated}")

            global_loss = sum(local_loss) / len(local_loss)
            print(f"Global Loss : {global_loss}")
            if abs(global_loss - prev_global_loss) < 0.0001:
                consecutive_loss_change_rounds += 1

            if consecutive_loss_change_rounds == 3:
                print("The global model parameters have not been updated for 3 consecutive rounds, so the training has converged.")
                break

            if self.stratergy == "fedaboost":
                weights = calculate_weights(weights, local_loss)
                stats.append(self.__collect_stats(local_loss, weights))

            self.__broadcast(self.global_model)
            
            if not updated:
                consecutive_no_update_rounds += 1
                print("The global model parameters have not been updated, so the training has converged.")
            else:
                consecutive_no_update_rounds = 0

            if consecutive_no_update_rounds == 3:
                print("The global model parameters have not been updated for 5 consecutive rounds, so the training has converged.")
                break

            if self.stratergy == "fedaboost":
                file_name = "stats/fedaboost_stats.csv"
                self.__save_stats(stats, file_name)
                print(f"Saved the training statistics to {file_name}")
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