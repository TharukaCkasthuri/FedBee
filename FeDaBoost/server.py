import torch
import os
import pandas as pd

from clients import Client
from aggregators import fedAvg, fedProx, weighted_avg
from utils import get_alpha,get_weights, influence_alpha, adjust_local_epochs

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

    def __init__(self,rounds:int, stratergy:callable, log_dir:str = 'runs') -> None:
        self.rounds = rounds
        self.stratergy = stratergy
        self.client_dict = {}

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
        

    def _aggregate(self, weights = None) -> None:
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

        aggregation_functions = {
        "fedavg": fedAvg,
        "fedprox": fedProx,
        "fedaboost-optima": weighted_avg,
        "fedaboost-ranker": weighted_avg,
        "fedaboost-concord": weighted_avg,
        }

        aggregation_function = aggregation_functions.get(self.stratergy)

        if aggregation_function is None:
            raise ValueError(f"Unsupported aggregation strategy: {self.stratergy}")
        
        if self.stratergy == "fedaboost-optima" or self.stratergy == "fedaboost-ranker" or self.stratergy == "fedaboost-concord":
            self.global_model = aggregation_function(self.global_model, client_models, weights)
        else:
            self.global_model = aggregation_function(self.global_model, client_models)

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
        prev_global_loss = 999999999
        stats = []

        for round in range(1,self.rounds+1):
            update_status = False
            print(f"\n | Global Training Round : {round} |\n")

            for client in self.client_dict.values():
                client.set_model(self.global_model.state_dict())
                _ = client.train(round)
                self._receive(client)

            self.global_model, update_status = self.__aggregate()

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
    