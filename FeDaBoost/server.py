import torch
from clients import Client
from aggregators import fedAvg

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
        self.client_dict = {
            client_id: client}
        
    def __aggregate(self) -> None:
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
        else:
            pass

    def __broadcast(self, model: torch.nn.Module) -> None:
        """
        Broadcast the model to the clients.
        """
        for client in self.client_dict.values():
            client.set_model(self.global_model.state_dict()) 

    def __receive(self, client:callable) -> list:
        """
        Receive the models from the clients.

        Returns:
        ----------------
        models: list;
            List of models
        """
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
        for round in range(self.rounds):
            print(f"\n | Global Training Round : {round+1} |\n")
            for client in self.client_dict.values():
                client.set_model(self.global_model.state_dict())
                client_model, client_loss = client.train()
                self.__receive(client)
            self.__aggregate()
            self.__broadcast(self.global_model)

        return self.global_model
    
    