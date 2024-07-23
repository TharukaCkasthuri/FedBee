import torch
from clients import Client
from aggregators import fedAvg, fedProx

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
        elif self.stratergy == "fedprox":
            self.global_model = fedProx(self.global_model, client_models)
        return self.global_model


    def __broadcast(self, model: torch.nn.Module) -> None:
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


    def __receive(self, client:callable) -> list:
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

        if self.stratergy == "fedaboost":
            weights = [(1/len(self.clients)) for client in self.clients]

        for round in range(self.rounds):
            updated = False
            print(f"\n | Global Training Round : {round+1} |\n")

            if self.stratergy == "fedaboost":
                local_loss = []
                
            for client in self.client_dict.values():
                client.set_model(self.global_model.state_dict())
                client_model, client_loss = client.train()
                self.__receive(client)

            m1_params = [p.clone() for p in self.global_model.parameters()]
            self.global_model = self.__aggregate()
            m2_params = [p.clone() for p in self.global_model.parameters()]

            for p1, p2 in zip(m1_params, m2_params):
                if not torch.equal(p1.data, p2.data):
                    updated = True
                    print("The global model parameters have been updated using", self.stratergy)
                    break

            self.__broadcast(self.global_model)
            
            if not updated:
                consecutive_no_update_rounds += 1
                print("The global model parameters have not been updated, so the training has converged.")
            else:
                consecutive_no_update_rounds = 0

            if consecutive_no_update_rounds == 3:
                print("The global model parameters have not been updated for 5 consecutive rounds, so the training has converged.")
                break

        return self.global_model