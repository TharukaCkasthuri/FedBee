import torch
import os
import numpy as np
import pandas as pd

from clients import Client
from server import Server
from aggregators import fedAvg, fedProx, weighted_avg
from utils import calculate_weights
from agent import DQNAgent

class RLServer(Server):

    def __init__(self, rounds: int, strategy: callable) -> None:
        super().__init__(rounds, strategy)

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
        state_size = len(self.client_dict.values())
        action_size = len(self.client_dict.values())
        self.agent = DQNAgent(state_size, action_size)

        consecutive_no_update_rounds = 0
        consecutive_loss_change_rounds = 0
        prev_global_loss = float('inf')
        stats = []

        weights = [(1/len(self.client_dict)) for client in self.client_dict]

        for round in range(1,self.rounds+1):
            updated = False
            print(f"\n | Global Training Round : {round+1} |\n")

            validation_loss = []
            for client in self.client_dict.values():
                client.set_model(self.global_model.state_dict())
                client_val_loss = client.evaluate()
                validation_loss.append(client_val_loss)
                client_model, _ = client.train(round)
                self._receive(client)

            weights = self.agent.determine_weights(validation_loss)
            reward = -sum(validation_loss)
            state = np.array(validation_loss)
            self.agent.remember(state, weights, reward, state, done=False)
            self.agent.replay(batch_size=8)
                
            prev_params = [p.clone() for p in self.global_model.parameters()]
            client_models = [client.get_model() for client in self.client_dict.values()]
            self.global_model = weighted_avg(self.global_model, client_models, weights)
            updated_params = [p.clone() for p in self.global_model.parameters()]

            updated = self._check_model_update(prev_params, updated_params)
            print(f"Model Updated: {updated}")

            global_loss = sum(validation_loss) / len(validation_loss)

            print(f"Global Loss : {global_loss}")
            if abs(global_loss - prev_global_loss) < 0.0001:
                consecutive_loss_change_rounds += 1

            if consecutive_loss_change_rounds == 3:
                print("The global model parameters have not been updated for 3 consecutive rounds, so the training has converged.")
                break

            stats.append(self._collect_stats(validation_loss, weights))
            self._broadcast(self.global_model)
            
            if not updated:
                consecutive_no_update_rounds += 1
                print("The global model parameters have not been updated, so the training has converged.")
            else:
                consecutive_no_update_rounds = 0

            if consecutive_no_update_rounds == 3:
                print("The global model parameters have not been updated for 5 consecutive rounds, so the training has converged.")
                break

            file_name = "stats/rl_stats.csv"
            self._save_stats(stats, file_name)
            print(f"Saved the training statistics to {file_name}")

        return self.global_model    