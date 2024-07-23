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

Paper: [FeDaBoost: AdaBoost Enhanced Federated Averaging]
Published in: [Journal/Conference Name]
"""

import time
import torch
import argparse
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import Client
from utils import get_device
from evals import evaluate
from utils import min_max_normalization
from utils import calculate_weights
from models import ShallowNN

from pyhessian import hessian

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

class Federation:
    """
    Class for federated learning.

    Parameters:
    ------------
    checkpt_path: str;
        Path to save the model
    features: int;
        Number of features in the input data.
    loss_fn: torch.nn.Module object;
        The loss function used for training.
    batch_size: int;
        The batch size for training.
    learning_rate: float;
        The learning rate for the optimizer.
    rounds : int
        The number of training rounds in federated learning.
    epochs_per_round : int
        The number of training epochs per round.

    Methods:
    ----------
    __init__(self, checkpt_path: str, features: int, loss_fn, batch_size, learning_rate, rounds, epochs_per_round):
        Initializes a Federation instance with the specified parameters.

    set_clients(self, client_ids: list) -> None:
        Sets up the clients for federated learning.

    train(self, model, summery=False) -> tuple:
        Trains the model using federated learning.

    save_stats(self, model, training_stat: list) -> None:
        Saves the training statistics and the model.
    """

    def __init__(
        self,
        checkpt_path: str,
        features: int,
        loss_fn: torch.nn.Module,
        batch_size: int,
        learning_rate: float,
        rounds: int,
        epochs_per_round: int,
        save_ckpt: bool = False,
        log_summary: bool = False,
    ) -> None:
        self.checkpt_path = checkpt_path
        self.features = features
        self.loss_fn = loss_fn
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.training_rounds = rounds
        self.epochs_per_round = epochs_per_round
        self.save_ckpt = save_ckpt
        self.log_summary = log_summary

        self.writer = SummaryWriter(comment="_fed_train_batch" + str(batch_size))

    def set_clients(self, client_ids: list) -> None:
        """
        Setting up the clients for federated learning.

        Parameters:
        ----------------
        client_ids: list;
            List of client ids

        Returns:
        ----------------
        None
        """
        self.client_ids = client_ids
        self.clients = [
            Client(
                id,
                torch.load("trainpt/" + id + ".pt"),
                torch.load("testpt/" + id + ".pt"),
                self.batch_size,
            )
            for id in self.client_ids
        ]

        self.client_model_dict = {}
        for i in self.client_ids:
            self.client_model_dict[i] = ShallowNN(self.features)

        test_dataset = torch.utils.data.ConcatDataset(
            [torch.load("testpt/" + id + ".pt") for id in self.client_ids]
        )
        self.test_dataloader = DataLoader(test_dataset, self.batch_size, shuffle=True)

    def train(
        self,
        model: torch.nn.Module,
        log_summary: bool = False,
    ) -> tuple:
        """
        Training the model.

        Parameters:
        ----------------
        model:torch.nn.Module;
            Model to be trained. The model should have a track_layers attribute which is a dictionary of the layers to be trained.
        summery: bool;
            Whether to save the training stats and the model. Default is False.

        Returns:
        ----------------
        global_model:torch.nn.Module;
            Trained model
        training_stats: list;
            Training stattistics as a list of dictionaries
        """

        # initiate global model
        global_model = model(self.features)
        # global_model.to(device)
        global_model.train()

        global_weights = global_model.state_dict()
        model_layers = global_model.track_layers.keys()

        training_stats = []
        weights = [(1/24) for client in self.clients]
        for iter in tqdm(range(self.training_rounds)):
            local_models, local_loss, top_eigens = [], [], []

            print(f"\n | Global Training Round : {iter+1} |\n")

            for client in self.clients:
                client_id = client.client_id
                training_stat_dict = {"client_id": client_id, "training_round": iter}

                #Calculate the top eigen values for clients
                global_model.load_state_dict(global_weights)
                hessian_comp = hessian(global_model, loss_fn,  dataloader=client.train_dataloader, cuda=False)
                top_eigenvalues, top_eigenvector = hessian_comp.eigenvalues(maxIter=100)
                top_eigens.append(abs(top_eigenvalues[0]))
                # client_eigenvalues = client.eigenvalues()

                # loading the global model weights to client model
                self.client_model_dict[client_id].load_state_dict(global_weights)

                # setting up the optimizer
                optimizer = torch.optim.SGD(
                    self.client_model_dict[client_id].parameters(),
                    lr=self.learning_rate,
                )
                print(f"Client: {client_id} Started it's local training")

                # training the client model for n' epochs
                for ep in range(self.epochs_per_round):
                    this_client_state_dict, training_loss = client.train(
                        self.client_model_dict[client_id],
                        self.loss_fn,
                        optimizer,
                        ep,
                    )

                if self.save_ckpt:
                    local_path = f"{self.checkpt_path}/global_{iter+1}/clients/client_model_{client_id}.pth"

                    if os.path.exists(local_path):
                        torch.save(
                            this_client_state_dict.state_dict(),
                            local_path,
                        )
                    else:
                        os.makedirs(os.path.dirname(local_path), exist_ok=True)
                        torch.save(
                            this_client_state_dict.state_dict(),
                            local_path,
                        )

                local_loss.append(training_loss)
                local_models.append(this_client_state_dict)

                training_stat_dict["fed_train"] = (
                    sum(local_loss[-self.epochs_per_round :]) / self.epochs_per_round
                )
                validation_loss = client.eval(
                    self.client_model_dict[client_id], self.loss_fn
                )
                training_stat_dict["fed_val"] = validation_loss
                training_stats.append(training_stat_dict)

            # update global model parameters here
            state_dicts = [model.state_dict() for model in local_models]
            weights = [round(weight,3) for weight in weights]

            for key in model_layers:
                # Stack the tensors and multiply each element by its corresponding weight
                stacked_weights = torch.stack(
                    [item[str(key) + ".weight"] * weight for item, weight in zip(state_dicts, weights)]
                )

                # Take the weighted mean along the specified dimension
                global_model.track_layers[key].weight.data = stacked_weights.sum(dim=0) / sum(weights)

                # Repeat the process for bias
                stacked_biases = torch.stack(
                    [item[str(key) + ".bias"] * weight for item, weight in zip(state_dicts, weights)]
                )
                global_model.track_layers[key].bias.data = stacked_biases.sum(dim=0) / sum(weights)

            global_weights = global_model.state_dict()

            weights = calculate_weights(weights, top_eigens)


            if log_summary:
                global_training_loss = sum(local_loss) / len(local_loss)
                global_validation_loss, _, _ = evaluate(
                    global_model, self.test_dataloader, loss_fn
                )

                self.writer.add_scalars(
                    "Global Model - Federated Learning",
                    {
                        "Training Loss": global_training_loss,
                        "Validation Loss": global_validation_loss,
                    },
                    iter,
                )

            if self.save_ckpt:
                global_path = f"{self.checkpt_path}/global_{iter+1}/global_model.pth"
                if os.path.exists(global_path):
                    torch.save(
                        global_model.state_dict(),
                        global_path,
                    )
                else:
                    os.makedirs(os.path.dirname(global_path), exist_ok=True)
                    torch.save(
                        global_model.state_dict(),
                        global_path,
                    )

        self.writer.flush()
        self.writer.close()

        return global_model, training_stats

    def save_stats(self, model: torch.nn.Module, training_stat: list) -> None:
        """
        Saving the training stats and the model.

        Parameters:
        ----------------
        model:
            Trained model.
        training_stat: list;
            Training stats.
        path_det: str;
            Path to save the model and the training stats. Default is None.

        Returns:
        ----------------
        None
        """

        path = f"losses/_federated_stats_epoch{self.training_rounds}_{self.epochs_per_round}.csv"

        if os.path.exists(path):
            pd.DataFrame.from_dict(training_stat).to_csv(
                path,
                index=False,
            )
        else:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            pd.DataFrame.from_dict(training_stat).to_csv(
                path,
                index=False,
            )

        if os.path.exists(self.checkpt_path):
            torch.save(
                model.state_dict(),
                f"{self.checkpt_path}_fedl_global_{self.training_rounds}_{self.epochs_per_round}.pth",
            )
        else:
            os.makedirs(os.path.dirname(self.checkpt_path), exist_ok=True)
            torch.save(
                model.state_dict(),
                f"{self.checkpt_path}_fedl_global_{self.training_rounds}_{self.epochs_per_round}.pth",
            )


if __name__ == "__main__":
    device = get_device()
    parser = argparse.ArgumentParser(description="Federated training parameters")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=0.005)
    parser.add_argument("--loss_function", type=str, default="L1Loss")
    parser.add_argument("--log_summary", action="store_true")
    parser.add_argument("--rounds", type=int, default=40)
    parser.add_argument("--epochs_per_round", type=int, default=1)
    parser.add_argument("--save_ckpt", action="store_true")
    args = parser.parse_args()

    features = 197

    # Hyper Parameters
    loss_fn = getattr(torch.nn, args.loss_function)()
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    log_summary = args.log_summary
    rounds = args.rounds
    epochs_per_round = args.epochs_per_round
    epochs = rounds * epochs_per_round
    save_ckpt = args.save_ckpt

    checkpt_path = f"checkpt/fedaboost/epoch_{epochs}/{rounds}_rounds_{epochs_per_round}_epochs_per_round/"

    federation = Federation(
        checkpt_path,
        features,
        loss_fn,
        batch_size,
        learning_rate,
        rounds,
        epochs_per_round,
        save_ckpt,
        log_summary,
    )

    client_ids = [f"{i}_{j}" for i in range(4) for j in range(6)]

    print("Federation with clients " + ", ".join(client_ids))

    start = time.time()
    federation.set_clients(client_ids=client_ids)
    trained_model, training_stats = federation.train(ShallowNN)
    federation.save_stats(trained_model, training_stats)
    print("Federation with clients " + ", ".join(client_ids))
    print(
        "Approximate time taken to train",
        str(round((time.time() - start) / 60, 2)) + " minutes",
    )