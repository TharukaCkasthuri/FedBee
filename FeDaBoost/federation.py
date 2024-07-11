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

import time
import torch
import argparse
import os

import pandas as pd
from tqdm import tqdm

from clients import Client, FlowerClient
from server import Server
from utils import get_device, get_client_ids
from evals import evaluate
from models.kv import ShallowNN
from models.femnist import SimpleNet
from params import model_hparams
from  aggregators import fedAvg,fedProx

from datasets.kv.preprocess import KVDataSet
from datasets.femnist.preprocess import FEMNISTDataset

from torch.utils.tensorboard import SummaryWriter

class Federation:
    """
    Class for federated learning.

    Parameters:
    ------------
    client_ids: list;
        List of client ids.
    model: torch.nn.Module;
        Model to be trained.
    loss_fn: torch.nn.Module;
        Loss function.
    global_rounds: int;
        Number of global rounds.
    stratergy: callable;
        Federated learning averaging stratergy.


    Methods:
    ----------
    __init__(self, client_ids: list, model: torch.nn.Module, loss_fn:torch.nn.Module, global_rounds: int, stratergy: callable, local_rounds: int) -> None:
        Initializes a Federation instance with the specified parameters.

    train(self, model, summery=False) -> tuple:
        Trains the model using federated learning.

    save_stats(self, model, training_stat: list) -> None:
        Saves the training statistics and the model.
    """

    def __init__(
        self,
        client_ids: list,
        model: torch.nn.Module,
        loss_fn: torch.nn.Module,
        global_rounds: int,
        stratergy: callable,
        local_rounds: int,
    ) -> None:
        
        self.client_ids = client_ids
        self.model = model
        self.loss_fn = loss_fn
        self.global_rounds = global_rounds
        self.local_rounds = local_rounds
        self.stratergy = stratergy

        # Initialize the server
        self.server = Server(global_rounds,stratergy)
        self.server.init_model(model)

        # Set up the clients
        for id in client_ids:
            self.server.connect_client(Client(
                id,
                torch.load(f"datasets/femnist/trainpt/{id}.pt"),
                torch.load(f"datasets/femnist/testpt/{id}.pt"),
                self.loss_fn,
                32,
                0.01,
                0.01,
                self.local_rounds,
                local_model=SimpleNet(62),
            ))

    def train(
        self
    ) -> tuple:
        """
        Training the model.

        Parameters:
        ----------------
        

        Returns:
        ----------------
       
        """

        self.server.train()

        return None


    def save_models(self, model: torch.nn.Module, ckptpath: str) -> None:
        """
        Saving the training stats and the model.

        Parameters:
        ----------------
        model:
            Trained model.
        ckptpath: str;
            Path to save the model and the training stats. Default is None.

        Returns:
        ----------------
        None
        """

        if os.path.exists(ckptpath):
            torch.save(
                model.state_dict(),
                ckptpath,
            )
        else:
            os.makedirs(os.path.dirname(ckptpath), exist_ok=True)
            torch.save(
                model.state_dict(),
                ckptpath,
            )


if __name__ == "__main__":
    device = get_device()
    parser = argparse.ArgumentParser(description="Federated training parameters")
    parser.add_argument("--loss_function", type=str, default="L1Loss")
    parser.add_argument("--log_summary", action="store_true")
    parser.add_argument("--global_rounds", type=int, default=2)
    parser.add_argument("--local_rounds", type=int, default=10)
    parser.add_argument("--save_ckpt", action="store_true")
    args = parser.parse_args()


    # Hyper Parameters
    loss_fn = getattr(torch.nn, args.loss_function)()
    log_summary = args.log_summary
    global_rounds = args.global_rounds
    local_rounds = args.local_rounds
    epochs = global_rounds * local_rounds
    save_ckpt = args.save_ckpt

    checkpt_path = f"checkpt/fedl/selected_/epoch_{epochs}/{global_rounds}_rounds_{local_rounds}_epochs_per_round/"

    client_ids = get_client_ids("datasets/femnist/trainpt")[0:10]
    model = SimpleNet(62)

    federation = Federation(
        client_ids,
        model,
        loss_fn,
        global_rounds,
        "fedavg",
        local_rounds,
    )
    
    print("Federation with clients " + ", ".join(client_ids))

    start = time.time()
    trained_model = federation.train()
    model_path = f"{checkpt_path}/global_model.pth"
    print("Federation with clients " + ", ".join(client_ids))
    print(
        "Approximate time taken to train",
        str(round((time.time() - start) / 60, 2)) + " minutes",
    )
