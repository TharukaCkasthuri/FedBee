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

from enum import Enum

from clients import Client
from server import Server
from utils import get_device, get_client_ids

from models.kv import ShallowNN
from models.femnist import FEMNISTNet
from models.mnist import MNISTNet

from  aggregators import fedAvg,fedProx

from datasets.kv.preprocess import KVDataSet
from datasets.femnist.preprocess import FEMNISTDataset
from datasets.mnist.preprocess import MNISTDataset

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
    local_rounds: int;
        Number of local rounds.

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
        train_data_dir: str,
        test_data_dir: str,
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
                torch.load(f"{train_data_dir}/{id}.pt"),
                torch.load(f"{test_data_dir}/{id}.pt"),
                self.loss_fn,
                32,
                0.0001,
                0.001,
                self.local_rounds,
                local_model=model,
            ))

    def train(self) -> tuple:
        """
        Training the model.

        Parameters:
        ----------------
        

        Returns:
        ----------------
       
        """
        trained_model = self.server.train()

        return trained_model

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

class Dataset(Enum):
    FEMNIST = "femnist"
    MNIST = "mnist"
    KV = "kv"

def dataset_enum(dataset_str):
    """
    Returns the dataset enum.
    """
    try:
        return Dataset(dataset_str.lower())
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid dataset. Choose from: {', '.join([dataset.value for dataset in Dataset])}")

if __name__ == "__main__":
    device = get_device()
    parser = argparse.ArgumentParser(description="Federated training parameters")
    parser.add_argument("--dataset", type=dataset_enum, default="mnist", help="Choose a dataset from the available options; femnist, mnist, kv")
    parser.add_argument("--train_data_dir", type=str, default="datasets/mnist/trainpt", help="Path to the training data directory")
    parser.add_argument("--test_data_dir", type=str, default="datasets/mnist/testpt", help="Path to the test data directory")
    parser.add_argument("--loss_function", type=str, default="CrossEntropyLoss")
    parser.add_argument("--stratergy", type=str, default="fedaboost")
    parser.add_argument("--log_summary", action="store_true")
    parser.add_argument("--global_rounds", type=int, default=70)
    parser.add_argument("--local_rounds", type=int, default=10)
    parser.add_argument("--save_ckpt", action="store_true")
    args = parser.parse_args()

    loss_fn = getattr(torch.nn, args.loss_function)()
    log_summary = args.log_summary
    global_rounds = args.global_rounds
    local_rounds = args.local_rounds
    epochs = global_rounds * local_rounds
    save_ckpt = args.save_ckpt
    stratergy = args.stratergy
    train_data_dir = args.train_data_dir
    test_data_dir = args.test_data_dir
    dataset = args.dataset

    checkpt_path = f"checkpt/{stratergy}/{dataset.name}/epoch_{epochs}/{global_rounds}_rounds_{local_rounds}_epochs_per_round/"

    client_ids = get_client_ids(train_data_dir)

    if args.dataset == Dataset.FEMNIST:
        model = FEMNISTNet(62)
    elif args.dataset == Dataset.MNIST:
        model = MNISTNet()
    elif args.dataset == Dataset.KV:
        model = ShallowNN(176)

    federation = Federation(
        client_ids,
        model,
        loss_fn,
        train_data_dir,
        test_data_dir,
        global_rounds,
        stratergy,
        local_rounds,
    )
    
    print("Federation with clients " + ", ".join(client_ids))

    start = time.time()
    trained_model = federation.train()
    model_path = f"{checkpt_path}/global_model.pth"
    federation.save_models(trained_model, model_path)
    print("Federation with clients " + ", ".join(client_ids))
    print(
        "Approximate time taken to train",
        str(round((time.time() - start) / 60, 2)) + " minutes",
    )
