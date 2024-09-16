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

from clients import Client, OptimaClient, RankClient
from server import Server
from server import OptimaServer, RankServer


from utils import get_device, get_client_ids

from models.kv import ShallowNN
from models.femnist import FEMNISTNet
from models.mnist import MNISTNet

from evals import FocalLoss, HybridLoss

from datasets.kv.preprocess import KVDataSet
from datasets.femnist.preprocess import FEMNISTDataset
from datasets.mnist.preprocess import MNISTDataset
from torch.utils.data import ConcatDataset

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

        # Initialize the server, and the clients. 
        if stratergy == "fedaboost-optima":
            self.server = OptimaServer(global_rounds,stratergy,checkpt_path=checkpt_path)
            self.server.init_model(model)

            # Set up the clients for fedaboost-optima server
            for id in client_ids:
                self.server.connect_client(OptimaClient(
                    id,
                    torch.load(f"{train_data_dir}/{id}.pt"),
                    torch.load(f"{test_data_dir}/{id}.pt"),
                    self.loss_fn,
                    32,
                    0.001,
                    0.01,
                    local_model=model,
                ))
        elif stratergy == "fedaboost-ranker":
            test_data = []
            for id in client_ids:
                ds = torch.load(f"{test_data_dir}/{id}.pt")
                test_data.append(ds)
            test_dataset = MNISTDataset(test_data)

            self.server = RankServer(global_rounds, stratergy, checkpt_path=checkpt_path, test_dataset=test_dataset,loss_fn=self.loss_fn)
            self.server.init_model(model)

            # Set up the clients for fedaboost-ranker server
            for id in client_ids:
                self.server.connect_client(RankClient(
                    id,
                    torch.load(f"{train_data_dir}/{id}.pt"),
                    torch.load(f"{test_data_dir}/{id}.pt"),
                    self.loss_fn,
                    32,
                    0.001,
                    0.01,
                    local_model=model,
                ))
        else:
            self.server = Server(global_rounds,stratergy,checkpt_path=checkpt_path)
            self.server.init_model(model)

            # Set up the clients for fedavg server
            for id in client_ids:
                self.server.connect_client(Client(
                    id,
                    torch.load(f"{train_data_dir}/{id}.pt"),
                    torch.load(f"{test_data_dir}/{id}.pt"),
                    self.loss_fn,
                    32,
                    0.001,
                    0.01,
                    local_model=model,
                    local_round=self.local_rounds,
                ))

    def train(self) -> tuple:
        """
        Training the model.

        Parameters:
        ----------------
        

        Returns:
        ----------------
        trained_model: torch.nn.Module;
            Trained model.
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

class Stratergy(Enum):
    FEDAVG = "fedavg"
    FEDPROX = "fedprox"
    FEDABOOSTOPTIMA = "fedaboost-optima"
    FEDABOOSTRANKER = "fedaboost-ranker"
    FEDABOOSTCONCORD = "fedaboost-concord"

def dataset_enum(dataset_str):
    """
    Returns the dataset enum.

    
    Parameters:
    ----------------
    dataset_str: str;
        Dataset string.
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
    parser.add_argument("--loss_function", type=str, default="FocalLoss", help="Choose a loss function from the available options; CrossEntropyLoss, FocalLoss, HybridLoss")
    parser.add_argument("--stratergy", type=str, default="fedaboost-ranker", help="Choose a federated learning stratergy from the available options; fedavg, fedprox, fedaboost")
    parser.add_argument("--log_summary", action="store_true")
    parser.add_argument("--global_rounds", type=int, default=70)
    parser.add_argument("--local_rounds", type=int, default=10)
    parser.add_argument("--save_ckpt", action="store_true")

    args = parser.parse_args()

    if args.loss_function == "HybridLoss":
        loss_fn = HybridLoss(focal_alpha=1, focal_gamma=2, focal_weight=0.7)
    elif args.loss_function == "FocalLoss":
        loss_fn = FocalLoss(alpha=1, gamma=2)
    else:
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
