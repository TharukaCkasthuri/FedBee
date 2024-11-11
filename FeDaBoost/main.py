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

Paper: [FeDBee: AdaBoost Enhanced Federated Learning]
Published in: 
"""
import os
import json
import time
import logging
import argparse
import configparser
from datetime import datetime
from enum import Enum

import torch

from clients import Client, BoostingClient
from server import Server, OptimaServer
from utils import get_device, get_client_ids

from models.kv import ShallowNN
from models.femnist import FEMNISTNet
from models.mnist import MNISTNet

from evals import FocalLoss, HybridLoss

from datasets.kv.preprocess import KVDataSet
from datasets.femnist.preprocess import FEMNISTDataset
from datasets.mnist.preprocess import MNISTDataset

def load_config(config_path="config.cfg"):
    config = configparser.ConfigParser()
    config.read(config_path)
    return config

def setup_logging(stratergy,dataset, timestamp):
    log_dir = '.logs'
    os.makedirs(log_dir, exist_ok=True)
    log_filename = os.path.join(log_dir, f'federation__{dataset}_{stratergy}_{timestamp}.log')
    
    logging.basicConfig(
        filename=log_filename,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return log_filename

def parse_arguments():
    parser = argparse.ArgumentParser(description="Federated training parameters")
    parser.add_argument("--dataset", type=dataset_enum, default="femnist", help="Choose a dataset from the available options; femnist, mnist, kv")
    parser.add_argument("--data_dir", type=str, default="datasets/femnist/", help="Path to the data directory, expected to have train and test directories with names trainpt and testpt respectively." )
    parser.add_argument("--loss_function", type=str, default="FocalLoss", help="Choose a loss function from the available options; CrossEntropyLoss, FocalLoss, HybridLoss")
    parser.add_argument("--stratergy", type=str, default="fedaboost-optima", help="Choose a federated learning stratergy from the available options; fedavg, fedprox, fedaboost")
    parser.add_argument("--log_summary", action="store_true")
    parser.add_argument("--global_rounds", type=int, default=100)
    parser.add_argument("--local_rounds", type=int, default=3)
    parser.add_argument("--save_ckpt", action="store_true")
    return parser.parse_args()

class Federation:
    """
    Class for federated learning.

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
        num_classes: int,
        global_rounds: int,
        stratergy: callable,
        learning_rate: float,
        batch_size: int,
        weight_decay: float,
    ) -> None:
        
        self.client_ids = client_ids
        self.num_classes = num_classes
        self.model = model
        self.loss_fn = loss_fn
        self.global_rounds = global_rounds
        self.stratergy = stratergy
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.weight_decay = weight_decay

        if stratergy == "fedaboost-optima":
            self.server = OptimaServer(global_rounds, stratergy, checkpt_path=checkpt_path)
            self.server.init_model(model)

            # Set up the clients for fedaboost-optima server
            for id in client_ids:
                self.server.connect_client(BoostingClient(
                    id,
                    torch.load(f"{train_data_dir}/{id}.pt"),
                    torch.load(f"{test_data_dir}/{id}.pt"),
                    self.loss_fn,
                    self.batch_size,
                    self.learning_rate,
                    self.weight_decay,
                    local_model=self.model,
                    num_classes=self.num_classes,
                ))

        elif stratergy == "fedavg":
            self.server = Server(global_rounds,stratergy,checkpt_path=checkpt_path)
            self.server.init_model(model)

            # Set up the clients for fedavg server
            for id in client_ids:
                self.server.connect_client(Client(
                    id,
                    torch.load(f"{train_data_dir}/{id}.pt"),
                    torch.load(f"{test_data_dir}/{id}.pt"),
                    self.loss_fn,
                    self.batch_size,
                    self.learning_rate,
                    self.weight_decay,
                    local_model=self.model,
                ))
        else:
                raise ValueError(f"Invalid stratergy. Choose from: {', '.join([stratergy.value for stratergy in Stratergy])}")

    def train(self, training_samples, max_local_round:int = 10, threshold:float = 0.01, patience = 2, alpha_constant=12) -> tuple:
        """
        Training the federated learning model.

        Returns:
        ----------------
        trained_model: torch.nn.Module;
            Trained model.
        """
        print(threshold)
        print(type(threshold))
        if self.stratergy == "fedaboost-optima":
            trained_model = self.server.train(training_samples, max_local_round,threshold, patience, alpha_constant)
        else:
            trained_model = self.server.train(training_samples, max_local_round,threshold, patience)
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

def dataset_enum(dataset_str: str) -> str:
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
    args = parse_arguments()
    config = load_config()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = setup_logging(args.stratergy, args.dataset, timestamp)

    loss_threshould = float(config['GENERAL']['loss_threshould'])
    patience = int(config['GENERAL']['patience'])

    global_rounds = args.global_rounds
    local_rounds = args.local_rounds
    epochs = global_rounds * local_rounds
    save_ckpt = args.save_ckpt
    stratergy = args.stratergy
    data_dir = args.data_dir
    dataset = args.dataset

    train_data_dir = f"{data_dir}/trainpt"
    test_data_dir = f"{data_dir}/testpt"


    # Warn if strategy is 'fedaboost-optima' or 'fedaboost-ranker' without 'FocalLoss'
    if args.stratergy in ["fedaboost-optima", "fedaboost-ranker"] and args.loss_function != "FocalLoss":
        warning_message = (
            f"Warning: The strategy '{args.stratergy}' is selected without using 'FocalLoss' as the loss function. "
            f"It is recommended to use 'FocalLoss' for better performance."
        )
        logging.warning(warning_message)
        print(warning_message)  # Optionally print to console as well

    if args.loss_function == "HybridLoss":
        loss_fn = HybridLoss(focal_alpha=1, focal_gamma=2, focal_weight=0.7)
    elif args.loss_function == "FocalLoss":
        logging.info("Using Focal Loss")
        loss_fn = FocalLoss(alpha=1, gamma=1)
    else:
        loss_fn = getattr(torch.nn, args.loss_function)()
    
    log_summary = args.log_summary
    checkpt_path = f"checkpt/{stratergy}/{dataset.name}/optimized/aplha_2/epoch_{epochs}/{global_rounds}_rounds_{local_rounds}_epochs_per_round/"
    # v2 skipped boosting 1st round, this is the best so
    # v3 in final alpha, changed lamba to 1
    # v4 stop boosting
    # v5 updated the error threshold
    # v6 (error_rate > self.error_threshold) added this condition for boosting
    # v7 removed constant alpha
    client_ids = get_client_ids(train_data_dir)

    if args.dataset == Dataset.FEMNIST:
        model = FEMNISTNet(62)
        learning_rate = float(config['FEMNIST']['learning_rate'])
        batch_size = int(config['FEMNIST']['batch_size'])
        weight_decay = float(config['FEMNIST']['weight_decay'])
        num_classes = int(config['FEMNIST']['num_classes'])
        training_samples = json.load(open(f"{data_dir}/training_samples.json"))
        alpha_constant = float(config['FEMNIST']['alpha_constant'])

    elif args.dataset == Dataset.MNIST:
        model = MNISTNet()
        learning_rate = float(config['MNIST']['learning_rate'])
        batch_size = int(config['MNIST']['batch_size'])
        weight_decay = float(config['MNIST']['weight_decay'])
        num_classes = int(config['MNIST']['num_classes'])
        training_samples = json.load(open(f"{data_dir}/training_samples.json"))
        alpha_constant = float(config['MNIST']['alpha_constant'])

    elif args.dataset == Dataset.KV:
        model = ShallowNN(176)

    federation = Federation(
        client_ids,
        model,
        loss_fn,
        train_data_dir,
        test_data_dir,
        num_classes,
        global_rounds,
        stratergy,
        learning_rate,
        batch_size,
        weight_decay,
    )

    print("Federation with clients " + ", ".join(client_ids))

    start = time.time()
    trained_model = federation.train(training_samples, max_local_round=local_rounds, threshold=loss_threshould, patience=patience, alpha_constant=alpha_constant)
    model_path = f"{checkpt_path}/global_model.pth"
    federation.save_models(trained_model, model_path)
    print("Federation with clients " + ", ".join(client_ids))
    print(
        "Approximate time taken to train",
        str(round((time.time() - start) / 60, 2)) + " minutes",
    )