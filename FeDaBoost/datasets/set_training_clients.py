import os
import json
import random
import argparse
import configparser
import numpy as np

def sample_clients(client_ids: list, min_clients, max_clients) -> dict:
    """
    Sample clients from the client dictionary.

    Parameters:
    ----------------
    num_clients: int;
        Number of clients to sample

    Returns:
    ----------------
    list
        Dict of sampled clients
    """
    num_clients = random.randint(min_clients, max_clients)
    sampled_client_ids = np.random.choice(client_ids, num_clients, replace=False)

    return sampled_client_ids

def get_client_ids(folder_path):
    client_ids = []
    try:
        files = os.listdir(folder_path)
        for file in files:
            if file.endswith('.pt'):
                client_id = file.split('.')[0]
                client_ids.append(client_id)
        return client_ids
    except Exception as e:
        raise e

def parse_arguments():
    parser = argparse.ArgumentParser(description="Federated training parameters")
    parser.add_argument("--dir", type=str, default="femnist", help="Choose a dataset from the available options; femnist, mnist, kv")
    return parser.parse_args()

def load_config(config_path="../config.cfg"):
    config = configparser.ConfigParser()
    config.read(config_path)
    return config

def main():
    args = parse_arguments()
    folder_path = args.dir
    config = load_config()

    min_clients = int(config['FEMNIST']['min_clients'])
    max_clients = int(config['FEMNIST']['max_clients'])

    client_ids = get_client_ids(f"{folder_path}/trainpt")

    training_samples = {i: sample_clients(client_ids, min_clients, max_clients) for i in range(1, 501)}
    training_samples = {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in training_samples.items()}
    output_file = f"{folder_path}/training_samples.json"

    with open(output_file, 'w') as f:
        json.dump(training_samples, f, indent=4)

if __name__ == "__main__":
    main()