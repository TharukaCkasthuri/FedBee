import os
import json
import torch
import argparse

from collections import defaultdict
from torch.utils.data import Dataset

def read_dir(data_dir:str)->tuple:
    """
    Reads data from the input directory.
    
    Parameters:
    ------------
    data_dir: str; path to the directory containing the data
    
    Returns:
    ------------
    clients: list; list of client ids
    groups: list; list of group ids
    data: dict; dictionary containing the data
    """

    clients = []
    groups = []
    data = defaultdict(lambda : None)

    files = os.listdir(data_dir)
    files = [f for f in files if f.endswith('.json')]
    for f in files:
        file_path = os.path.join(data_dir,f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        clients.extend(cdata['users'])
        if 'hierarchies' in cdata:
            groups.extend(cdata['hierarchies'])
        data.update(cdata['user_data'])

    clients = list(sorted(data.keys()))

    return clients, groups, data


def read_data(train_data_dir:str, test_data_dir:str)->tuple:
    """parses data in given train and test data directories

    assumes:
    - the data in the input directories are .json files with 
        keys 'users' and 'user_data'
    - the set of train set users is the same as the set of test set users
    
    Parameters:
    ------------
    train_data_dir: str; path to the directory containing the training data
    test_data_dir: str; path to the directory containing the test data

    Returns:
    ------------
    clients: list of client ids
    groups: list of group ids; empty list if none found
    train_data: dictionary of train data
    test_data: dictionary of test data

    """

    train_clients, train_groups, train_data = read_dir(train_data_dir)
    test_clients, test_groups, test_data = read_dir(test_data_dir)

    assert train_clients == test_clients
    assert train_groups == test_groups

    return test_clients, test_groups, train_data, test_data

class FEMNISTDataset(Dataset):
    """Custom Dataset for loading FEMNIST data"""

    def __init__(self, data: dict)->None:
        """
        Args:
            data: dictionary with keys 'x' and 'y' for input and labels.
        """
        self.x = data['x']
        self.y = data['y']
        
    def __len__(self)->int:
        """
        Returns the length of the dataset.
        
        Returns:
        ------------
        length: int; length of the dataset
        """
        return len(self.y)
    
    def __getitem__(self, idx)->tuple:
        """
        Returns the item at the given index.
        
        Parameters:
        ------------
        idx: int; index of the item
        
        Returns:
        ------------
        sample_x: torch.tensor object; input data
        sample_y: torch.tensor object; label
        """
        sample_x = self.x[idx]
        sample_y = self.y[idx]
        return torch.tensor(sample_x, dtype=torch.float32), torch.tensor(sample_y, dtype=torch.long)

def main():

    BATCH_SIZE = 32
    min_samples = 1 * BATCH_SIZE

    parser = argparse.ArgumentParser(description="FEMNIST dataset preprocessing parameters")
    parser.add_argument("--train_data_dir", type=str, 
                        default=os.path.join('/Users/tak/Documents/BTH/','leaf','data','femnist','data','train'), 
                        help="Path to the training data directory")
    parser.add_argument("--test_data_dir", type=str, 
                        default=os.path.join('/Users/tak/Documents/BTH/','leaf','data','femnist','data','test'),
                        help="Path to the test data directory")
    
    args = parser.parse_args()
    train_data_dir = args.train_data_dir
    test_data_dir = args.test_data_dir
    
    train_clients, train_groups, train_data, test_data = read_data(train_data_dir, test_data_dir)

    for client in train_clients:
        if len(train_data[client]['y']) < min_samples and len(test_data[client]['y']) < min_samples:
            continue

        if not os.path.exists("./trainpt"):
            os.makedirs("./trainpt")
        if not os.path.exists("./testpt"):
            os.makedirs("./testpt")

        train_dataset = FEMNISTDataset(train_data[client])
        torch.save(train_dataset, "./trainpt/" + str(client) + ".pt")
        test_dataset = FEMNISTDataset(test_data[client])
        torch.save(test_dataset, "./testpt/" + str(client) + ".pt")

        train_path = f"./trainpt/{client}.pt"
        test_path  = f"./testpt/{client}.pt"

        if os.path.exists(train_path):
            torch.save(train_dataset, train_path)
        else:
            os.makedirs(os.path.dirname(train_path), exist_ok=True)
            torch.save(train_dataset, train_path)

        if os.path.exists(test_path):
            torch.save(test_dataset, test_path)
        else:
            os.makedirs(os.path.dirname(test_path), exist_ok=True)
            torch.save(test_dataset, test_path)
    
if __name__ == "__main__":
    main()