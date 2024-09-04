import numpy as np
import pickle
import cv2
import os
import argparse

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset

from imutils import paths

def load(paths, verbose=-1):
    """
    Loads the images and labels from disk.

    Parameters:
    ------------
    paths: list of image paths
    verbose: whether to display progress

    Returns:
    ------------
        tuple of images and labels
    """
    data = list()
    labels = list()

    for (i, imgpath) in enumerate(paths):    
        im_gray = cv2.imread(imgpath , cv2.IMREAD_GRAYSCALE)
        image = np.array(im_gray).flatten() 
        label = imgpath.split(os.path.sep)[-2]
        data.append(image/255)
        labels.append(label)
        if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
            print("[INFO] processed {}/{}".format(i + 1, len(paths)))

    return data, labels

def create_clients(image_list, label_list, num_clients=20, initial='clients', save_dir='client_data'):
    """
    Create clients using the given images and labels, and save each client's data into a directory.
    
    Parameters:
    ------------
    image_list: list of numpy arrays
    label_list: list of binary labels
    num_clients: number of clients (default=20)
    initial: client name prefix
    save_dir: the base directory to save client data
    
    Returns:
    ------------
    clients: dictionary of client data
    """
    
    # Ensure the save directory exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print('Creating {} clients'.format(num_clients))

    client_names = ['{}_{}'.format(initial, i+1) for i in range(num_clients)]
    
    max_y = np.argmax(label_list, axis=-1)
    sorted_zip = sorted(zip(max_y, label_list, image_list), key=lambda x: x[0])
    data = [(x, y) for _, y, x in sorted_zip]
    size = len(data) // num_clients
    shards = [data[i:i + size] for i in range(0, size * num_clients, size)]
    assert len(shards) == len(client_names)

    # Create dictionary for clients and save data
    clients = {}
    for i, client_name in enumerate(client_names):
        client_data = shards[i]
        clients[client_name] = client_data
        
        # Save the data
        file_name = str(client_name)+ '.pkl'
        with open(os.path.join(save_dir, file_name), 'wb') as f:
            pickle.dump(client_data, f)
    
    return clients

class MNISTDataset(Dataset):
    """
    Custom dataset class for the training and validation dataset.
    """

    def __init__(self, data) -> None:
        self.data = data

    def __len__(self) -> int:
        """
        Returns the length of the dataset.

        Returns:
        --------
        length: int; length of the dataset
        """
        return len(self.data)

    def __getitem__(self, idx) -> tuple:
        """
        Returns the item at the given index.

        Parameters:
        ------------
        idx: int; index of the item

        Returns:
        ------------
        x_train: torch.tensor object; input data
        y_train: torch.tensor object; label
        """
        image, label = self.data[idx]
        return torch.tensor(image, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

    
def build_dataset(data_dir, saving_dir) -> None:
    """
    Split the pickles into train and test, saving as a PyTorch dataset.

    Parameters:
    ------------
    data_dir: str; path to the pickle files
    saving_dir: str; path to save the PyTorch datasets

    Returns:
    ------------
    None
    """

    files = os.listdir(data_dir)
    ids = [file.split(".")[0] for file in files]
    files_path = [os.path.join(data_dir, file) for file in files]

    if not os.path.exists(saving_dir):
        os.makedirs(saving_dir)
        os.makedirs(os.path.join(saving_dir, "trainpt"))
        os.makedirs(os.path.join(saving_dir, "testpt"))

    for id, pickle_file in zip(ids, files_path):
        with open(pickle_file, 'rb') as f:
            data = pickle.load(f)

        # Split the data into train and test sets
        train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

        # Create the datasets
        train_dataset = MNISTDataset(train_data)
        test_dataset = MNISTDataset(test_data)

        # Save the datasets
        trainpt_dir = os.path.join(saving_dir, f"trainpt")
        testpt_dir = os.path.join(saving_dir, f"testpt")
        if not os.path.exists(trainpt_dir):
            os.makedirs(trainpt_dir)
        if not os.path.exists(testpt_dir):
            os.makedirs(testpt_dir)

        torch.save(train_dataset, os.path.join(trainpt_dir, f"{id}.pt"))
        torch.save(test_dataset, os.path.join(testpt_dir, f"{id}.pt"))

        print(f"Saved { os.path.join(trainpt_dir, f"{id}.pt")} and {os.path.join(testpt_dir, f"{id}.pt")}")

def main():
    parser = argparse.ArgumentParser(description="Preprocess the MNIST dataset.")
    parser.add_argument("--num_clients", type=int, default=200)
    parser.add_argument("--image_path", type=str, default="/Users/tak/Documents/BTH/MNIST/trainingSet")
    args = parser.parse_args()

    image_path = args.image_path
    image_paths = list(paths.list_images(image_path))
    image_list, label_list = load(image_paths, verbose=10000)

    #binarize the labels
    lb = LabelBinarizer()
    label_list = lb.fit_transform(label_list)

    #split data into training and test set
    X_train, X_test, y_train, y_test = train_test_split(image_list, 
                                                        label_list, 
                                                        test_size=0.1, 
                                                        random_state=42)

    create_clients(X_train, y_train, num_clients=args.num_clients, initial='client')
    build_dataset("/Users/tak/Documents/BTH/FeDABoost/FeDaBoost/datasets/mnist/client_data", "/Users/tak/Documents/BTH/FeDABoost/FeDaBoost/datasets/mnist")

if __name__ == "__main__":
    main()