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

Paper: [Title of Your Paper]
Published in: [Journal/Conference Name]
"""

import os
import torch

import pandas as pd
import numpy as np

from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_file(file_path) -> pd.DataFrame:
    """
    Loads the pickle file into a dataframe.

    Parameters:
    ------------
    file_path: str; path to the pickle file

    Returns:
    ------------
    """
    tmp = pd.read_pickle(file_path)
    df = pd.merge(
        left=tmp["dataset"]["X"],
        right=tmp["dataset"]["Y"],
        how="left",
        left_index=True,
        right_index=True,
    )
    df.drop(columns=["TimeStamp", "WritesAvg"], inplace=True)
    df.rename(columns={"ReadsAvg": "label"}, inplace=True)
    df.reset_index(inplace=True, drop=True)

    return df

class KVDataSet(Dataset):
    """
    Custom dataset class for the training and validation dataset.
    """

    def __init__(self, x, y) -> None:
        self.x_train = torch.tensor(x.values, dtype=torch.float32)
        self.y_train = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        """
        Returns the length of the dataset.

        Returns:
        --------
        length: int; length of the dataset
        """
        return len(self.y_train)

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
        return self.x_train[idx], self.y_train[idx]


def build_dataset(dataframe, client_id) -> None:
    """
    Split the dataframe into train and test, saving as a pytorch dataset.

    Parameters:
    ------------
    dataframe: pd.DataFrame object; dataframe to be split
    client_id: str; client id

    Returns:
    ------------
    None
    """

    for c in dataframe.columns:
        dataframe[c] = dataframe[c].apply(lambda a: np.ma.log(a))

    X = dataframe.drop(columns=["label"])
    y = dataframe["label"].values

    scaler_x = StandardScaler()
    scaler_y = StandardScaler()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    X_train = pd.DataFrame(
        scaler_x.fit_transform(X_train.values),
        index=X_train.index,
        columns=X_train.columns,
    )

    X_test = pd.DataFrame(
        scaler_x.transform(X_test.values), index=X_test.index, columns=X_test.columns
    )

    y_train = scaler_y.fit_transform(y_train.reshape(-1, 1))
    y_test = scaler_y.transform(y_test.reshape(-1, 1))

    train_dataset = KVDataSet(X_train, y_train)
    test_dataset = KVDataSet(X_test, y_test)

    torch.save(train_dataset, "./trainpt/" + str(client_id) + ".pt")
    torch.save(test_dataset, "./testpt/" + str(client_id) + ".pt")


def main():
    data_path = "filtered_df/all_data"
    files = os.listdir(data_path)
    ids = [file.split(".")[0] for file in files]
    files_path = [os.path.join(data_path, file) for file in files]

    for id, csv_file in zip(ids, files_path):
        build_dataset(pd.read_csv(csv_file), id)


if __name__ == "__main__":
    main()
