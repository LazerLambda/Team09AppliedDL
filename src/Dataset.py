"""Dataset Module."""

import gzip
import shutil
from random import shuffle

import pandas as pd
import torch
from torch.nn import functional
from torch.utils.data import Dataset


class AminoDS(Dataset):
    """Amino dataset."""

    def __init__(
            self,
            path: str,
            dataset_type: str,
            proportion_train: float = 0.5,
            proportion_test: float = 0.2,
            proportion_hyperparam: float = 0.15,
            debug: bool = False):
        """Initialize Class.

        :param path: Path to dataset.
        :param dataset_type: Parameter for train ("train"), test ("test"),
            hyperparameter test ("hyperparam") and validation ("val") split.
        :param proportion_train: Proportion size of split dataset train.
        :param proportion_test: Proportion size of split dataset test.
        :param proportion_hyperparam: Proportion size of split dataset
            hyperparam. The remaining data will form the validation set.
        :param debug: Truncate dataset for debug purposes.
        """
        with gzip.open(path, 'rb') as f_in:
            with open('data.csv', 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

        df1: pd.DataFrame = pd.read_csv(r'data.csv')

        df1 = df1.head(n=10000) if debug else df1

        trn_len: int = int(proportion_train * len(df1)) - 1
        tst_len: int = int(proportion_test * len(df1)) - 1
        hyperparam_len: int = int(proportion_hyperparam * len(df1)) - 1

        if dataset_type == "train":
            df1 = df1[0:trn_len]

        elif dataset_type == "test":
            df1 = df1[
                (trn_len + 1):(trn_len + tst_len)].reset_index()

        elif dataset_type == "hyperparam":
            df1 = df1[
                (trn_len + tst_len + 1):(trn_len + tst_len + hyperparam_len)]\
                .reset_index()

        elif dataset_type == "val":
            df1 = df1[
                (trn_len + tst_len + hyperparam_len + 1)::].reset_index()

        else:
            raise ValueError("dataset_type is not correctly specified.")

        # define input string
        data: pd.DataFrame = df1['window']

        # define universe of possible input values
        alphabet: str = 'ABCDEFGHIKLMNOPQRSTUVWXYZ'

        # define a mapping of chars to integers
        char_to_int: dict = dict((c, i) for i, c in enumerate(alphabet))

        # create an empty list with lenth of data
        integer_encoded: list = [[] for x in range(len(data))]

        for j in range(0, len(data)):
            integer_encoded[j] = [char_to_int[char] for char in data[j]]

        integer_encoded = torch.tensor(integer_encoded)
        final_data: torch.Tensor = functional.one_hot(
            integer_encoded,
            num_classes=-1)

        labels = df1['is_positive']

        train_data = []
        for i in range(len(final_data)):
            train_data.append([final_data[i], labels[i]])
        self.result = train_data

    def shuffle(self) -> None:
        """Shuffle Dataset."""
        shuffle(self.result)

    def __len__(self) -> int:
        """Return Length of Dataset.

        :return: Length of dataset.
        """
        return len(self.result)

    def __getitem__(self, idx) -> torch.Tensor:
        """Return Item at Position idx.

        :param idx: Index of item to be returned.

        :return: Tensor at position idx.
        """
        return self.result[idx]
