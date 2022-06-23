"""Dataset Module."""

import gzip
import shutil

import pandas as pd
import torch
from torch.nn import functional
from torch.utils.data import Dataset


class AminoDS(Dataset):
    """Amino dataset."""

    def __init__(
            self,
            path: str,
            train: bool,
            proportion: float = 0.8,
            debug: bool = False):
        """Initialize Class.

        :param path: Path to dataset.
        :param train: Parameter for test train split.
        :param proportion: Proportion of tain-test split.
        :param debug: Truncate dataset for debug purposes.
        """
        assert isinstance(train, bool),\
            "ERROR:`train` must be of type boolean."

        with gzip.open(path, 'rb') as f_in:
            with open('data.csv', 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

        df1: pd.DataFrame = pd.read_csv(r'data.csv')
        df1 = df1.head(n=10000) if debug else df1

        train_len: int = int(proportion * len(df1)) - 1
        if train:
            df1 = df1[0:train_len]
        else:
            df1 = df1[train_len::].reset_index()

        # define input string
        data: pd.DataFrame = df1['window']

        # define universe of possible input values
        alphabet: str = 'ABCDEFGHIKLMNOPQRSTUVWXYZ'

        # define a mapping of chars to integers
        char_to_int: dict = dict((c, i) for i, c in enumerate(alphabet))

        # create an empty list with lenth of data
        integer_encoded: list = [[] for x in range(len(data))]

        # TODO: Padding for non occuring characters
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
