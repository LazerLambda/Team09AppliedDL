"""Modul for Debug Dataset."""

import os
import random

import pandas as pd


class DebugDataset:
    """Class for Debug."""

    def __init__(self, n: int, p: int, path: str = "TEST_DATA.gzip"):
        """Initialize Class.

        :param n: Size of dataset.
        :param p: Length of features/proteines etc..
        :param path: Path where dataset will be saved.
        """
        self.path: str = path
        self.n: int = n
        self.p: int = p
        self.data: pd.DataFrame = None

    def create_debug_dataset(self) -> None:
        """Create Dataset as .gzip File."""
        x: list = list(
            map(
                lambda e: e.join(random.choices(
                    'ABCDEFGHIKLMNOPQRSTUVWXYZ',
                    k=self.p)),
                [''] * self.n))
        y: list = random.choices([0, 1], k=self.n)
        df: pd.DataFrame = pd.DataFrame({'window': x, 'is_positive': y})

        df.to_csv(self.path, compression='gzip')

        self.data = df

    def rm_csv(self) -> None:
        """Delete Dataset."""
        os.remove(self.path)
