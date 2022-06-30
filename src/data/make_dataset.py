import os
import random
import string

import pandas as pd


class DebugDataset:

    def __init__(self, n: int, p: int, path: str = "TEST_DATA.gzip"):
        self.path: str = path
        self.n: int = n
        self.p: int = p
        self.data: pd.DataFrame = None

    def create_debug_dataset(self) -> any:
        x: list = list(
            map(
                lambda e:e.join(random.choices(
                    'ABCDEFGHIKLMNOPQRSTUVWXYZ',
                    k=self.p)),
                [''] * self.n))
        y: list = random.choices([0,1], k=self.n)
        df: pd.DataFrame = pd.DataFrame({'window': x, 'is_positive': y})

        df.to_csv(self.path, compression='gzip')

        self.data = df

    def rm_csv(self) -> None:
        os.remove(self.path)


# if __name__ == "__main__":
# test  = DebugDataset(10, 5)
# test.create_debug_dataset()
# #     # test.rm_csv()


