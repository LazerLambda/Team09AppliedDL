import os
import random
import string

import pandas as pd


class DebugDataset:

    def __init__(self):
        self.title: str = os.path.abspath(__file__) + ".TEST_DATA.csv"

    def create_debug_dataset(self, n: int, p: int) -> any:
        x: list = list(
            map(
                lambda e:e.join(random.choices(
                    string.ascii_uppercase + string.digits,
                    k=p)),
                [''] * n))
        y: list = random.choices([0,1], k=n)
        df: pd.DataFrame = pd.DataFrame({'x': x, 'y': y})

        df.to_csv(self.title)

    def rm_csv(self) -> None:
        os.remove(self.title)

if __name__ == "__main__":
    test  = DebugDataset()
    test.create_debug_dataset(10, 5)
    test.rm_csv()


