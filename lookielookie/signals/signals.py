import pandas as pd
import pandas_ta as ta
from abc import ABC, abstractmethod

class Signal(ABC):
    def __init__(self, data):
        self.data = data

    @abstractmethod
    def calculate(self):
        """Each subclass must implement this method to calculate the indicator."""
        pass

class Lookielookie(Signal):
    def __init__(self, data, length, multiplier):
        super().__init__(data)
        self.length = length
        self.multiplier = multiplier
    
    def __str__(self):
        return f"lookielookie_{self.length}_{self.multiplier}"

    def calculate(self):
        try:
            res = (
                self.data.ta.supertrend(length=self.length, multiplier=self.multiplier)
                .rename(columns={f"SUPERTd_{self.length}_{self.multiplier}.0": f"supertrend_{self.length}_{self.multiplier}"})
                [[f"supertrend_{self.length}_{self.multiplier}"]]
                .assign(
                    **{
                        "flag": lambda df: (df.supertrend_10_2!=df.supertrend_10_2.shift(1)).astype(int).cumsum(),
                        "cnt": lambda df: df.groupby("flag").transform(lambda f: abs(f.cumsum())),
                        self.__str__(): lambda df: df.apply(lambda f: {"supertrend": int(f["supertrend_10_2"]), "cnt": int(f["cnt"])}, axis=1)
                    }
                )
                .drop(columns=["flag",f"supertrend_{self.length}_{self.multiplier}","cnt"])
            )
        except Exception as e:
            print(str(e))
            res = None

        return res