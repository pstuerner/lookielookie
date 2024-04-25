import numpy as np
import pandas as pd
import pandas_ta as ta
from abc import ABC, abstractmethod

class Indicator(ABC):
    def __init__(self, data):
        self.data = data

    @abstractmethod
    def calculate(self):
        """Each subclass must implement this method to calculate the indicator."""
        pass

class EMA(Indicator):
    def __init__(self, data, length):
        super().__init__(data)
        self.length = length
    
    def __str__(self):
        return f"ema_{self.length}"

    def calculate(self):
        try:
            res = (
                pd.DataFrame(
                    self.data.ta.ema(length=self.length)
                )
                .round(2)
                .rename(columns={f"EMA_{self.length}": self.__str__()})
            )
        except Exception as e:
            print(str(e))
            res = None

        return res


class ATR(Indicator):
    def __init__(self, data, length):
        super().__init__(data)
        self.length = length
    
    def __str__(self):
        return f"atr_{self.length}"

    def calculate(self):
        if self.data.shape[0] < self.length:
            return pd.DataFrame({self.__str__(): [np.nan]*self.data.shape[0]})
        
        try:
            res = (
                pd.DataFrame(
                    self.data.ta.atr(length=self.length)
                )
                .round(2)
                .rename(columns={f"ATRr_{self.length}": self.__str__()})
            )
        except Exception as e:
            print(str(e))
            res = None

        return res


class Supertrend(Indicator):
    def __init__(self, data, length, multiplier):
        super().__init__(data)
        self.length = length
        self.multiplier = multiplier
    
    def __str__(self):
        return f"supertrend_{self.length}_{self.multiplier}"

    def calculate(self):
        try:
            res = self.data.ta.supertrend(length=self.length, multiplier=self.multiplier)
        except Exception as e:
            print(str(e))
            res = None

        return res