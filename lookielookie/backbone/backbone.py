from pymongo import MongoClient
from datetime import datetime as dt
from datetime import timedelta as td
import pandas as pd
import numpy as np
from yahooquery import Ticker
from tqdm import tqdm
from ta.volatility import AverageTrueRange
from ta.trend import EMAIndicator

from lookielookie.utils.constants import MONGO_URI, SP500_URL, R1000_URL, IGNORE, INIT_TS


class Backbone:
    def __init__(self):
        self.db = self._db_connect()
        self.tickers = self._get_tickers()
        print(MONGO_URI)
        self.last_updated = self.db.config.find_one({"last_updated": {"$exists":True}})["last_updated"]

    def _db_connect(self):
        client = MongoClient(MONGO_URI)
        return client.superguppy

    def _get_tickers(self):
        sp500_tickers = pd.read_html(SP500_URL)[0]['Symbol'].tolist()
        r1000_tickers = pd.read_html(R1000_URL)[2]["Ticker"].to_list()

        return list(set(sp500_tickers + r1000_tickers))

    def get_ohlcv_from_yahoo(self, start, tickers=None):
        dt_today = dt.today()
        dt_today = dt(dt_today.year, dt_today.month, dt_today.day)
        
        tickers = self.tickers if tickers is None else tickers
        ts = Ticker(tickers, asynchronous=False)
        
        if start:
            df = ts.history(start=start, end=dt_today, interval='1d').reset_index().round(2)
        else:
            df = ts.history(period='max', end=dt_today, interval='1d').reset_index().round(2)
        
        if df.shape[0] > 0:
            df = (
                df
                .assign(
                    date = lambda df: pd.to_datetime(df.date)
                )
                .set_index("date")
            )

            return df
        else:
            return None

    def get_ohlcv_from_db(self, ticker=None):
        if ticker is None:
            return pd.DataFrame(self.db.timeseries.find({},{"_id":0,"indicators":0}))
        else:
            return pd.DataFrame(self.db.timeseries.find({"ticker":ticker},{"_id":0,"indicators":0}))

    def signals(self, df):
        df = (
            df
            .assign(
                FU = lambda df: df.apply(lambda df: df.ema_3 > df.ema_6 > df.ema_9 > df.ema_12 > df.ema_15 > df.ema_18 > df.ema_21, axis=1),
                FD = lambda df: df.apply(lambda df: df.ema_3 < df.ema_6 < df.ema_9 < df.ema_12 < df.ema_15 < df.ema_18 < df.ema_21, axis=1),
                SU = lambda df: df.apply(lambda df: df.ema_24 > df.ema_27 > df.ema_30 > df.ema_33 > df.ema_36 > df.ema_39 > df.ema_42 > df.ema_45 > df.ema_48 > df.ema_51 > df.ema_54 > df.ema_57 > df.ema_60 > df.ema_63 > df.ema_66, axis=1),
                SD = lambda df: df.apply(lambda df: df.ema_24 < df.ema_27 < df.ema_30 < df.ema_33 < df.ema_36 < df.ema_39 < df.ema_42 < df.ema_45 < df.ema_48 < df.ema_51 < df.ema_54 < df.ema_57 < df.ema_60 < df.ema_63 < df.ema_66, axis=1),
                LONG = lambda df: np.where(np.logical_and(df.FU,df.SU), True, False),
                SHORT = lambda df: np.where(np.logical_and(df.FD,df.SD), True, False),
                SIDE = lambda df: np.where(np.logical_and(df.LONG==False,df.SHORT==False), True, False),
            )
        )

        return df[["FU","FD","SU","SD","LONG","SHORT","SIDE"]]

    def timeseries(self):
        new_last_updated = INIT_TS
        updated_cnt = 0
        df_all = self.get_ohlcv_from_yahoo(start=self.last_updated + td(days=1))

        if df_all is None:
            return False
        else:
            all_tickers = df_all.symbol.unique().tolist()

        for ticker in tqdm(all_tickers):
            df_ticker = df_all.loc[lambda df: df.symbol==ticker]
            df_db = pd.DataFrame(self.db.timeseries.find({"ticker": ticker}, {"_id": 0, "indicators": 0, "signals": 0, "ticker": 0}))
            
            if df_db.shape[0] > 0:
                # Ticker was fetched before
                df_db = df_db.set_index("date")
            else:
                # Ticker was not fetched before
                if ticker not in IGNORE:
                    df_db = pd.DataFrame(columns=["date","open","high","low","close","adjclose","volume"]).set_index("date")
                else:
                    continue
            
            last_ts = INIT_TS if len(df_db.index)==0 else df_db.index[-1]
                
            if last_ts < self.last_updated:
                # DB is outdated
                df_ticker = self.get_ohlcv_from_yahoo(start=last_ts + td(days=1), tickers=[ticker])
            elif last_ts > self.last_updated:
                # Ticker was already updated
                continue

            df_ohlcv = pd.concat([df_db, df_ticker])
            last_ts_ohlcv = df_ohlcv.index[-1]
            new_last_updated = max(last_ts_ohlcv, new_last_updated)

            # Indicators
            # EMAs
            emas = (
                pd
                .concat([EMAIndicator(close=df_ohlcv["adjclose"], window=window).ema_indicator() for window in range(3,69,3)], axis=1)
                .round(2)
            )
            
            # ATR
            try:
                atr = (
                    pd.DataFrame(
                        AverageTrueRange(df_ohlcv["high"], df_ohlcv["low"], df_ohlcv["adjclose"])
                        .average_true_range()
                    )
                    .round(2)
                )
            except IndexError:
                atr = pd.Series(np.nan, index=df_ohlcv.index, name="atr")
            
            df_ind = pd.concat([emas, atr], axis=1)
            df = pd.concat(
                [
                    df_ohlcv.loc[lambda df: df.index > last_ts],
                    df_ind.loc[lambda df: df.index > last_ts]
                ],
                axis=1
            )
            
            # Signals
            df_sig = self.signals(df)
            df = pd.concat(
                [
                    df,
                    df_sig
                ],
                axis=1
            )
            
            # Insert
            df_insert = (
                df
                .assign(
                    ticker=ticker,
                    indicators=lambda df: df.apply(lambda df: {**{f"ema_{i}":df[f"ema_{i}"] for i in range(3,69,3)},**{"atr":df["atr"]}}, axis=1),
                    signals=lambda df: df.apply(lambda df: {c:df[c] for c in ["FU","FD","SU","SD","LONG","SHORT","SIDE"]}, axis=1)
                )
                .drop(columns=[f"ema_{i}" for i in range(3,69,3)]+["atr"]+["FU","FD","SU","SD","LONG","SHORT","SIDE"])
                .reset_index()
                .rename(columns={"index":"date"})
                [["date","ticker","open","high","low","close","adjclose","volume","indicators","signals"]]
            )

            if df_insert.shape[0] > 0:
                self.db.timeseries.insert_many(df_insert.to_dict("records"))
                updated_cnt += 1
        
        if updated_cnt > 0:
            self.db.config.update_one({"last_updated": {"$exists":True}}, {"$set": {"last_updated": new_last_updated}})
            
        return True
    
    def fundamentals(self):
        ts = Ticker(self.tickers, asynchronous=False)
        all_modules = ts.all_modules

        for ticker, d in tqdm(all_modules.items()):
            if type(d) != dict:
                print(f"Skipped {ticker}")
                continue
            
            self.db.fundamentals.delete_one({"quoteType.symbol": ticker})
            self.db.fundamentals.insert_one(d)
        
        return True

if __name__=="__main__":
    bb = Backbone()
    bb.timeseries()