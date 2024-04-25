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
from lookielookie.utils.utils import topological_sort
from lookielookie.indicators import indicators as inds
from lookielookie.signals import signals as sigs


class Backbone:
    def __init__(self):
        self.db = self._db_connect()
        # self.tickers = self._get_tickers()
        self.tickers = ["AMT","UBER"]
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
                .rename(columns={"symbol":"ticker"})
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

    def ohlcv(self):
        new_last_updated = INIT_TS
        updated_cnt = 0
        df_all = self.get_ohlcv_from_yahoo(start=self.last_updated + td(days=1))

        if df_all is None:
            return False
        else:
            all_tickers = df_all.ticker.unique().tolist()

        for ticker in tqdm(all_tickers):
            df_ticker = df_all.loc[lambda df: df.ticker==ticker]
            df_db = pd.DataFrame(self.db.timeseries.find({"ticker": ticker}, {"_id": 0, "indicators": 0, "signals": 0, "ticker": 0}))
            last_ts = INIT_TS if df_db.shape[0]==0 else df_db.date.iloc[-1]
                
            if last_ts < self.last_updated:
                # DB is outdated
                df_ticker = self.get_ohlcv_from_yahoo(start=last_ts + td(days=1), tickers=[ticker])
            elif last_ts > self.last_updated:
                # Ticker was already updated
                continue
                
            df_ticker = (
                df_ticker
                .assign(
                    indicators = lambda df: df["ticker"].apply(lambda f: {}),
                    signals = lambda df: df["ticker"].apply(lambda f: {})
                )
                .reset_index()
            )
            last_ts_ohlcv = df_ticker.date.iloc[-1]
            new_last_updated = max(last_ts_ohlcv, new_last_updated)

            if df_ticker.shape[0] > 0:
                self.db.timeseries.insert_many(df_ticker.to_dict("records"))
                updated_cnt += 1
        
        if updated_cnt > 0:
            self.db.config.update_one({"last_updated": {"$exists":True}}, {"$set": {"last_updated": new_last_updated}})
            
        return True

    def indicators(self):
        indicators = topological_sort(list(self.db.indicators.find({},{"_id": 0})))
        ind_names = [x["name"] for x in indicators]
        
        for ticker in tqdm(self.tickers):
            df = pd.DataFrame(self.db.timeseries.find({"ticker": ticker}).sort("date", 1))
            if df.shape[0] == 0:
                continue

            df["indicators"] = df.indicators.apply(lambda f: {} if pd.isna(f) else f)
            df["signals"] = df.signals.apply(lambda f: {} if pd.isna(f) else f)
            inds_db = [k for k,v in df.sample(1)["indicators"].iloc[0].items()]
            inds_cont = [x for x in ind_names if x in inds_db]
            inds_remove = [x for x in inds_db if x not in ind_names]
            inds_add = [x for x in ind_names if x not in inds_db]
            update_indices = df.loc[lambda df: df.indicators=={}].index

            if len(inds_cont) > 0 and len(update_indices) > 0:
                df_indices = None
                for d in [x for x in indicators if x["name"] in inds_cont]:
                    cls = getattr(inds, d["class"])
                    indicator = cls(data=df[d["requires"]], **d["params"])
                    df_indices = indicator.calculate() if df_indices is None else pd.concat([df_indices, indicator.calculate()], axis=1)
                df.iloc[update_indices,df.columns.get_loc("indicators")] = df_indices.iloc[update_indices].to_dict("records")

            # Remove
            if len(inds_remove) > 0:
                update_indices = df.index
                df = (
                    df
                    .assign(
                        indicators = lambda df: df.loc[lambda df: ~pd.isna(df.indicators)].indicators.apply(lambda f: {k:v for k,v in f.items() if k not in inds_remove})
                    )
                )
            
            if len(inds_add) > 0:
                update_indices = df.index
                for d in [x for x in indicators if x["name"] in inds_add]:
                    cls = getattr(inds, d["class"])
                    indicator = cls(data=df[d["requires"]], **d["params"])
                    df = pd.concat([df, indicator.calculate()], axis=1)
                df = (
                    df
                    .assign(
                        inds_add = lambda df: df[inds_add].to_dict("records"),
                        indicators=lambda df: df.apply(lambda f: {**f["indicators"], **f["inds_add"]}, axis=1)
                    )
                    .drop(columns=["inds_add"]+inds_add)
                )
            
            if len(update_indices) < 5:
                for ui in update_indices:
                    document = df.iloc[ui].to_dict()
                    self.db.timeseries.update_one(
                        {"_id": document["_id"]},
                        {"$set": {"indicators": document["indicators"]}}
                    )
            else:
                docs = df.iloc[update_indices][["date","ticker","open","high","low","close","adjclose","volume","indicators","signals"]].to_dict("records")
                self.db.timeseries.delete_many({"_id": {"$in": df.iloc[update_indices]["_id"].to_list()}})
                self.db.timeseries.insert_many(docs)

        return True

    def signals(self):
        signals = list(self.db.signals.find({},{"_id": 0}))
        sig_names = [x["name"] for x in signals]

        for ticker in tqdm(self.tickers):
            df = pd.DataFrame(self.db.timeseries.find({"ticker": ticker}).sort("date", 1))
            if df.shape[0] == 0:
                continue
            
            df["signals"] = df.signals.apply(lambda f: {} if pd.isna(f) else f)
            sigs_db = [k for k,v in df.sample(1)["signals"].iloc[0].items()]
            sigs_cont = [x for x in sig_names if x in sigs_db]
            sigs_remove = [x for x in sigs_db if x not in sig_names]
            sigs_add = [x for x in sig_names if x not in sigs_db]
            update_signals = df.loc[lambda df: df.signals=={}].index

            if len(sigs_cont) > 0 and len(update_signals) > 0:
                df_signals = None
                for d in [x for x in signals if x["name"] in sigs_cont]:
                    cls = getattr(sigs, d["class"])
                    signal = cls(data=df[d["requires"]], **d["params"])
                    df_signals = signal.calculate() if df_signals is None else pd.concat([df_signals, signal.calculate()], axis=1)
                df.iloc[update_signals,df.columns.get_loc("signals")] = df_signals.iloc[update_signals].to_dict("records")

            # Remove
            if len(sigs_remove) > 0:
                update_signals = df.index
                df = (
                    df
                    .assign(
                        signals = lambda df: df.loc[lambda df: ~pd.isna(df.signals)].signals.apply(lambda f: {k:v for k,v in f.items() if k not in sigs_remove})
                    )
                )
            
            if len(sigs_add) > 0:
                update_signals = df.index
                for d in [x for x in signals if x["name"] in sigs_add]:
                    cls = getattr(sigs, d["class"])
                    signal = cls(data=df[d["requires"]], **d["params"])
                    df = pd.concat([df, signal.calculate()], axis=1)
                df = (
                    df
                    .assign(
                        sigs_add = lambda df: df[sigs_add].to_dict("records"),
                        signals=lambda df: df.apply(lambda f: {**f["signals"], **f["sigs_add"]}, axis=1)
                    )
                    .drop(columns=["sigs_add"]+sigs_add)
                )
            
            if len(update_signals) < 5:
                for ui in update_signals:
                    document = df.iloc[ui].to_dict()
                    self.db.timeseries.update_one(
                        {"_id": document["_id"]},
                        {"$set": {"signals": document["signals"]}}
                    )
            else:
                docs = df.iloc[update_signals][["date","ticker","open","high","low","close","adjclose","volume","indicators","signals"]].to_dict("records")
                self.db.timeseries.delete_many({"_id": {"$in": df.iloc[update_signals]["_id"].to_list()}})
                self.db.timeseries.insert_many(docs)

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
    # bb.ohlcv()
    # bb.indicators()
    # bb.signals()