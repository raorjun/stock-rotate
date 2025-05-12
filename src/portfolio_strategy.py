import pandas as pd
import numpy as np

class PortfolioStrategy:
    def __init__(self, returns: pd.DataFrame, labels: np.ndarray):
        self.returns = returns
        self.labels = labels
        self.tickers = returns.columns.tolist()

    def _cluster_map(self):
        clusters = {}
        for ticker, lbl in zip(self.tickers, self.labels):
            clusters.setdefault(lbl, []).append(ticker)
        return clusters

    def compute_monthly_returns(self):
        records = []
        for date, df in self.returns.resample('ME'):
            compounded = (1 + df).prod() - 1
            record = compounded.to_dict()
            record['date'] = date
            records.append(record)
        mdf = pd.DataFrame(records).set_index('date')
        return mdf

    def backtest_rotation(self):
        clusters = self._cluster_map()
        monthly = self.compute_monthly_returns()
        series = []
        dates = monthly.index
        for i in range(1, len(dates)):
            prev, curr = dates[i-1], dates[i]
            perf = {lbl: monthly.loc[prev, ticks].mean() for lbl, ticks in clusters.items()}
            best = max(perf, key=perf.get)
            series.append(monthly.loc[curr, clusters[best]].mean())
        return pd.Series(series, index=dates[1:])

    def best_cluster(self):
        clusters = self._cluster_map()
        monthly = self.compute_monthly_returns()
        last = monthly.index[-1]
        perf = {lbl: monthly.loc[last, ticks].mean() for lbl, ticks in clusters.items()}
        best = max(perf, key=perf.get)
        return best, clusters[best], perf[best]