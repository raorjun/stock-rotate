import numpy as np
import pandas as pd

class PortfolioEvaluator:
    def __init__(self, returns: pd.Series):
        self.returns = returns

    def cumulative(self) -> float:
        cum = (1 + self.returns).cumprod() - 1
        return cum.iloc[-1]

    def sharpe(self, risk_free=0.0) -> float:
        excess = self.returns - risk_free/12
        return np.sqrt(12) * excess.mean() / excess.std()

    def drawdown(self) -> float:
        cum = (1 + self.returns).cumprod()
        peak = cum.cummax()
        dd = (cum - peak) / peak
        return dd.min()

    def evaluate(self) -> dict:
        return {
            'Cumulative Return': self.cumulative(),
            'Sharpe Ratio': self.sharpe(),
            'Max Drawdown': self.drawdown()
        }