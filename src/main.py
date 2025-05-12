import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from pca import PCA
from spectral_clustering import SpectralClustering
from portfolio_strategy import PortfolioStrategy
from evaluator import PortfolioEvaluator


def main():
    SRC_DIR = Path(__file__).resolve().parent
    ROOT_DIR = SRC_DIR.parent
    DATA_PATH = ROOT_DIR / 'data' / 'sp500_clean_data.csv'

    print("Loading data from:", DATA_PATH)
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Data file not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    daily_returns = df.pct_change().dropna()
    X = daily_returns.T.values

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    sc = SpectralClustering(n_clusters=5, n_neighbors=10)
    labels = sc.fit_predict(X)

    plt.figure()
    for lbl in np.unique(labels):
        pts = X_pca[labels == lbl]
        plt.scatter(pts[:, 0], pts[:, 1], label=f'Cluster {lbl}')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('Clusters in PCA Space')
    plt.legend()
    plt.show()

    strategy = PortfolioStrategy(daily_returns, labels)
    lookback = int(input("Enter lookback window (months): "))
    clusters = strategy._cluster_map()
    monthly = strategy.compute_monthly_returns()

    window = monthly.index[-lookback:]

    perf = {}
    for lbl, tickers in clusters.items():
        block = monthly.loc[window, tickers]
        perf[lbl] = block.values.mean()
    best = max(perf, key=perf.get)
    print(f"\nBest cluster over the past {lookback} months: Cluster {best}")
    print(f"Average return: {perf[best]:.4f}")
    print("Tickers:", clusters[best])

if __name__ == "__main__":
    main()