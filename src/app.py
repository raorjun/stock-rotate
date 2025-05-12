import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from pca import PCA
from spectral_clustering import SpectralClustering
from portfolio_strategy import PortfolioStrategy
from evaluator import PortfolioEvaluator

def main():
    st.title("S&P 500 Cluster Rotation Strategy App")

    st.sidebar.header("Configuration")
    n_clusters = st.sidebar.slider("Number of Clusters", 2, 10, 5)
    n_neighbors = st.sidebar.slider("k-NN Neighbors", 2, 20, 10)
    lookback_months = st.sidebar.slider("Lookback Window (months)", 1, 12, 3)

    DATA_PATH = Path(__file__).resolve().parent.parent / 'data' / 'sp500_clean_data.csv'
    if not DATA_PATH.exists():
        st.error(f"Data file not found: {DATA_PATH}")
        return

    df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    daily_returns = df.pct_change().dropna()
    X = daily_returns.T.values
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)


    sc = SpectralClustering(n_clusters=n_clusters, n_neighbors=n_neighbors)
    labels = sc.fit_predict(X)


    fig, ax = plt.subplots()
    for lbl in np.unique(labels):
        pts = X_pca[labels == lbl]
        ax.scatter(pts[:, 0], pts[:, 1], label=f"Cluster {lbl}")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Spectral Clustering on PCA Projection")
    ax.legend()
    st.pyplot(fig)

    strategy = PortfolioStrategy(daily_returns, labels)
    monthly = strategy.compute_monthly_returns()
    clusters_map = strategy._cluster_map()

    window = monthly.index[-lookback_months:]
    perf = {}
    for lbl, tickers in clusters_map.items():
        block = monthly.loc[window, tickers]
        perf[lbl] = block.values.mean()
    best_lbl = max(perf, key=perf.get)

    st.subheader(f"Best Cluster over Last {lookback_months} Months")
    col1, col2 = st.columns(2)
    col1.metric("Cluster", best_lbl)
    col2.metric("Avg Return", f"{perf[best_lbl]:.2%}")

    tickers = clusters_map[best_lbl]
    n_cols = 4
    n_rows = (len(tickers) + n_cols - 1) // n_cols
    grid = []
    for r in range(n_rows):
        row = []
        for c in range(n_cols):
            idx = c * n_rows + r
            row.append(tickers[idx] if idx < len(tickers) else "")
        grid.append(row)
    ticker_df = pd.DataFrame(grid, columns=[f"Col {i+1}" for i in range(n_cols)])
    st.table(ticker_df)

    rotation_returns = strategy.backtest_rotation()
    evaluator = PortfolioEvaluator(rotation_returns)
    metrics = evaluator.evaluate()

    metrics_df = pd.DataFrame(metrics.items(), columns=["Metric", "Value"])

    st.subheader("Rotation Strategy Performance (1-Month Lookback)")
    st.table(metrics_df)

if __name__ == "__main__":
    main()
