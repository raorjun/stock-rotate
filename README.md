# S&P 500 Cluster Rotation Strategy

A (not good) momentum-based trading dashboard built with **Streamlit**. This app utilizes machine learning to identify structural stock clusters and execute a rotation strategy based on historical performance.

**Live App:** [stock-cluster.streamlit.app](https://stock-cluster.streamlit.app/)

## Strategy Methodology
The engine utilizes a sophisticated pipeline to filter market noise and identify leadership:

1.  **Data Acquisition:** Fetched daily **Adjusted Closing Prices** via Yahoo Finance. This "Adjusted" price is critical as it accounts for dividends and stock splits, providing the true total return.
2.  **Dimensionality Reduction (PCA):** Principal Component Analysis is used to extract the primary "Eigen-factors" driving the S&P 500, reducing noise before clustering.
3.  **Spectral Clustering:** Unlike standard K-Means, Spectral Clustering uses the connectivity of the stock correlation matrix to find "manifolds" or natural communities of stocks that move together structurally.
4.  **Rotation Logic:** The strategy evaluates clusters over a **Lookback Window** (e.g., 3 months) and rotates capital into the top-performing cluster for the subsequent month.

## Performance (1-Month Lookback)

| Metric | Value |
| :--- | :--- |
| **Cumulative Return** | 0.2315 (23.15%) |
| **Sharpe Ratio** | 0.4405 |
| **Max Drawdown** | -0.2578 |

## Tech Stack
* **Frontend:** [Streamlit](https://streamlit.io/)
* **Data:** `yfinance`, `pandas`
* **ML:** `scikit-learn` (PCA, Spectral Clustering)
* **Visuals:** `matplotlib`

## How to Use
1. **Set Parameters:** Use the sidebar to adjust the Lookback Window, number of clusters, and k-NN neighbors.
2. **Analyze Clusters:** View the "Best Cluster" summary to see which tickers are currently leading the market.
3. **Backtest:** Review the Rotation Strategy Performance table to evaluate historical effectiveness.

---
*Disclaimer: For educational purposes only. Past performance does not guarantee future results. **THIS DOES NOT WORK AND WAS MADE FOR MY OWN EDUCATION.***
