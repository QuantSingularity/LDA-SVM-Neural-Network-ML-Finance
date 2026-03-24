# LDA-SVM-Neural-Network-ML-Finance

> Machine Learning in Finance

> Three advanced classification techniques demonstrated on real market data: Linear Discriminant Analysis for corporate distress detection, Support Vector Machines for market regime identification, and Multi-Layer Perceptron for market cycle phase prediction.

---

## Overview

This project demonstrates three supervised learning classifiers applied to quantitative finance problems. Each model is trained on features derived from real SPY and VIX market data downloaded via yfinance, with a fully self-contained synthetic fallback for reproducibility. The project covers the full pipeline: data acquisition, feature engineering, preprocessing, model training, evaluation, and decision boundary visualisation.

| Technique                          | Financial Application             |
| ---------------------------------- | --------------------------------- |
| Linear Discriminant Analysis (LDA) | Corporate distress classification |
| Support Vector Machine (SVM, RBF)  | Market regime detection           |
| Neural Network (MLP)               | Market cycle phase prediction     |

---

## Data

All features are derived from real market data downloaded via yfinance, with a synthetic GBM fallback when live data is unavailable.

**LDA - Corporate Distress:**

- Tickers: SPY, XLF, XLE, XLK, XLV, XLI, XLY, XLP, XLU, XLB (10 sector ETFs)
- Period: 2015-2023
- Liquidity Ratio: 12-month rolling log-return (profitability proxy)
- Solvency Ratio: 12-month realised annualised volatility (financial stress proxy)
- Label: Distressed (1) if 3-month forward return < -10%, Healthy (0) otherwise
- Sample: 400 observations (200 per class, balanced)

**SVM - Market Regime:**

- Tickers: SPY, VIX
- Period: 2010-2023
- Metric 1: SPY 20-day rolling annualised return (momentum)
- Metric 2: VIX level (fear / volatility regime)
- Label: Market Crash (1) if VIX > 25 AND SPY 20-day return < 0, Normal (0) otherwise
- Sample: 300 observations

**MLP - Market Cycle:**

- Ticker: SPY
- Period: 2010-2023
- Indicator A: RSI(14) normalised to [-1, 1]
- Indicator B: MACD signal line (12/26/9 EMA) as percentage of price
- Label: Bull Phase (1) if SPY > 200-day MA, Bear Phase (0) otherwise
- Sample: 300 observations

---

## Models

### Linear Discriminant Analysis

LDA finds a linear combination of features that maximises between-class variance relative to within-class variance, projecting data onto a lower-dimensional space optimised for class separation:

```
maximise: (mu_1 - mu_2)^2 / (sigma_1^2 + sigma_2^2)
```

where mu is the class mean and sigma^2 is the within-class variance.

**Implementation:**

- Features scaled with StandardScaler before training
- Solver: SVD (numerically stable, no matrix inversion required)
- 70/30 stratified train/test split
- Decision boundary visualised over the full feature space
- Confusion matrix reported alongside accuracy and classification report

**Hyperparameters:** solver (svd, lsqr, eigen), n_components, shrinkage, store_covariance

**Advantages:** Fast to train, interpretable linear boundary, works well when class distributions are approximately Gaussian, provides dimensionality reduction alongside classification

**Disadvantages:** Assumes Gaussian distributions and equal covariance matrices, sensitive to outliers, linear boundary cannot capture complex non-linear separations in real markets

---

### Support Vector Machine

SVM finds the optimal separating hyperplane that maximises the margin between classes. The RBF kernel maps data into a higher-dimensional space to handle non-linear boundaries without computing the transformation explicitly:

```
minimise: (1/2)||w||^2  subject to  y_i(w * x_i + b) >= 1

RBF kernel: K(x_i, x_j) = exp(-gamma * ||x_i - x_j||^2)
```

**Implementation:**

- Features scaled with StandardScaler before training
- Kernel: RBF with gamma='scale' (1 / (n_features \* X.var()))
- C=1.0 (balanced margin width vs misclassification penalty)
- probability=True for soft decision output
- Support vectors highlighted on the decision boundary plot
- 70/30 stratified train/test split

**Hyperparameters:** kernel (rbf, linear, poly, sigmoid), C (regularisation), gamma, degree (poly), coef0

**Advantages:** Effective in high-dimensional spaces, robust to overfitting with proper C tuning, kernel trick handles complex non-linear boundaries, works well on small-to-medium datasets

**Disadvantages:** Computationally expensive on large datasets, sensitive to feature scaling, hyperparameter tuning (C and gamma) is non-trivial, no probabilistic output by default without Platt scaling

---

### Neural Network (MLP)

A Multi-Layer Perceptron learns non-linear mappings through multiple layers of weighted connections and activation functions. Backpropagation minimises binary cross-entropy loss:

```
a^(l) = f(W^(l) * a^(l-1) + b^(l))

L = -(1/n) * sum(y_i * log(y_hat_i) + (1 - y_i) * log(1 - y_hat_i))
```

where f is the ReLU activation, W^(l) are layer weights, and b^(l) are biases.

**Architecture:**

```
Input (2 features)
  -> Dense(20, ReLU)
  -> Dense(10, ReLU)
  -> Dense(1, sigmoid)
```

**Implementation:**

- Features scaled with StandardScaler before training
- Solver: Adam, L2 regularisation alpha=0.01
- Early stopping with patience=20, validation_fraction=0.15
- max_iter=2000 to ensure convergence
- 70/30 stratified train/test split
- Training loss curve and confusion matrix reported

**Hyperparameters:** hidden_layer_sizes, activation (relu, tanh, logistic), solver (adam, sgd, lbfgs), alpha (L2), learning_rate, max_iter, batch_size, early_stopping

**Advantages:** Universal function approximator, handles highly non-linear patterns, flexible architecture, scales well with more data

**Disadvantages:** Requires care to avoid overfitting, computationally more expensive than LDA/SVM, black-box nature limits interpretability, sensitive to initialisation and hyperparameters

---

## Hyperparameter Notes

- **LDA solver:** SVD is the default and numerically stable. For small datasets or regularised LDA, use shrinkage with Ledoit-Wolf estimation.
- **SVM C:** Controls the trade-off between margin width and misclassification penalty. Low C gives a wide smooth margin; high C fits the training data more tightly. Tune via cross-validation.
- **SVM gamma:** Controls the RBF kernel radius. Low gamma produces a smooth boundary; high gamma produces a complex, potentially overfit boundary. Co-tune with C via grid search.
- **MLP hidden_layer_sizes:** Start with a single hidden layer and increase complexity if underfitting. Two layers of (20, 10) neurons provide sufficient capacity for 2-feature inputs.
- **MLP alpha:** L2 regularisation strength. Increase from the default (0.0001) if the model overfits. Set to 0.01 here given the small training set.

---

## Model Comparison

| Feature                  | LDA                          | SVM (RBF)                  | MLP                             |
| ------------------------ | ---------------------------- | -------------------------- | ------------------------------- |
| Decision Boundary        | Linear only                  | Non-linear (kernel)        | Highly non-linear               |
| Interpretability         | High                         | Medium                     | Low                             |
| Scalability              | High                         | Low on large datasets      | High with GPU                   |
| Probabilistic Output     | Yes                          | Yes (Platt scaling)        | Yes                             |
| Requires Feature Scaling | Yes                          | Yes                        | Yes                             |
| Overfitting Risk         | Low                          | Low (with tuned C)         | Moderate (needs regularisation) |
| Best For                 | Gaussian class distributions | Small, non-linear datasets | Complex patterns                |

---

## Key Findings

- LDA cleanly separated healthy from distressed companies using real rolling return and volatility features, confirming that distressed periods (low return, high vol) are approximately Gaussian-distributed and linearly separable from healthy periods
- SVM with RBF kernel successfully identified non-linear market regime boundaries in SPY return vs VIX space: crash regimes occupy a distinct high-vol / negative-return region that no linear classifier can capture
- The MLP learned the most complex decision boundary from RSI and MACD indicators, correctly classifying bull and bear phases driven by the cyclical and oscillating nature of technical momentum signals
- All three models require feature standardisation - raw financial magnitudes would otherwise bias distance-based and gradient-based learners
- Model complexity should match the complexity of the financial signal: LDA for linearly separable financial ratios, SVM for regime detection with non-linear boundaries, and MLP for technical indicator-driven cycle classification

---

## Tech Stack

```
Python 3.x
yfinance
scikit-learn
pandas
numpy
matplotlib
seaborn
```

---

## Installation

```bash
git clone https://github.com/QuantSingularity/LDA-SVM-Neural-Network-ML-Finance.git
cd LDA-SVM-Neural-Network-ML-Finance
pip install yfinance scikit-learn pandas numpy matplotlib seaborn
jupyter notebook LDA_SVM_Neural_Network_ML_Finance.ipynb
```

> Live SPY, VIX, and sector ETF data is fetched automatically via yfinance. If unavailable, a fully self-contained synthetic dataset is generated - no external files required.

---

## Topics

`lda` `svm` `neural-network` `mlp` `classification` `market-regime` `corporate-distress` `market-cycle` `rsi` `macd` `vix` `spy` `yfinance` `decision-boundary` `scikit-learn` `quantitative-finance` `python` `QuantSingularity`

---

## References

- Fisher, R.A. (1936). The Use of Multiple Measurements in Taxonomic Problems. _Annals of Eugenics_, 7(2), 179-188.
- Cortes, C., & Vapnik, V. (1995). Support-Vector Networks. _Machine Learning_, 20(3), 273-297.
- Hornik, K., Stinchcombe, M., & White, H. (1989). Multilayer Feedforward Networks are Universal Approximators. _Neural Networks_, 2(5), 359-366.
- Lopez de Prado, M. (2018). _Advances in Financial Machine Learning_. Wiley.
- Wilder, J.W. (1978). _New Concepts in Technical Trading Systems_. Trend Research.
