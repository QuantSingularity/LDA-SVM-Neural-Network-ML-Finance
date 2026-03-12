# LDA-SVM-NeuralNetwork-ML-Finance

> **Machine Learning in Finance**
> A computational handbook demonstrating three advanced ML techniques for quantitative finance: Linear Discriminant Analysis, Support Vector Machines, and Neural Networks.

---

## Overview

This project extends the ML marketing handbook with three advanced classification techniques used on the strategies desk. Each model is demonstrated computationally using synthetic financial data designed to mimic real investment scenarios, with full analysis of advantages, disadvantages, equations, hyperparameters, and investment implications.

Three categories covered:

| Category   | Technique                          | Financial Application             |
| ---------- | ---------------------------------- | --------------------------------- |
| Category 5 | Linear Discriminant Analysis (LDA) | Corporate distress classification |
| Category 6 | Support Vector Machines (SVM)      | Market regime detection           |
| Category 7 | Neural Networks (MLP)              | Market cycle phase prediction     |

---

## Methodology

### Category 5 - Linear Discriminant Analysis (LDA)

**What it does:** LDA finds a linear combination of features that best separates two or more classes by maximizing between-class variance and minimizing within-class variance. It projects high-dimensional data onto a lower-dimensional space optimized for class separation.

**Equation:**

maximize: (mu_1 - mu_2)^2 / (sigma_1^2 + sigma_2^2)

where mu is the class mean and sigma^2 is the within-class variance.

**Financial application:** Classifying companies as Healthy vs Distressed using Liquidity Ratio and Debt-to-Equity Ratio. 400 synthetic companies are generated with realistic financial ratio distributions and a decision boundary is learned to separate the two classes.

**Key steps:**

- Generate 200 healthy companies (high liquidity, low debt) and 200 distressed companies (low liquidity, high debt)
- Train LDA with SVD solver
- Evaluate accuracy, confusion matrix, and classification report
- Visualize decision boundary over the feature space

**Hyperparameters:** solver (svd, lsqr, eigen), n_components, shrinkage, store_covariance

**Advantages:** Fast to train, interpretable decision boundary, works well when class distributions are Gaussian, provides dimensionality reduction alongside classification

**Disadvantages:** Assumes Gaussian distribution and equal covariance matrices, sensitive to outliers, linear boundary may not capture complex separations in real markets

---

### Category 6 - Support Vector Machines (SVM)

**What it does:** SVM finds the optimal hyperplane that maximizes the margin between classes. With the kernel trick, it maps data into higher-dimensional space to handle non-linear separations without computing the transformation explicitly.

**Equation:**

minimize: (1/2)||w||^2 subject to y_i(w \* x_i + b) >= 1

With RBF kernel: K(x_i, x_j) = exp(-gamma \* ||x_i - x_j||^2)

**Financial application:** Detecting market regimes - Normal Market vs Market Crash - using two non-linearly separable financial metrics generated with concentric circles, mimicking the "safe zone vs danger zone" structure of market risk.

**Key steps:**

- Generate 300 concentric-circle samples representing market regimes
- Train SVM with RBF kernel (handles non-linear boundary)
- Evaluate accuracy and classification report
- Visualize non-linear decision boundary over feature space

**Hyperparameters:** kernel (rbf, linear, poly, sigmoid), C (regularization), gamma, degree (for poly kernel), coef0

**Advantages:** Effective in high-dimensional spaces, robust to overfitting with proper C tuning, kernel trick handles complex non-linear boundaries, works well on small datasets

**Disadvantages:** Computationally expensive on large datasets, sensitive to feature scaling, hyperparameter tuning (C and gamma) is non-trivial, no probabilistic output by default

---

### Category 7 - Neural Networks (MLP)

**What it does:** A Multi-Layer Perceptron learns non-linear mappings from input to output through multiple layers of weighted connections and activation functions. Backpropagation updates weights to minimize prediction error.

**Equation:**

a^(l) = f(W^(l) \* a^(l-1) + b^(l))

where f is the activation function (ReLU, tanh), W^(l) are layer weights, b^(l) are biases.

Loss: L = -(1/n) _ sum(y_i _ log(y_hat_i) + (1 - y_i) \* log(1 - y_hat_i))

**Financial application:** Predicting market cycle phases - Bull Phase vs Bear Phase - from two interleaved moon-shaped financial indicators that cannot be separated by any linear or simple non-linear model, demonstrating the universal approximation power of neural networks.

**Key steps:**

- Generate 300 interleaved moon samples representing bull and bear market phases
- Train MLP with 2 hidden layers (20 neurons, 10 neurons), ReLU activation, Adam optimizer
- Evaluate accuracy, confusion matrix, and classification report
- Visualize learned non-linear decision boundary

**Hyperparameters:** hidden_layer_sizes, activation (relu, tanh, logistic), solver (adam, sgd, lbfgs), alpha (L2 regularization), learning_rate, max_iter, batch_size

**Advantages:** Universal function approximator, handles highly non-linear patterns, scales well with more data, flexible architecture for various financial problems

**Disadvantages:** Requires large datasets for best performance, prone to overfitting without regularization, computationally expensive, black-box nature limits interpretability, sensitive to initialization

---

### Marketing Alpha

The three techniques in this project address a critical challenge in quantitative finance: markets generate data that is rarely linearly separable. LDA provides fast, interpretable classification when asset distributions are approximately Gaussian - ideal for credit scoring and sector rotation. SVM with kernel methods handles non-linear regime boundaries without requiring explicit feature engineering - useful for detecting market stress. Neural networks approximate any non-linear function given sufficient data, making them the most powerful tool for capturing complex market dynamics at the cost of interpretability. Together, these three form an escalating toolkit from transparent to powerful.

---

### Technical Section - Hyperparameter Tuning

- **LDA solver:** SVD is default and numerically stable. For regularized LDA use shrinkage with ledoit_wolf estimation.
- **SVM C:** Controls margin width vs misclassification penalty. Low C = wide margin (underfitting), high C = narrow margin (overfitting). Tuned via grid search.
- **SVM gamma:** Controls RBF kernel radius. Low gamma = smooth boundary, high gamma = complex boundary. Co-tuned with C via cross-validation.
- **MLP hidden_layer_sizes:** Number and size of hidden layers. Start with (100,) and increase complexity if underfitting. Use early stopping to prevent overfitting.
- **MLP alpha:** L2 regularization term. Increase if model overfits. Default 0.0001.

---

### Comparing Models

| Feature                  | LDA                          | SVM                           | Neural Network                |
| ------------------------ | ---------------------------- | ----------------------------- | ----------------------------- |
| Decision Boundary        | Linear only                  | Linear or non-linear (kernel) | Highly non-linear             |
| Handles Missing Data     | Poorly                       | Poorly                        | Poorly                        |
| Interpretability         | High                         | Medium                        | Low                           |
| Scalability              | High                         | Low on large datasets         | High with GPU                 |
| Probabilistic Output     | Yes                          | No (by default)               | Yes                           |
| Requires Feature Scaling | Yes                          | Yes                           | Yes                           |
| Overfitting Risk         | Low                          | Low (with tuned C)            | High (without regularization) |
| Best For                 | Gaussian class distributions | Small, non-linear datasets    | Complex patterns, large data  |

---

## Key Findings

- LDA cleanly separated healthy from distressed companies with a linear boundary, confirming financial ratios are approximately Gaussian-distributed
- SVM with RBF kernel successfully detected non-linear market regime boundaries that no linear classifier could capture
- Neural Networks learned the most complex decision boundary, correctly classifying interleaved bull and bear market phases with high accuracy
- All three models require feature standardization for reliable performance
- Model complexity should match the complexity of the financial signal - LDA for clean separations, SVM for moderate non-linearity, MLP for complex dynamics

---

## Tech Stack

```
Python 3.x
scikit-learn
pandas
numpy
matplotlib
seaborn
```

---

## Installation

```bash
git clone https://github.com/QuantSingularity/LDA-SVM-NeuralNetwork-ML-Finance.git
cd LDA-SVM-NeuralNetwork-ML-Finance
pip install scikit-learn pandas numpy matplotlib seaborn
jupyter notebook LDA-SVM-NeuralNetwork-ML-Finance.ipynb
```

---

## Data

All data is synthetically generated using numpy and scikit-learn with fixed random seed (42) for reproducibility:

- LDA: 400 companies (200 healthy, 200 distressed) with Gaussian financial ratios
- SVM: 300 samples using make_circles representing market crash vs normal market
- Neural Network: 300 samples using make_moons representing bull vs bear market phases

---

## Topics

`machine-learning` `quantitative-finance` `lda` `svm` `neural-network` `classification` `market-regime` `distress-prediction` `decision-boundary` `feature-engineering` `python` `scikit-learn` `jupyter-notebook` `QuantSingularity`

---

## Organization

| Field        | Detail                      |
| ------------ | --------------------------- |
| Organization | QuantSingularity            |
| Domain       | Machine Learning in Finance |

---

## References

- Fisher, R.A. (1936). The Use of Multiple Measurements in Taxonomic Problems. _Annals of Eugenics_, 7(2), 179-188.
- Cortes, C., & Vapnik, V. (1995). Support-Vector Networks. _Machine Learning_, 20(3), 273-297.
- Hornik, K., Stinchcombe, M., & White, H. (1989). Multilayer Feedforward Networks are Universal Approximators. _Neural Networks_, 2(5), 359-366.
- Lopez de Prado, M. (2018). _Advances in Financial Machine Learning_. Wiley.
