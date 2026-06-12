# BasicDT

A standalone, high-performance, histogram-based axis-aligned decision tree engine.

`BasicDT` extracts the optimized pre-binned histogram-based decision tree core from OQBoost, removing all oblique projection, coordinate descent, and random projection logic to deliver pure axis-aligned gradient boosted trees.

## Features

- **Extremely Fast**: Employs cache-blocked sample-parallelism and dynamic thread scaling in C++ (OpenMP).
- **Native NaN Handling**: Missing values (NaNs) are automatically handled and imputed during binning.
- **Native Categorical Support**: Supports categorical features out of the box using rank-encoded target gradients.
- **Scikit-Learn Compatible**: Implements standard `fit`, `predict`, and `predict_proba` APIs.

## Installation

```bash
# Install in editable mode
pip install -e .
```

## Usage

```python
from basicdt import BasicDTClassifier

# Initialize model
clf = BasicDTClassifier(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8
)

# Fit model
clf.fit(X_train, y_train)

# Predict probabilities
probas = clf.predict_proba(X_test)
```
