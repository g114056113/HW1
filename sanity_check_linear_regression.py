# sanity_check_linear_regression.py
# Minimal (non-Streamlit) test to verify your Python + NumPy install works.
# Run with:  python sanity_check_linear_regression.py

import numpy as np

rng = np.random.default_rng(42)
n = 50
a_true, b_true, noise = 2.0, 1.0, 0.5

x = rng.uniform(-5, 5, size=n)
y = a_true * x + b_true + rng.normal(0, noise, size=n)

X = np.column_stack([np.ones_like(x), x])
beta_hat, *_ = np.linalg.lstsq(X, y, rcond=None)
b_hat, a_hat = beta_hat

print("Estimated a (slope):", a_hat)
print("Estimated b (intercept):", b_hat)
print("Should be close to a=2.0, b=1.0")
