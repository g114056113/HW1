# HW1_CRISP_DM_Linear_Regression_Streamlit.py
# -------------------------------------------------------------
# HW1 — Simple Linear Regression with CRISP‑DM (Streamlit App)
# This single-file Streamlit app walks through CRISP‑DM steps while solving
# a simple linear regression y = a*x + b + noise on synthetic data.
#
# How to run (from VS Code Terminal or PowerShell):
#   1) pip install -r requirements.txt
#   2) streamlit run "HW1_CRISP_DM_Linear_Regression_Streamlit.py"
#
# If your path includes non-ASCII characters, keep the filename exactly as above
# and run the command from the same directory as this file.
#
# Tested on Python 3.9–3.12

import os
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# Make relative paths (if any) work even when run from VS Code/PowerShell.
try:
    os.chdir(os.path.dirname(__file__))
except Exception:
    pass

# ----------------------------
# Page / App Configuration
# ----------------------------
st.set_page_config(
    page_title="HW1 · CRISP‑DM Linear Regression",
    page_icon="📈",
    layout="wide",
)

st.title("📈 HW1: Simple Linear Regression · CRISP‑DM")
st.caption(
    "This app demonstrates the full CRISP‑DM process on a synthetic linear dataset: "
    "y = a·x + b + noise. Adjust parameters in the sidebar and see the results update."
)

# ----------------------------
# Sidebar Controls (Problem Setup / Data Generation)
# ----------------------------
st.sidebar.header("🔧 Experiment Controls")

true_a = st.sidebar.slider("True slope a", min_value=-10.0, max_value=10.0, value=2.0, step=0.1)
true_b = st.sidebar.slider("True intercept b", min_value=-10.0, max_value=10.0, value=1.0, step=0.1)
noise_std = st.sidebar.slider("Noise std (σ)", min_value=0.0, max_value=10.0, value=2.0, step=0.1)
n_points = st.sidebar.slider("Number of points", min_value=10, max_value=2000, value=200, step=10)

x_min, x_max = st.sidebar.slider("x range", min_value=-50, max_value=50, value=(-10, 10))
seed = st.sidebar.number_input("Random seed", min_value=0, max_value=10000, value=42, step=1)

st.sidebar.markdown("---")
st.sidebar.subheader("Download")
include_header = st.sidebar.checkbox("Include CSV header", value=True)

# ----------------------------
# (1) CRISP‑DM: Business Understanding
# ----------------------------
st.header("1) Business Understanding 🧭")
st.markdown(
    """
**Goal**: Given observed data pairs (x, y), we want to model the relationship
between x (independent variable) and y (dependent variable) with a simple linear
model: $y \\approx a\\,x + b$. 

**Why**: Linear regression is widely used to quantify trends, forecast outcomes,
and explain how changes in x relate to changes in y.

**Success criteria**: Low prediction error (e.g., small MSE/MAE), sensible
parameters $(a, b)$ that are close to the true underlying process if known, and
residuals with no obvious patterns (indicating model adequacy).
"""
)

st.divider()

# ----------------------------
# (2) CRISP‑DM: Data Understanding
# ----------------------------
np.random.seed(seed)
X = np.random.uniform(low=x_min, high=x_max, size=n_points)
noise = np.random.normal(loc=0.0, scale=noise_std, size=n_points)
Y = true_a * X + true_b + noise

df = pd.DataFrame({"x": X, "y": Y})

st.header("2) Data Understanding 🔍")
st.markdown(
    f"""
**Data source**: Synthetic.

**Generation**: $x \\sim \\mathcal{{U}}({x_min}, {x_max})$, noise $\\sim \\mathcal{{N}}(0, {noise_std}^2)$,
then $y = {true_a}\\,x + {true_b} + \\text{{noise}}$.

**Shape**: {df.shape[0]} rows × {df.shape[1]} columns.
    """
)

col_du1, col_du2 = st.columns([2, 1], gap="large")
with col_du1:
    st.dataframe(df.head(20), use_container_width=True)
with col_du2:
    fig_scatter, ax = plt.subplots()
    ax.scatter(df["x"], df["y"], alpha=0.7)
    ax.set_title("Raw Data: y vs x")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    st.pyplot(fig_scatter)
    plt.close(fig_scatter)

st.divider()

# ----------------------------
# (3) CRISP‑DM: Data Preparation
# ----------------------------
st.header("3) Data Preparation 🧹")
st.markdown(
    """
We construct the design matrix $X_\\text{design}$ with a bias column of ones for the intercept:
$X_\\text{design} = [\\mathbf{1}, x]$.

For this simple case, no missing values or categorical features exist; we keep the
raw scale to retain interpretability.
"""
)

X_design = np.column_stack([np.ones_like(X), X])  # shape: (n, 2)
y_vec = Y.reshape(-1, 1)                          # shape: (n, 1)

st.divider()

# ----------------------------
# (4) CRISP‑DM: Modeling
# ----------------------------
st.header("4) Modeling 🧪")
st.markdown(
    """
We fit parameters $(b, a)$ by minimizing the sum of squared residuals. Using the
normal equation solution via least squares:

$\\hat{\\beta} = (X^\\top X)^{-1} X^\\top y$  \\; (solved numerically with `numpy.linalg.lstsq`).

`\\hat{\\beta}[0]` is the intercept $\\hat{b}$ and `\\hat{\\beta}[1]` is the slope $\\hat{a}$.
"""
)

beta_hat, residuals, rank, s = np.linalg.lstsq(X_design, y_vec, rcond=None)

b_hat = float(beta_hat[0, 0])
a_hat = float(beta_hat[1, 0])

y_pred = (a_hat * X + b_hat)
resid = Y - y_pred

st.subheader("Estimated Parameters")
st.write({"a_hat (slope)": a_hat, "b_hat (intercept)": b_hat})

# Visualize fit
fig_fit, ax_fit = plt.subplots()
ax_fit.scatter(X, Y, alpha=0.7, label="data")
order = np.argsort(X)
ax_fit.plot(X[order], y_pred[order], linewidth=2, label="fitted line")
ax_fit.set_title("Fitted Line vs Data")
ax_fit.set_xlabel("x")
ax_fit.set_ylabel("y")
ax_fit.legend()
st.pyplot(fig_fit)
plt.close(fig_fit)

st.divider()

# ----------------------------
# (5) CRISP‑DM: Evaluation
# ----------------------------
st.header("5) Evaluation 📏")

mse = float(np.mean(resid ** 2))
mae = float(np.mean(np.abs(resid)))
ss_tot = float(np.sum((Y - np.mean(Y)) ** 2))
ss_res = float(np.sum(resid ** 2))
r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

metrics_df = pd.DataFrame(
    {
        "Metric": ["MSE", "MAE", "R^2"],
        "Value": [mse, mae, r2],
    }
)

st.dataframe(metrics_df, use_container_width=True)

col_eval1, col_eval2 = st.columns(2)
with col_eval1:
    fig_resid, ax_resid = plt.subplots()
    ax_resid.scatter(X, resid, alpha=0.7)
    ax_resid.axhline(0, linewidth=2)
    ax_resid.set_title("Residuals vs x")
    ax_resid.set_xlabel("x")
    ax_resid.set_ylabel("Residual (y − ŷ)")
    st.pyplot(fig_resid)
    plt.close(fig_resid)

with col_eval2:
    fig_hist, ax_hist = plt.subplots()
    ax_hist.hist(resid, bins=30)
    ax_hist.set_title("Residual Distribution")
    ax_hist.set_xlabel("Residual")
    ax_hist.set_ylabel("Count")
    st.pyplot(fig_hist)
    plt.close(fig_hist)

st.markdown(
    """
**Interpretation**:
- **MSE/MAE** closer to 0 ⇒ better fit.
- **$R^2$** closer to 1 ⇒ model explains more variance in y.
- Residuals should look roughly symmetric around 0 with no pattern vs. x.
    """
)

st.divider()

# ----------------------------
# (6) CRISP‑DM: Deployment
# ----------------------------
st.header("6) Deployment 🚀")
st.markdown(
    """
You are already using the **deployed** Streamlit app locally. To share with others, you can:

1. **Package** this file and a `requirements.txt` (containing `streamlit`, `numpy`, `pandas`, `matplotlib`).
2. **Run** on a server or a free hosting (e.g., Streamlit Community Cloud) and share the URL.
3. Or provide a **Flask** API that returns fitted parameters for given data (optional extension).

**Download your dataset** below for reporting or offline analysis.
    """
)

csv_buf = io.StringIO()
df.to_csv(csv_buf, index=False, header=include_header)
st.download_button(
    label="📥 Download generated dataset (CSV)",
    data=csv_buf.getvalue(),
    file_name="synthetic_linear_data.csv",
    mime="text/csv",
)

with st.expander("Appendix: Prompt & Process (for your report)"):
    st.markdown(
        """
**Prompt (assignment brief)**:
> *Write Python to solve a simple linear regression problem following CRISP‑DM steps. Include the prompt and the process (not only code and result). Allow modifying `a` in `a x + b`, noise level, and number of points. Provide a web framework deployment (Streamlit or Flask).* 

**Process summary**:
1. **Define objective**: estimate linear relationship $y \\approx a x + b$.
2. **Design synthetic dataset** with controllable ground truth (a, b), noise σ, sample size n.
3. **Inspect data** (preview table, scatter plot).
4. **Prepare features**: build design matrix with bias term.
5. **Fit model** via least squares (`numpy.linalg.lstsq`).
6. **Evaluate** with MSE, MAE, $R^2$, residual diagnostics.
7. **Deploy** interactive Streamlit app; enable parameter tuning and data download.

You may copy this appendix into your homework report as the "prompt & process" section.
        """
    )

st.success("Done! Adjust parameters in the sidebar to explore different scenarios.")
