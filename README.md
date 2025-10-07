**Goal**: Given observed data pairs (x, y), we want to model the relationship
between x (independent variable) and y (dependent variable) with a simple linear
model: $y \\approx a\\,x + b$. 

**Why**: Linear regression is widely used to quantify trends, forecast outcomes,
and explain how changes in x relate to changes in y.

**Success criteria**: Low prediction error (e.g., small MSE/MAE), sensible
parameters $(a, b)$ that are close to the true underlying process if known, and
residuals with no obvious patterns (indicating model adequacy).
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
