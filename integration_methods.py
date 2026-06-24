"""
integration_methods.py
----------------------
Implements multidimensional integration methods from scratch using NumPy.
Each method is designed to work for a general function f: R^d -> R
over a hyperrectangular domain [a, b]^d, using a specified budget
of function evaluations.

Methods:
  1. Trapezoidal Rule (d-dimensional)
  2. Simpson's Rule (d-dimensional)
  3. Romberg Integration (d-dimensional)
  4. Monte Carlo Integration (d-dimensional)
"""

import numpy as np


def get_grid_points_and_weights_1d_trap(a, b, m):
    """
    Returns 1D trapezoidal grid points and weights for interval [a, b] with m points.
    """
    if m < 2:
        m = 2
    x = np.linspace(a, b, m)
    h = (b - a) / (m - 1)
    w = np.full(m, h)
    w[0] = 0.5 * h
    w[-1] = 0.5 * h
    return x, w


def get_grid_points_and_weights_1d_simp(a, b, m):
    """
    Returns 1D Simpson's rule grid points and weights for interval [a, b] with m points.
    Note: m must be an odd number (number of intervals n = m - 1 must be even).
    """
    if m < 3:
        m = 3
    if m % 2 == 0:
        m += 1  # force odd
    x = np.linspace(a, b, m)
    h = (b - a) / (m - 1)
    w = np.empty(m)
    w[0] = 1.0
    w[-1] = 1.0
    # alternating 4 and 2 weights
    w[1:-1:2] = 4.0
    w[2:-1:2] = 2.0
    w = w * (h / 3.0)
    return x, w


def evaluate_on_grid(f, xs_1d, ws_1d):
    """
    Evaluates f on a d-dimensional grid formed by tensor product of 1D grids.
    xs_1d: list of 1D coordinate arrays, len = d
    ws_1d: list of 1D weight arrays, len = d
    Returns:
      approx_integral: scalar
      n_evals: integer (number of grid points)
    """
    d = len(xs_1d)
    
    # Generate d-dimensional grid
    # np.meshgrid with indexing='ij' returns grids suitable for matrix indices
    grids = np.meshgrid(*xs_1d, indexing='ij')
    # Reshape grid points to shape (N, d) for vectorized evaluation
    flat_grids = [g.ravel() for g in grids]
    points = np.stack(flat_grids, axis=-1)  # (N, d)
    
    # Evaluate function
    y = f(points)
    n_evals = len(y)
    
    # Generate weight grid (tensor product of weights)
    w_grids = np.meshgrid(*ws_1d, indexing='ij')
    flat_w_grids = [w.ravel() for w in w_grids]
    weights = np.prod(np.stack(flat_w_grids, axis=-1), axis=-1)  # (N,)
    
    approx_integral = float(np.sum(y * weights))
    return approx_integral, n_evals


# ---------------------------------------------------------------------------
# 1. d-dimensional Trapezoidal Rule
# ---------------------------------------------------------------------------
def trapezoidal_nd(f, a, b, target_evals=1024):
    """
    Integrates f over [a, b] using d-dimensional Trapezoidal rule.
    a, b: lists or arrays of length d, defining bounds.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    d = len(a)
    
    # Choose m points per axis such that m^d is close to target_evals
    m = int(round(target_evals ** (1.0 / d)))
    if m < 2:
        m = 2
        
    xs_1d = []
    ws_1d = []
    for j in range(d):
        x, w = get_grid_points_and_weights_1d_trap(a[j], b[j], m)
        xs_1d.append(x)
        ws_1d.append(w)
        
    return evaluate_on_grid(f, xs_1d, ws_1d)


# ---------------------------------------------------------------------------
# 2. d-dimensional Simpson's Rule
# ---------------------------------------------------------------------------
def simpson_nd(f, a, b, target_evals=1024):
    """
    Integrates f over [a, b] using d-dimensional Simpson's 1/3 rule.
    a, b: lists or arrays of length d, defining bounds.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    d = len(a)
    
    # Choose m (odd) points per axis such that m^d is close to target_evals
    m = int(round(target_evals ** (1.0 / d)))
    if m % 2 == 0:
        # Check whether m-1 or m+1 is better
        m_lower = max(3, m - 1)
        m_upper = m + 1
        if abs(m_lower**d - target_evals) < abs(m_upper**d - target_evals):
            m = m_lower
        else:
            m = m_upper
    if m < 3:
        m = 3
        
    xs_1d = []
    ws_1d = []
    for j in range(d):
        x, w = get_grid_points_and_weights_1d_simp(a[j], b[j], m)
        xs_1d.append(x)
        ws_1d.append(w)
        
    return evaluate_on_grid(f, xs_1d, ws_1d)


# ---------------------------------------------------------------------------
# 3. d-dimensional Romberg Integration
# ---------------------------------------------------------------------------
def romberg_nd(f, a, b, target_evals=1024):
    """
    Integrates f over [a, b] using d-dimensional Romberg Integration.
    a, b: lists or arrays of length d, defining bounds.
    Extrapolates multidimensional Trapezoidal rule approximations.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    d = len(a)
    
    # Determine the maximum K level such that (2^K + 1)^d <= target_evals * 1.5
    # We want the final evaluation count to be near the budget.
    K = 0
    while True:
        m_next = 2**(K + 1) + 1
        if m_next**d > target_evals * 1.5 and K > 0:
            break
        K += 1
        if m_next**d > target_evals * 5.0:  # Safety cap
            break
            
    # Build the first column of Romberg table (Trapezoidal approximations)
    R = np.zeros((K + 1, K + 1))
    total_evals = 0
    
    for k in range(K + 1):
        m = 2**k + 1
        xs_1d = []
        ws_1d = []
        for j in range(d):
            x, w = get_grid_points_and_weights_1d_trap(a[j], b[j], m)
            xs_1d.append(x)
            ws_1d.append(w)
        val, n_eval = evaluate_on_grid(f, xs_1d, ws_1d)
        R[k, 0] = val
        if k == K:
            total_evals = n_eval  # Report the final evaluation grid size
            
    # Fill in the rest of the Romberg table using extrapolation
    for j in range(1, K + 1):
        for k in range(j, K + 1):
            factor = 4.0**j
            R[k, j] = (factor * R[k, j-1] - R[k-1, j-1]) / (factor - 1.0)
            
    return float(R[K, K]), total_evals


# ---------------------------------------------------------------------------
# 4. d-dimensional Monte Carlo Integration
# ---------------------------------------------------------------------------
def monte_carlo_nd(f, a, b, n_evals=1024, seed=42):
    """
    Integrates f over [a, b] using d-dimensional Monte Carlo integration.
    a, b: lists or arrays of length d, defining bounds.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    d = len(a)
    
    # Calculate domain volume
    volume = np.prod(b - a)
    
    # Draw uniform samples
    rng = np.random.default_rng(seed)
    points = rng.uniform(a, b, size=(n_evals, d))
    
    # Evaluate and compute mean
    y = f(points)
    integral = float(volume * np.mean(y))
    return integral, n_evals


if __name__ == "__main__":
    # Quick test of 2D integration of f(x,y) = x^2 * y
    # Domain: x in [0, 1], y in [0, 2]
    # Exact integral: \int_0^1 x^2 dx * \int_0^2 y dy = [x^3/3]_0^1 * [y^2/2]_0^2 = 1/3 * 2 = 2/3 ≈ 0.6666667
    
    def test_f(pts):
        return pts[:, 0]**2 * pts[:, 1]
        
    a = [0.0, 0.0]
    b = [1.0, 2.0]
    
    print("Exact: 0.6666667")
    print("Trap nd:        ", trapezoidal_nd(test_f, a, b, target_evals=1024))
    print("Simp nd:        ", simpson_nd(test_f, a, b, target_evals=1024))
    print("Romb nd:        ", romberg_nd(test_f, a, b, target_evals=1024))
    print("Monte Carlo nd: ", monte_carlo_nd(test_f, a, b, n_evals=1024))
