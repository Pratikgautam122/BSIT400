"""
dataset_generator.py
-------------------
Defines a framework to programmatically generate diverse mathematical functions
with exact analytical integrals in d-dimensions (d = 1 to 5).
Also extracts mathematical features from these functions to train an ML model.
"""

import numpy as np
from integration_methods import trapezoidal_nd, simpson_nd, romberg_nd, monte_carlo_nd


# ---------------------------------------------------------------------------
# 1. 1D Component Functions with Analytical Integrals
# ---------------------------------------------------------------------------
class Component1D:
    """Base class for 1D mathematical functions with analytical integrals."""
    def eval(self, x):
        raise NotImplementedError
        
    def integral(self, a, b):
        raise NotImplementedError


class PolynomialComponent(Component1D):
    def __init__(self, coeff, p):
        self.coeff = coeff
        self.p = p
        
    def eval(self, x):
        return self.coeff * (x ** self.p)
        
    def integral(self, a, b):
        return self.coeff * (b**(self.p + 1) - a**(self.p + 1)) / (self.p + 1)


class ExponentialComponent(Component1D):
    def __init__(self, coeff, c):
        self.coeff = coeff
        self.c = c
        if abs(self.c) < 1e-5:
            self.c = 1e-5
            
    def eval(self, x):
        return self.coeff * np.exp(self.c * x)
        
    def integral(self, a, b):
        return self.coeff * (np.exp(self.c * b) - np.exp(self.c * a)) / self.c


class TrigComponent(Component1D):
    def __init__(self, coeff, c, is_sin=True):
        self.coeff = coeff
        self.c = c
        self.is_sin = is_sin
        if abs(self.c) < 1e-5:
            self.c = 1e-5
            
    def eval(self, x):
        if self.is_sin:
            return self.coeff * np.sin(self.c * x)
        else:
            return self.coeff * np.cos(self.c * x)
            
    def integral(self, a, b):
        if self.is_sin:
            return self.coeff * (-np.cos(self.c * b) + np.cos(self.c * a)) / self.c
        else:
            return self.coeff * (np.sin(self.c * b) - np.sin(self.c * a)) / self.c


class AbsValueComponent(Component1D):
    """Low-smoothness C0 function (absolute value) with corner at xc."""
    def __init__(self, coeff, xc):
        self.coeff = coeff
        self.xc = xc
        
    def eval(self, x):
        return self.coeff * np.abs(x - self.xc)
        
    def integral(self, a, b):
        # Integral of |x - xc| over [a, b]
        def int_to_point(pt):
            if pt <= self.xc:
                # \int_a^pt (xc - x) dx = [xc*x - x^2/2]_a^pt
                return self.xc * pt - 0.5 * pt**2
            else:
                # \int_a^xc (xc - x) dx + \int_xc^pt (x - xc) dx
                base = self.xc * self.xc - 0.5 * self.xc**2
                term = 0.5 * pt**2 - self.xc * pt
                offset = 0.5 * self.xc**2 - self.xc * self.xc
                return base + term - offset
                
        i_b = int_to_point(b)
        i_a = int_to_point(a)
        return self.coeff * (i_b - i_a)


class StepComponent(Component1D):
    """Discontinuous step function (Heaviside step) at xc."""
    def __init__(self, coeff, xc):
        self.coeff = coeff
        self.xc = xc
        
    def eval(self, x):
        return self.coeff * (x >= self.xc).astype(float)
        
    def integral(self, a, b):
        # Integral of step function over [a, b]
        if b <= self.xc:
            return 0.0
        elif a >= self.xc:
            return self.coeff * (b - a)
        else:
            return self.coeff * (b - self.xc)


class SingularComponent(Component1D):
    """Weak singularity near bounds, represented by (x - a + eps)^(-0.5)."""
    def __init__(self, coeff, domain_min, eps=1e-3):
        self.coeff = coeff
        self.domain_min = domain_min
        self.eps = eps
        
    def eval(self, x):
        # Shift and add epsilon to avoid actual division by zero
        return self.coeff / np.sqrt(x - self.domain_min + self.eps)
        
    def integral(self, a, b):
        # Analytical integral of (x - domain_min + eps)^(-0.5) over [a, b]
        # Anti-derivative is 2 * sqrt(x - domain_min + eps)
        val_b = 2.0 * np.sqrt(b - self.domain_min + self.eps)
        val_a = 2.0 * np.sqrt(a - self.domain_min + self.eps)
        return self.coeff * (val_b - val_a)


# ---------------------------------------------------------------------------
# 2. Multidimensional Function Assembly
# ---------------------------------------------------------------------------
class SyntheticFunction:
    """
    A multidimensional function f: R^d -> R on a hyperrectangle [a, b]^d.
    Supports analytical evaluation and exact analytical integrals.
    """
    def __init__(self, dim, a, b, is_separable=True, terms=None):
        self.dim = dim
        self.a = np.asarray(a, dtype=float)
        self.b = np.asarray(b, dtype=float)
        self.is_separable = is_separable
        self.terms = terms if terms is not None else []
        
    def __call__(self, pts):
        """
        pts: array of shape (N, d).
        Returns y: array of shape (N,).
        """
        pts = np.asarray(pts, dtype=float)
        if pts.ndim == 1:
            pts = pts.reshape(1, -1)
            
        N = pts.shape[0]
        y = np.zeros(N)
        
        for term in self.terms:
            weight = term['weight']
            components = term['components']  # list of Component1D, length = d
            
            if self.is_separable:
                # Term is weight * \prod_j components[j](pts[:, j])
                term_val = np.ones(N)
                for j in range(self.dim):
                    term_val *= components[j].eval(pts[:, j])
                y += weight * term_val
            else:
                # Term is weight * \sum_j components[j](pts[:, j])
                term_val = np.zeros(N)
                for j in range(self.dim):
                    term_val += components[j].eval(pts[:, j])
                y += weight * term_val
                
        return y

    def get_exact_integral(self):
        """Returns the exact analytical integral over the domain."""
        total_integral = 0.0
        
        for term in self.terms:
            weight = term['weight']
            components = term['components']
            
            if self.is_separable:
                # Integral of product is product of integrals
                term_integral = 1.0
                for j in range(self.dim):
                    term_integral *= components[j].integral(self.a[j], self.b[j])
                total_integral += weight * term_integral
            else:
                # Integral of sum is sum of integrals scaled by volume of other dimensions
                term_integral = 0.0
                for j in range(self.dim):
                    integ_j = components[j].integral(self.a[j], self.b[j])
                    # product of lengths of all other dimensions
                    other_vol = 1.0
                    for k in range(self.dim):
                        if k != j:
                            other_vol *= (self.b[k] - self.a[k])
                    term_integral += integ_j * other_vol
                total_integral += weight * term_integral
                
        return total_integral


# ---------------------------------------------------------------------------
# 3. Random Function Factory
# ---------------------------------------------------------------------------
def generate_random_function(dim, seed=None):
    """
    Creates a random SyntheticFunction of a given dimension with random bounds
    and mixtures of polynomials, exponentials, trigonometric, and low-smoothness/step terms.
    """
    rng = np.random.default_rng(seed)
    
    # 1. Random bounds [a_j, b_j]
    a = []
    b = []
    for _ in range(dim):
        domain_width = rng.uniform(0.5, 3.0)
        domain_start = rng.uniform(-1.0, 1.0)
        a.append(domain_start)
        b.append(domain_start + domain_width)
    a = np.array(a)
    b = np.array(b)
    
    # 2. Decide if separable or additive
    is_separable = rng.choice([True, False])
    
    # 3. Create random terms
    n_terms = rng.integers(1, 3)
    terms = []
    
    for _ in range(n_terms):
        weight = rng.uniform(-1.5, 1.5)
        components = []
        
        for j in range(dim):
            # Select random component type
            # 0: Poly, 1: Exp, 2: Trig, 3: Abs, 4: Step, 5: Singular
            comp_type = rng.choice([0, 1, 2, 3, 4, 5], p=[0.3, 0.2, 0.2, 0.15, 0.1, 0.05])
            coeff = rng.uniform(0.5, 1.5)
            
            if comp_type == 0:
                p = int(rng.integers(0, 5))
                comp = PolynomialComponent(coeff, p)
            elif comp_type == 1:
                c = rng.uniform(-1.0, 1.0)
                comp = ExponentialComponent(coeff, c)
            elif comp_type == 2:
                c = rng.uniform(1.0, 5.0)
                is_sin = rng.choice([True, False])
                comp = TrigComponent(coeff, c, is_sin)
            elif comp_type == 3:
                xc = rng.uniform(a[j], b[j])
                comp = AbsValueComponent(coeff, xc)
            elif comp_type == 4:
                xc = rng.uniform(a[j], b[j])
                comp = StepComponent(coeff, xc)
            else:
                comp = SingularComponent(coeff, a[j], eps=1e-3)
                
            components.append(comp)
            
        terms.append({
            'weight': weight,
            'components': components
        })
        
    return SyntheticFunction(dim, a, b, is_separable=is_separable, terms=terms)


# ---------------------------------------------------------------------------
# 4. Feature Extractor
# ---------------------------------------------------------------------------
def extract_features(f):
    """
    Numerically extracts mathematical features from a black-box SyntheticFunction.
    Does NOT inspect the analytical formulation.
    """
    dim = f.dim
    a, b = f.a, f.b
    volume = np.prod(b - a)
    
    # 1. Sample 500 points in domain to compute general statistics
    rng = np.random.default_rng(42)
    P_stat = rng.uniform(a, b, size=(500, dim))
    y = f(P_stat)
    
    mean_y = np.mean(y)
    std_y = np.std(y)
    min_y = np.min(y)
    max_y = np.max(y)
    range_y = max_y - min_y
    std_range_ratio = std_y / (range_y + 1e-15)
    
    min_abs_y = np.min(np.abs(y))
    max_abs_y = np.max(np.abs(y))
    
    # 2. Estimate derivatives via finite differences in coordinate directions
    # Sample 100 points in the interior
    P_diff = rng.uniform(a + 0.05 * (b - a), b - 0.05 * (b - a), size=(100, dim))
    
    abs_df = []
    abs_d2f = []
    
    for j in range(dim):
        h = 1e-4 * (b[j] - a[j])
        
        # Shifted points
        P_plus = P_diff.copy()
        P_plus[:, j] += h
        P_minus = P_diff.copy()
        P_minus[:, j] -= h
        
        y_center = f(P_diff)
        y_plus = f(P_plus)
        y_minus = f(P_minus)
        
        # Central differences
        df_j = (y_plus - y_minus) / (2.0 * h)
        d2f_j = (y_plus - 2.0 * y_center + y_minus) / (h ** 2)
        
        abs_df.extend(np.abs(df_j))
        abs_d2f.extend(np.abs(d2f_j))
        
    mean_abs_df = float(np.mean(abs_df))
    max_abs_df = float(np.max(abs_df))
    mean_abs_d2f = float(np.mean(abs_d2f))
    max_abs_d2f = float(np.max(abs_d2f))
    
    d2f_to_df_ratio = mean_abs_d2f / (mean_abs_df + 1e-15)
    max_to_mean_df = max_abs_df / (mean_abs_df + 1e-15)
    max_to_mean_d2f = max_abs_d2f / (mean_abs_d2f + 1e-15)
    
    # 3. Oscillation/Extrema count along a 1D diagonal slice
    # Sample 100 points linearly from lower bounds to upper bounds
    t = np.linspace(0.0, 1.0, 100).reshape(-1, 1)
    diagonal_pts = a + t * (b - a)  # (100, dim)
    y_diag = f(diagonal_pts)
    dy_diag = np.diff(y_diag)
    # Count sign changes in dy_diag
    sign_changes = np.diff(np.sign(dy_diag))
    extrema_count = int(np.sum(sign_changes != 0))
    
    # 4. Boundary periodic similarity (for 1D, or average over dimensions)
    # Checks if values at bounds are equal
    boundary_diffs = []
    for j in range(dim):
        pt_a = 0.5 * (a + b)
        pt_a[j] = a[j]
        pt_b = 0.5 * (a + b)
        pt_b[j] = b[j]
        diff = abs(f(pt_a.reshape(1, -1))[0] - f(pt_b.reshape(1, -1))[0])
        boundary_diffs.append(diff)
    mean_boundary_diff = float(np.mean(boundary_diffs))
    
    # Detect singularities (unusually large values or derivative spikes)
    has_singularity = 1.0 if (max_abs_y > 1e4 or max_abs_df > 1e6) else 0.0

    features = {
        'dim': float(dim),
        'volume': float(volume),
        'mean_y': float(mean_y),
        'std_y': float(std_y),
        'range_y': float(range_y),
        'std_range_ratio': float(std_range_ratio),
        'min_abs_y': float(min_abs_y),
        'max_abs_y': float(max_abs_y),
        'mean_abs_df': mean_abs_df,
        'max_abs_df': max_abs_df,
        'mean_abs_d2f': mean_abs_d2f,
        'max_abs_d2f': max_abs_d2f,
        'd2f_to_df_ratio': d2f_to_df_ratio,
        'max_to_mean_df': max_to_mean_df,
        'max_to_mean_d2f': max_to_mean_d2f,
        'extrema_count': float(extrema_count),
        'mean_boundary_diff': mean_boundary_diff,
        'has_singularity': has_singularity
    }
    return features


# ---------------------------------------------------------------------------
# 5. Dataset Generation & Labeling
# ---------------------------------------------------------------------------
def generate_dataset(n_samples=500, target_evals=1024, seed=42):
    """
    Generates a dataset of n_samples mathematical functions, evaluates
    Trapezoidal, Simpson, Romberg, and Monte Carlo on each, and labels them
    with the method that achieves the lowest integration error.
    
    Returns:
      dataset_features: list of dicts
      labels: list of integers
      methods_info: list of names
    """
    print(f"Generating synthetic dataset of {n_samples} functions...")
    rng = np.random.default_rng(seed)
    
    dataset_features = []
    labels = []
    
    # Method index mapping
    methods = {
        0: 'trapezoidal',
        1: 'simpson',
        2: 'romberg',
        3: 'monte_carlo'
    }
    
    success_count = 0
    failures = 0
    
    for i in range(n_samples):
        # Cycle through dimensions 1 to 5
        dim = int(i % 5) + 1
        
        # Generate random function
        f = generate_random_function(dim, seed=int(rng.integers(1, 1000000)))
        
        try:
            I_exact = f.get_exact_integral()
            
            # If the exact integral is very close to 0, or blows up, skip to avoid division/numerical issues
            if not np.isfinite(I_exact) or abs(I_exact) > 1e6 or abs(I_exact) < 1e-6:
                continue
                
            # Run integrations
            # Trapezoidal
            I_trap, _ = trapezoidal_nd(f, f.a, f.b, target_evals=target_evals)
            # Simpson
            I_simp, n_simp = simpson_nd(f, f.a, f.b, target_evals=target_evals)
            # Romberg
            I_rom, _ = romberg_nd(f, f.a, f.b, target_evals=target_evals)
            # Monte Carlo - match Simpson's evaluation budget exactly for fairness
            I_mc, _ = monte_carlo_nd(f, f.a, f.b, n_evals=n_simp, seed=42)
            
            # Compute absolute errors
            err_trap = abs(I_trap - I_exact)
            err_simp = abs(I_simp - I_exact)
            err_rom = abs(I_rom - I_exact)
            err_mc = abs(I_mc - I_exact)
            
            errors = [err_trap, err_simp, err_rom, err_mc]
            
            # Find minimum error index
            best_idx = int(np.argmin(errors))
            
            # Check for non-finite values in errors
            if not all(np.isfinite(errors)):
                continue
                
            # Extract numerical features
            feats = extract_features(f)
            
            # Check for non-finite values in features
            if not all(np.isfinite(list(feats.values()))):
                continue
                
            dataset_features.append(feats)
            labels.append(best_idx)
            success_count += 1
            
            if success_count % 100 == 0:
                print(f"  Processed {success_count} functions successfully...")
                
        except Exception as e:
            failures += 1
            # Some randomly generated bounds/singularities might raise OverflowError etc., just skip
            continue
            
    print(f"Dataset generated successfully. Total samples: {success_count} (skipped {failures} failed integrations).\n")
    return dataset_features, labels, list(methods.values())


if __name__ == "__main__":
    # Test generation of a few functions and print features
    feats, labels, methods = generate_dataset(n_samples=5, target_evals=1024)
    print("Methods mapping:", methods)
    print("First function features:")
    for k, v in feats[0].items():
        print(f"  {k:20s}: {v}")
    print("First function label:", labels[0], f"({methods[labels[0]]})")
