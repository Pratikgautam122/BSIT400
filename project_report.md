# Machine Learning Model to Select Best Numerical Integration Method
**A Feature-Based Classification Approach to Solver Selection in Multi-Dimensional Calculus**

---

## 1. Abstract
Selecting the optimal numerical integration method for a given mathematical function is a classical challenge in numerical analysis. The choice depends heavily on function characteristics such as dimension, smoothness, boundary conditions, and frequency of oscillations. 

This project implements a complete pipeline that programmatically generates a diverse dataset of 600 multidimensional functions (with dimensions $d \in [1, 5]$) whose exact integrals are computed analytically. We extract 18 numerical features (including dimension, volume, finite-difference approximation of derivatives, spikiness, boundary properties, and oscillation counts) to characterize each function. Four numerical integration methods—**Trapezoidal Rule**, **Simpson's Rule**, **Romberg Integration**, and **Monte Carlo Integration**—are executed on each function using a fixed evaluation budget of approximately $1024$ points. 

Using the method with the lowest absolute error as the label, we train a **Decision Tree Classifier** and a **Random Forest Classifier** to choose the optimal integration method. The Random Forest model achieves a test set accuracy of **55.33%** (more than double the 25% random baseline). Feature importance analysis indicates that the domain dimension ($d$) and derivative spikes ($\text{max-to-mean}$ derivative ratio) are the primary drivers of solver selection, aligning perfectly with mathematical integration theory.

---

## 2. Introduction
Numerical integration is the core engine for physical simulations, bayesian inference, and financial modeling. Standard numerical integration methods make different assumptions about the underlying function:
1. **Trapezoidal Rule:** Assumes local linear behavior.
2. **Simpson's Rule:** Fits local quadratic polynomials.
3. **Romberg Integration:** Assumes high-order differentiability to perform Richardson extrapolation.
4. **Monte Carlo Integration:** Relies on statistical sampling, requiring only integrability.

For any specific function, picking the wrong solver can result in either slow convergence (requiring millions of evaluations) or catastrophic loss of accuracy (if singularities or discontinuities are present). Because checking mathematical properties analytically is impossible for black-box simulators, we train a machine learning model to evaluate function characteristics and recommend the best solver.

---

## 3. Mathematical Formulations of the Solvers

All solvers are implemented from scratch in [integration_methods.py](file:///Users/pratikgautam/Documents/bsit400/integration_methods.py) for arbitrary dimension $d$ over hyperrectangular domains $[a, b]^d$.

### 3.1 Multi-Dimensional Trapezoidal Rule
Let $m$ be the number of grid points along each coordinate axis. The grid spacing for axis $j$ is $h_j = (b_j - a_j)/(m - 1)$. The 1D weight vector is:
$$w_j = \left[ \frac{h_j}{2}, h_j, h_j, \dots, h_j, \frac{h_j}{2} \right]$$
The $d$-dimensional integration weight at index $(i_1, \dots, i_d)$ is the tensor product of 1D weights:
$$W(i_1, \dots, i_d) = \prod_{j=1}^d w_{j, i_j}$$
The integral is approximated by evaluating $f(x)$ on the grid:
$$I_{\text{trap}} = \sum_{i_1=0}^{m-1} \dots \sum_{i_d=0}^{m-1} W(i_1, \dots, i_d) f(x_{1, i_1}, \dots, x_{d, i_d})$$

### 3.2 Multi-Dimensional Simpson's Rule
Requires the number of subdivisions $n = m-1$ to be even (making $m$ odd). The 1D weight vector is:
$$w_j = \frac{h_j}{3} \left[ 1, 4, 2, 4, \dots, 2, 4, 1 \right]$$
The multidimensional weight $W(i_1, \dots, i_d)$ is the tensor product of these 1D weights, evaluated on a regular grid of size $m^d$.

### 3.3 Multi-Dimensional Romberg Integration
Romberg integration generalizes to $d$ dimensions by extrapolating $d$-dimensional Trapezoidal approximations. At step $k$, we compute the $d$-dimensional Trapezoidal approximation $T_k$ using a grid of size $m_k = 2^k + 1$ along each axis.
We initialize the Romberg table:
$$R_{k, 0} = T_k \quad \text{for } k = 0, \dots, K$$
Then we perform extrapolation:
$$R_{k, j} = \frac{4^j R_{k, j-1} - R_{k-1, j-1}}{4^j - 1} \quad \text{for } 1 \le j \le k \le K$$
The final extrapolated value is $R_{K, K}$. Richardson extrapolation removes the $\mathcal{O}(h^2), \mathcal{O}(h^4), \dots, \mathcal{O}(h^{2j})$ error terms.

### 3.4 Multi-Dimensional Monte Carlo Integration
Given $N$ points sampled uniformly in the domain $[a, b]^d$:
$$x_i \sim \mathcal{U}(a, b), \quad i = 1, \dots, N$$
The integral approximation is:
$$I_{\text{MC}} = V \cdot \frac{1}{N} \sum_{i=1}^N f(x_i)$$
where $V = \prod_{j=1}^d (b_j - a_j)$ is the hyper-volume of the integration domain.

---

## 4. Feature Extraction Framework

To classify functions without inspecting their analytical definitions, we extract **18 numerical features** by evaluating each function at a small set of sample points. The feature extractor is implemented in [dataset_generator.py](file:///Users/pratikgautam/Documents/bsit400/dataset_generator.py#L182-L270):

1. **Dimension (`dim`):** Integer $d \in [1, 5]$.
2. **Domain Volume (`volume`):** $\prod (b_j - a_j)$.
3. **Function Statistics:** Mean (`mean_y`), standard deviation (`std_y`), range (`range_y`), min/max absolute values (`min_abs_y`/`max_abs_y`), and ruggedness ratio (`std_range_ratio` $= \text{std\_y}/\text{range\_y}$).
4. **Smoothness/Derivative Estimation:** We sample points in the interior and compute first ($D_1$) and second ($D_2$) central differences along coordinate axes:
   - `mean_abs_df` / `max_abs_df`: Mean and max of absolute first derivatives.
   - `mean_abs_d2f` / `max_abs_d2f`: Mean and max of absolute second derivatives.
   - `d2f_to_df_ratio`: Ratio of second to first derivative magnitude.
   - `max_to_mean_df` / `max_to_mean_d2f`: Ratio of max derivative to mean derivative. A high ratio indicates sharp corners or step discontinuities.
5. **Oscillation Count (`extrema_count`):** Number of sign changes in the first derivative along a 1D diagonal slice of the domain.
6. **Boundary Periodicity (`mean_boundary_diff`):** Average $|f(a_j) - f(b_j)|$ across all dimensions.
7. **Singularity Flag (`has_singularity`):** Set to 1 if maximum values or derivatives exceed extreme thresholds ($10^4$ and $10^6$ respectively).

---

## 5. Synthetic Data Generation

To build a reliable training dataset, we implement a **synthetic function generator** in [dataset_generator.py](file:///Users/pratikgautam/Documents/bsit400/dataset_generator.py#L106-L180). To obtain exact machine-precision integrals in any dimension $d \in [1, 5]$, we construct functions as combinations of:
* **Separable terms:** $f(x) = \prod_{j=1}^d \phi_j(x_j)$ with exact integral $I = \prod_{j=1}^d \int_{a_j}^{b_j} \phi_j(x_j) dx_j$.
* **Additive terms:** $f(x) = \sum_{j=1}^d \phi_j(x_j)$ with exact integral $I = \sum_{j=1}^d \left( \int_{a_j}^{b_j} \phi_j(x_j) dx_j \times \prod_{k \neq j} (b_k - a_k) \right)$.

The 1D component functions $\phi_j(x)$ are randomly selected from:
* Polynomials ($x^p$, $p \le 4$)
* Exponentials ($e^{cx}$)
* Trigonometric ($\sin(cx)$, $\cos(cx)$)
* Absolute values ($|x - x_c|$, low-smoothness $C^0$)
* Steps ($\Theta(x - x_c)$, discontinuous step functions)
* Weak singularities ($(x - a + \epsilon)^{-0.5}$)

We generate **600 distinct functions** distributed evenly across dimensions 1 to 5. For each function, we run all 4 integration methods with a fixed evaluation budget of $N \approx 1024$ points. The solver yielding the lowest absolute error is assigned as the class label:
* **Class 0 (Trapezoidal):** 14.3%
* **Class 1 (Simpson):** 22.2%
* **Class 2 (Romberg):** 32.5%
* **Class 3 (Monte Carlo):** 31.0%

---

## 6. Machine Learning Model Training and Evaluation

The model training and validation pipeline is implemented in [train_model.py](file:///Users/pratikgautam/Documents/bsit400/train_model.py). We split the dataset into 75% training and 25% testing subsets (stratified by class). We compare a single **Decision Tree Classifier** and a **Random Forest Classifier**.

### 6.1 Classification Reports

#### Decision Tree (Max Depth = 4)
* **Test Accuracy:** **47.33%**
* **Detailed Metrics:**
  - *Trapezoidal:* Precision: 0.23, Recall: 0.14, F1-Score: 0.17
  - *Simpson:* Precision: 0.26, Recall: 0.30, F1-Score: 0.28
  - *Romberg:* Precision: 0.57, Recall: 0.57, F1-Score: 0.57
  - *Monte Carlo:* Precision: 0.61, Recall: 0.65, F1-Score: 0.63

#### Random Forest (100 Trees, Max Depth = 6)
* **Test Accuracy:** **55.33%**
* **Detailed Metrics:**
  - *Trapezoidal:* Precision: 0.44, Recall: 0.18, F1-Score: 0.26
  - *Simpson:* Precision: 0.37, Recall: 0.21, F1-Score: 0.27
  - *Romberg:* Precision: 0.60, Recall: 0.65, F1-Score: 0.63
  - *Monte Carlo:* Precision: 0.58, Recall: 0.87, F1-Score: 0.70

---

## 7. Analysis and Visualizations

All figures have been saved to the [figures/](file:///Users/pratikgautam/Documents/bsit400/figures) directory during execution.

### 7.1 Feature Importance Analysis
The feature importances plot ([feature_importances.png](file:///Users/pratikgautam/Documents/bsit400/figures/feature_importances.png)) highlights the most critical factors driving the classification decision:
* **`max_to_mean_d2f` (13.1%):** Dictates smoothness. A high ratio indicates localized derivative spikes (discontinuities or singularities), which breaks high-order polynomial fits like Romberg.
* **`dim` (12.4%):** Specifies dimension $d$. In high dimensions, grid methods fail, making Monte Carlo the optimal choice.
* **`std_range_ratio` (7.8%):** Captures ruggedness and how "spiky" the function behaves.
* **`max_to_mean_df` (7.4%):** Indicates first-order derivative discontinuities.

### 7.2 Decision Rules and Boundaries
The decision boundary map ([decision_boundary.png](file:///Users/pratikgautam/Documents/bsit400/figures/decision_boundary.png)) plots the decision regimes in the 2D space of **Dimension ($d$)** and **Ruggedness Ratio**:
* **Low Dimensions ($d \le 3$):** Romberg dominates for smooth functions (low ruggedness/flat regions). Simpson wins for moderate oscillations and lower smoothness.
* **High Dimensions ($d \ge 4$):** Monte Carlo dominates the parameter space, especially for rugged or step-like functions.
* **Discontinuities ($max\_to\_mean\_d2f > 22.28$):** In low dimensions, if a function has localized sharp corners or steps, Romberg is rejected, and Simpson or Monte Carlo is recommended due to their robustness.

The confusion matrix ([confusion_matrix.png](file:///Users/pratikgautam/Documents/bsit400/figures/confusion_matrix.png)) shows that the model is highly accurate at identifying **Monte Carlo** (87% recall) and **Romberg** (65% recall), but occasionally confuses Trapezoidal and Simpson's rules due to overlapping intermediate smoothness levels.

---

## 8. Conclusion
This project successfully trains an ML classifier to choose the optimal numerical integration method. By extracting black-box numerical features, we demonstrate that a **Random Forest Classifier** can predict the best solver with **55.33% accuracy** on unseen mathematical functions, compared to a random baseline of 25%. 

The model's decision rules align perfectly with mathematical principles: choosing Monte Carlo for high dimensions and rejecting Romberg when derivative spikes reveal step-discontinuities or singularities. This framework can be integrated directly into numerical packages as a pre-solver check to dynamically optimize integration performance.
