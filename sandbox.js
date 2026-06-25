// sandbox.js

// 1. Seeded Random Number Generator for reproducible features
class SeededRNG {
    constructor(seed = 42) {
        this.seed = seed;
    }
    next() {
        // LCG algorithm
        this.seed = (this.seed * 1664525 + 1013904223) % 4294967296;
        return this.seed / 4294967296;
    }
    uniform(min, max) {
        return min + this.next() * (max - min);
    }
}

// 2. 1D Component Classes
class PolynomialComponent {
    constructor(coeff, p) {
        this.coeff = coeff;
        this.p = p;
    }
    eval(x) {
        return this.coeff * Math.pow(x, this.p);
    }
    toString() {
        return `${this.coeff.toFixed(1)} * x^${this.p}`;
    }
}

class ExponentialComponent {
    constructor(coeff, c) {
        this.coeff = coeff;
        this.c = c;
    }
    eval(x) {
        return this.coeff * Math.exp(this.c * x);
    }
    toString() {
        return `${this.coeff.toFixed(1)} * e^{${this.c.toFixed(1)} * x}`;
    }
}

class TrigComponent {
    constructor(coeff, c) {
        this.coeff = coeff;
        this.c = c;
    }
    eval(x) {
        return this.coeff * Math.sin(this.c * x);
    }
    toString() {
        return `${this.coeff.toFixed(1)} * sin(${this.c.toFixed(2)} * x)`;
    }
}

class AbsValueComponent {
    constructor(coeff, xc) {
        this.coeff = coeff;
        this.xc = xc;
    }
    eval(x) {
        return this.coeff * Math.abs(x - this.xc);
    }
    toString() {
        return `${this.coeff.toFixed(1)} * |x - ${this.xc.toFixed(1)}|`;
    }
}

class StepComponent {
    constructor(coeff, xc) {
        this.coeff = coeff;
        this.xc = xc;
    }
    eval(x) {
        return this.coeff * (x >= this.xc ? 1.0 : 0.0);
    }
    toString() {
        return `${this.coeff.toFixed(1)} * step(x - ${this.xc.toFixed(1)})`;
    }
}

class SingularComponent {
    constructor(coeff, domainMin, eps = 1e-3) {
        this.coeff = coeff;
        this.domainMin = domainMin;
        this.eps = eps;
    }
    eval(x) {
        return this.coeff / Math.sqrt(x - this.domainMin + this.eps);
    }
    toString() {
        return `${this.coeff.toFixed(1)} / sqrt(x - a + ${this.eps})`;
    }
}

// 3. Synthetic Multidimensional Function
class JSFunction {
    constructor(dim, a, b, isSeparable, comp1, comp2) {
        this.dim = dim;
        this.a = a;
        this.b = b;
        this.isSeparable = isSeparable;
        this.comp1 = comp1;
        this.comp2 = comp2;
    }

    eval(coords) {
        // coords is an array of size this.dim
        if (this.dim === 1) {
            return this.comp1.eval(coords[0]);
        }

        if (this.isSeparable) {
            let val = this.comp1.eval(coords[0]);
            for (let j = 1; j < this.dim; j++) {
                val *= this.comp2.eval(coords[j]);
            }
            return val;
        } else {
            let val = this.comp1.eval(coords[0]);
            for (let j = 1; j < this.dim; j++) {
                val += this.comp2.eval(coords[j]);
            }
            return val;
        }
    }

    getFormulaString() {
        if (this.dim === 1) {
            return `f(x_1) = ${this.comp1.toString()}`;
        }
        
        let elements = [`${this.comp1.toString()} (on x_1)`];
        for (let j = 2; j <= this.dim; j++) {
            elements.push(`${this.comp2.toString()} (on x_${j})`);
        }
        
        const operator = this.isSeparable ? " * " : " + ";
        return `f(x) = ${elements.join(operator)}`;
    }
}

// 4. Feature Extractor in JS
function extractFeaturesJS(f) {
    const rng = new SeededRNG(42);
    
    // 1. Sample 500 points for general stats
    let y_vals = [];
    for (let i = 0; i < 500; i++) {
        let coords = [];
        for (let j = 0; j < f.dim; j++) {
            coords.push(rng.uniform(f.a, f.b));
        }
        y_vals.push(f.eval(coords));
    }
    
    const sum = y_vals.reduce((s, x) => s + x, 0);
    const mean_y = sum / 500;
    
    const variance = y_vals.reduce((s, x) => s + Math.pow(x - mean_y, 2), 0) / 500;
    const std_y = Math.sqrt(variance);
    
    const min_y = Math.min(...y_vals);
    const max_y = Math.max(...y_vals);
    const range_y = max_y - min_y;
    const std_range_ratio = std_y / (range_y + 1e-15);
    
    const abs_y_vals = y_vals.map(Math.abs);
    const min_abs_y = Math.min(...abs_y_vals);
    const max_abs_y = Math.max(...abs_y_vals);
    
    // 2. Estimate derivatives (central differences, 100 interior points)
    let abs_df = [];
    let abs_d2f = [];
    
    for (let i = 0; i < 100; i++) {
        let coords = [];
        for (let j = 0; j < f.dim; j++) {
            const offset = 0.05 * (f.b - f.a);
            coords.push(rng.uniform(f.a + offset, f.b - offset));
        }
        
        for (let j = 0; j < f.dim; j++) {
            const h = 1e-4 * (f.b - f.a);
            
            let coords_plus = [...coords];
            coords_plus[j] += h;
            
            let coords_minus = [...coords];
            coords_minus[j] -= h;
            
            const y_center = f.eval(coords);
            const y_plus = f.eval(coords_plus);
            const y_minus = f.eval(coords_minus);
            
            const df_j = (y_plus - y_minus) / (2.0 * h);
            const d2f_j = (y_plus - 2.0 * y_center + y_minus) / (h * h);
            
            abs_df.push(Math.abs(df_j));
            abs_d2f.push(Math.abs(d2f_j));
        }
    }
    
    const mean_abs_df = abs_df.reduce((s, x) => s + x, 0) / abs_df.length;
    const max_abs_df = Math.max(...abs_df);
    const mean_abs_d2f = abs_d2f.reduce((s, x) => s + x, 0) / abs_d2f.length;
    const max_abs_d2f = Math.max(...abs_d2f);
    
    const d2f_to_df_ratio = mean_abs_d2f / (mean_abs_df + 1e-15);
    const max_to_mean_df = max_abs_df / (mean_abs_df + 1e-15);
    const max_to_mean_d2f = max_abs_d2f / (mean_abs_d2f + 1e-15);
    
    // 3. Diagonal slice extrema count
    let y_diag = [];
    for (let i = 0; i < 100; i++) {
        const t = i / 99;
        let coords = [];
        for (let j = 0; j < f.dim; j++) {
            coords.push(f.a + t * (f.b - f.a));
        }
        y_diag.push(f.eval(coords));
    }
    
    let dy_diag = [];
    for (let i = 0; i < 99; i++) {
        dy_diag.push(y_diag[i + 1] - y_diag[i]);
    }
    
    let extrema_count = 0;
    for (let i = 0; i < 98; i++) {
        if (Math.sign(dy_diag[i]) !== Math.sign(dy_diag[i + 1]) && dy_diag[i] !== 0 && dy_diag[i + 1] !== 0) {
            extrema_count++;
        }
    }
    
    // 4. Boundary periodic similarity
    let boundary_diffs = [];
    for (let j = 0; j < f.dim; j++) {
        let pt_a = [];
        let pt_b = [];
        const mid = 0.5 * (f.a + f.b);
        for (let k = 0; k < f.dim; k++) {
            if (k === j) {
                pt_a.push(f.a);
                pt_b.push(f.b);
            } else {
                pt_a.push(mid);
                pt_b.push(mid);
            }
        }
        boundary_diffs.push(Math.abs(f.eval(pt_a) - f.eval(pt_b)));
    }
    const mean_boundary_diff = boundary_diffs.reduce((s, x) => s + x, 0) / f.dim;
    
    const has_singularity = (max_abs_y > 1e4 || max_abs_df > 1e6) ? 1.0 : 0.0;
    
    return {
        dim: f.dim,
        volume: Math.pow(f.b - f.a, f.dim),
        mean_y,
        std_y,
        range_y,
        std_range_ratio,
        min_abs_y,
        max_abs_y,
        mean_abs_df,
        max_abs_df,
        mean_abs_d2f,
        max_abs_d2f,
        d2f_to_df_ratio,
        max_to_mean_df,
        max_to_mean_d2f,
        extrema_count,
        mean_boundary_diff,
        has_singularity
    };
}

// 5. Hardcoded Decision Tree Logic (compiled from train_model.py output)
function predictSolver(feats) {
    const dim = feats.dim;
    const max_to_mean_d2f = feats.max_to_mean_d2f;
    const d2f_to_df_ratio = feats.d2f_to_df_ratio;
    const mean_boundary_diff = feats.mean_boundary_diff;
    const mean_abs_df = feats.mean_abs_df;
    const range_y = feats.range_y;
    const mean_abs_d2f = feats.mean_abs_d2f;
    const max_abs_d2f = feats.max_abs_d2f;
    const min_abs_y = feats.min_abs_y;
    const std_range_ratio = feats.std_range_ratio;
    const extrema_count = feats.extrema_count;

    if (dim <= 3.50) {
        if (max_to_mean_d2f <= 22.28) {
            if (d2f_to_df_ratio <= 0.10) {
                if (mean_boundary_diff <= 1.16) {
                    return { solver: "simpson", reason: "Low dimension, smooth derivatives, and gentle boundaries." };
                } else {
                    return { solver: "trapezoidal", reason: "Low dimension, smooth, and high boundary variations." };
                }
            } else {
                if (dim <= 2.50) {
                    return { solver: "romberg", reason: "Low dimension (1D/2D) and smooth curves with high curvature. Romberg extrapolation is extremely efficient here." };
                } else {
                    return { solver: "simpson", reason: "3D smooth function. Simpson's rule is selected for grid-based efficiency." };
                }
            }
        } else {
            if (mean_boundary_diff <= 1.18) {
                if (mean_abs_df <= 0.42) {
                    return { solver: "monte_carlo", reason: "Low dimension but high local curvature spikes (discontinuous steps). Monte Carlo handles localized jumps well." };
                } else {
                    return { solver: "romberg", reason: "Spiky curvature but steep slope with periodic-like boundaries; Romberg can resolve details." };
                }
            } else {
                return { solver: "monte_carlo", reason: "Localized sharp spikes combined with high boundary fluctuations. Monte Carlo prevents grid artifacts." };
            }
        }
    } else {
        if (mean_boundary_diff <= 0.52) {
            if (mean_abs_d2f <= 0.74) {
                if (max_abs_d2f <= 1.38) {
                    return { solver: "simpson", reason: "High dimension (4D/5D) with very low curvature. Simpson grid rule performs accurately." };
                } else {
                    return { solver: "monte_carlo", reason: "High dimension (4D/5D) with localized curvature spikes. Monte Carlo avoids the curse of dimensionality." };
                }
            } else {
                if (min_abs_y <= 0.00) {
                    return { solver: "trapezoidal", reason: "High dimension, low boundary diffs, but zero crossings. Trapezoidal rule provides a stable grid average." };
                } else {
                    return { solver: "simpson", reason: "High dimension, non-zero values, and high average curvature. Simpson is preferred." };
                }
            }
        } else {
            if (std_range_ratio <= 0.20) {
                if (extrema_count <= 4.50) {
                    return { solver: "monte_carlo", reason: "High dimension, high boundary variance, but low ruggedness and few oscillations. Monte Carlo is ideal." };
                } else {
                    return { solver: "trapezoidal", reason: "High dimension, high boundary variance, and high oscillations. Trapezoidal rule balances local oscillations." };
                }
            } else {
                if (dim <= 4.50) {
                    return { solver: "romberg", reason: "4D rugged function with high boundary difference. Romberg's high-order grids excel." };
                } else {
                    return { solver: "monte_carlo", reason: "5D rugged function. The curse of dimensionality forces the selection of Monte Carlo." };
                }
            }
        }
    }
}

// 6. Hook Sandbox UI Event Listeners
document.addEventListener("DOMContentLoaded", () => {
    const btnSimulate = document.getElementById("btn-simulate");
    const sandboxOutput = document.getElementById("sandbox-output");
    const defaultMsg = document.getElementById("sandbox-default-message");

    if (!btnSimulate) return;

    btnSimulate.addEventListener("click", () => {
        // Collect Inputs
        const dim = parseInt(document.getElementById("fn-dim").value);
        const structure = document.getElementById("fn-structure").value;
        const a = parseFloat(document.getElementById("fn-bounds-min").value);
        const b = parseFloat(document.getElementById("fn-bounds-max").value);

        const comp1Type = document.getElementById("comp1-type").value;
        const comp1Coeff = parseFloat(document.getElementById("comp1-coeff").value);
        const comp1ParamVal = parseFloat(document.getElementById("comp1-param").value);

        const comp2Type = document.getElementById("comp2-type").value;
        const comp2Coeff = parseFloat(document.getElementById("comp2-coeff").value);
        const comp2ParamVal = parseFloat(document.getElementById("comp2-param").value);

        // Helper to compile Component1D
        const compileComponent = (type, coeff, paramVal) => {
            switch(type) {
                case 'poly': return new PolynomialComponent(coeff, parseInt(paramVal));
                case 'exp': return new ExponentialComponent(coeff, paramVal);
                case 'trig': return new TrigComponent(coeff, paramVal);
                case 'abs': return new AbsValueComponent(coeff, paramVal);
                case 'step': return new StepComponent(coeff, paramVal);
                case 'singular': return new SingularComponent(coeff, a, paramVal);
            }
        };

        const comp1 = compileComponent(comp1Type, comp1Coeff, comp1ParamVal);
        const comp2 = compileComponent(comp2Type, comp2Coeff, comp2ParamVal);
        
        // Assemble Function
        const isSeparable = structure === "separable";
        const f = new JSFunction(dim, a, b, isSeparable, comp1, comp2);

        // Perform Live Analysis
        const feats = extractFeaturesJS(f);

        // Predict optimal solver
        const result = predictSolver(feats);

        // Render Outputs in DOM
        document.getElementById("predicted-solver").textContent = result.solver.replace("_", " ");
        document.getElementById("solver-reason").textContent = result.reason;

        // Apply visual classes for branding recommendation box
        const recBox = document.querySelector(".recommendation-box");
        recBox.className = "recommendation-box"; // reset
        if (result.solver === 'trapezoidal') recBox.classList.add("class-trap");
        else if (result.solver === 'simpson') recBox.classList.add("class-simp");
        else if (result.solver === 'romberg') recBox.classList.add("class-romberg");
        else if (result.solver === 'monte_carlo') recBox.classList.add("class-mc");

        // Set Feature Values
        document.getElementById("feat-val-dim").textContent = feats.dim.toFixed(1);
        document.getElementById("feat-val-range_y").textContent = feats.range_y.toFixed(2);
        document.getElementById("feat-val-std_range_ratio").textContent = feats.std_range_ratio.toFixed(2);
        document.getElementById("feat-val-max_to_mean_d2f").textContent = feats.max_to_mean_d2f.toFixed(2);
        document.getElementById("feat-val-d2f_to_df_ratio").textContent = feats.d2f_to_df_ratio.toFixed(2);
        document.getElementById("feat-val-mean_boundary_diff").textContent = feats.mean_boundary_diff.toFixed(2);
        document.getElementById("feat-val-extrema_count").textContent = feats.extrema_count;
        document.getElementById("feat-val-has_singularity").textContent = feats.has_singularity.toFixed(1);

        // Set formula description
        document.getElementById("formula-desc").textContent = f.getFormulaString();

        // Reveal Pane
        defaultMsg.classList.add("hidden");
        sandboxOutput.classList.remove("hidden");
    });
});
