// script.js

// 1. Embedded Project Metrics (from ml_integration_results.json)
const projectMetrics = {
    totalFunctions: 600,
    datasetGenTimeS: 0.50,
    modelTrainingTimeS: 0.42,
    totalExecutionTimeS: 1.01,
    dtAccuracy: 0.4733,
    rfAccuracy: 0.5533
};

// 2. DOM Elements
document.addEventListener("DOMContentLoaded", () => {
    // Nav elements
    const navButtons = document.querySelectorAll(".nav-btn");
    const tabPanes = document.querySelectorAll(".tab-pane");

    // Visualization elements
    const visTabButtons = document.querySelectorAll(".vis-tab-btn");
    const visDisplay = document.getElementById("vis-display");

    // Dynamic metrics
    document.getElementById("val-total-functions").textContent = projectMetrics.totalFunctions;
    document.getElementById("val-gen-time").textContent = `${projectMetrics.datasetGenTimeS.toFixed(2)}s`;
    document.getElementById("val-train-time").textContent = `${projectMetrics.modelTrainingTimeS.toFixed(2)}s`;
    document.getElementById("val-total-time").textContent = `${projectMetrics.totalExecutionTimeS.toFixed(2)}s`;
    document.getElementById("val-dt-accuracy").textContent = `${(projectMetrics.dtAccuracy * 100).toFixed(1)}%`;
    document.getElementById("val-rf-accuracy").textContent = `${(projectMetrics.rfAccuracy * 100).toFixed(1)}%`;

    // 3. Tab Switching Logic
    navButtons.forEach(btn => {
        btn.addEventListener("click", () => {
            const targetTab = btn.getAttribute("data-tab");

            // Toggle Nav Active state
            navButtons.forEach(b => b.classList.remove("active"));
            btn.classList.add("active");

            // Toggle Pane display
            tabPanes.forEach(pane => {
                pane.classList.remove("active");
                if (pane.id === `tab-${targetTab}`) {
                    pane.classList.add("active");
                }
            });
        });
    });

    // 4. Visualization Graph Switching
    visTabButtons.forEach(btn => {
        btn.addEventListener("click", () => {
            const imgPath = btn.getAttribute("data-img");
            
            // Toggle active visual tab
            visTabButtons.forEach(b => b.classList.remove("active"));
            btn.classList.add("active");

            // Update Image Src
            visDisplay.style.opacity = 0;
            setTimeout(() => {
                visDisplay.src = imgPath;
                visDisplay.style.opacity = 1;
            }, 150);
        });
    });

    // 5. Progress Ring/Gauge Animations
    const setGaugeProgress = (elementId, percent) => {
        const circle = document.getElementById(elementId);
        if (circle) {
            const radius = circle.r.baseVal.value;
            const circumference = 2 * Math.PI * radius;
            circle.style.strokeDasharray = `${circumference} ${circumference}`;
            
            // Animate gauge from empty to target percentage
            setTimeout(() => {
                const offset = circumference - (percent * circumference);
                circle.style.strokeDashoffset = offset;
            }, 100);
        }
    };

    setGaugeProgress("gauge-dt", projectMetrics.dtAccuracy);
    setGaugeProgress("gauge-rf", projectMetrics.rfAccuracy);

    // 6. Interactive Sandbox UI helpers (Dynamic Parameter Labels)
    const setupSandboxFormLabels = () => {
        const comp1Type = document.getElementById("comp1-type");
        const comp1ParamLabel = document.getElementById("comp1-param-label");
        const comp1Param = document.getElementById("comp1-param");

        const comp2Type = document.getElementById("comp2-type");
        const comp2ParamLabel = document.getElementById("comp2-param-label");
        const comp2Param = document.getElementById("comp2-param");

        const updateLabels = (selectEl, labelEl, inputEl) => {
            const type = selectEl.value;
            if (type === "poly") {
                labelEl.textContent = "Exponent (p)";
                inputEl.value = "2";
                inputEl.step = "1";
            } else if (type === "exp") {
                labelEl.textContent = "Growth (c)";
                inputEl.value = "1.0";
                inputEl.step = "0.1";
            } else if (type === "trig") {
                labelEl.textContent = "Freq (c)";
                inputEl.value = "3.14";
                inputEl.step = "0.01";
            } else if (type === "abs" || type === "step") {
                labelEl.textContent = "Kink Center (xc)";
                inputEl.value = "0.5";
                inputEl.step = "0.1";
            } else if (type === "singular") {
                labelEl.textContent = "Offset (eps)";
                inputEl.value = "0.001";
                inputEl.step = "0.001";
            }
        };

        if (comp1Type && comp1ParamLabel && comp1Param) {
            comp1Type.addEventListener("change", () => updateLabels(comp1Type, comp1ParamLabel, comp1Param));
        }
        if (comp2Type && comp2ParamLabel && comp2Param) {
            comp2Type.addEventListener("change", () => updateLabels(comp2Type, comp2ParamLabel, comp2Param));
        }
    };

    setupSandboxFormLabels();
});
