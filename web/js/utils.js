function paramLabel(key) {
    if (!window.HPOState.data.hpoConfig) return key;
    return window.HPOState.data.hpoConfig.param_labels?.[key] || key;
}

function fanovaParamLabel(param) {
    const legacy = { encoder_name: "model_capacity" };
    return paramLabel(legacy[param] || param);
}

function formatTrainResolution(trial) {
    const key =
        window.HPOState.data.hpoConfig?.eval_protocol?.train_resolution_param || "resolution";
    const val =
        trial.train_resolution ??
        trial.params?.[key] ??
        trial.params?.resolution;
    return val !== undefined && val !== null ? val : "-";
}

function formatMetric(val, empty = "-") {
    return val !== null && val !== undefined ? Number(val).toFixed(4) : empty;
}

function formatLR(value) {
    if (value === undefined || value === null) return "-";
    const n = Number(value);
    if (!Number.isFinite(n)) return "-";
    if (n >= 0.01) return n.toFixed(3);
    if (n >= 0.001) return n.toFixed(4);
    return n.toExponential(0);
}

function formatTime(totalSeconds) {
    const mins = Math.floor(totalSeconds / 60);
    const secs = totalSeconds % 60;
    return `${mins}m ${secs}s`;
}

function copyTrialConfigToClipboard(trialNumber, format) {
    const trial = window.HPOState.data.trials.find(t => t.number === trialNumber);
    if (!trial) return;

    let text = "";
    if (format === 'json') {
        text = JSON.stringify(trial.params, null, 2);
    } else if (format === 'cli') {
        text = Object.entries(trial.params)
            .map(([k, v]) => `--${k} ${v}`)
            .join(" ");
    }

    navigator.clipboard.writeText(text).then(() => {
        showToast(`Trial #${trialNumber} config copied as ${format.toUpperCase()}`);
    }).catch(err => {
        console.error("Failed to copy: ", err);
    });
}

function copyTrialsToJson() {
    if (!window.HPOState.data.trials || window.HPOState.data.trials.length === 0) {
        showToast("No trial data available to copy.");
        return;
    }
    const cleanJson = JSON.stringify(window.HPOState.data.trials, null, 2);
    navigator.clipboard.writeText(cleanJson).then(() => {
        showToast("All trials data copied to clipboard as JSON.");
    }).catch(err => {
        console.error("Failed to copy trials JSON: ", err);
    });
}

function showEmptyState(containerId, message) {
    const container = document.getElementById(containerId);
    if (!container) return;

    // Chart.js instance cleanup
    let chartKey = null;
    if (containerId === "pareto-chart") chartKey = "pareto";
    if (chartKey && window.HPOState?.charts?.[chartKey]?.instance) {
        try {
            window.HPOState.charts[chartKey].instance.destroy();
        } catch (e) {
            console.error("Error destroying chart instance:", e);
        }
        window.HPOState.charts[chartKey].instance = null;
    }

    if (container.tagName.toLowerCase() === 'canvas') {
        let placeholder = container.parentElement.querySelector('.chart-empty-placeholder');
        if (!placeholder) {
            placeholder = document.createElement('div');
            placeholder.className = 'chart-empty-placeholder';
            placeholder.style.display = 'flex';
            placeholder.style.alignItems = 'center';
            placeholder.style.justifyContent = 'center';
            placeholder.style.width = '100%';
            placeholder.style.height = '100%';
            placeholder.style.minHeight = '220px';
            placeholder.style.color = 'var(--text-muted)';
            placeholder.style.fontFamily = 'var(--font-main)';
            placeholder.style.fontSize = '0.875rem';
            placeholder.style.textAlign = 'center';
            placeholder.style.backgroundColor = 'rgba(255, 255, 255, 0.02)';
            placeholder.style.border = '1px solid var(--border-color, #2d3748)';
            placeholder.style.padding = '20px';
            placeholder.style.boxSizing = 'border-box';
            container.parentElement.appendChild(placeholder);
        }
        placeholder.textContent = message;
        placeholder.style.display = 'flex';
        container.style.display = 'none';
    } else {
        container.innerHTML = '';
        const emptyDiv = document.createElement('div');
        emptyDiv.style.display = 'flex';
        emptyDiv.style.alignItems = 'center';
        emptyDiv.style.justifyContent = 'center';
        emptyDiv.style.width = '100%';
        emptyDiv.style.height = '100%';
        emptyDiv.style.minHeight = '150px';
        emptyDiv.style.color = 'var(--text-muted)';
        emptyDiv.style.fontFamily = 'var(--font-main)';
        emptyDiv.style.fontSize = '0.875rem';
        emptyDiv.style.textAlign = 'center';
        emptyDiv.style.border = '1px solid var(--border-color, #2d3748)';
        emptyDiv.style.backgroundColor = 'rgba(255, 255, 255, 0.02)';
        emptyDiv.style.padding = '20px';
        emptyDiv.style.boxSizing = 'border-box';
        emptyDiv.textContent = message;
        container.appendChild(emptyDiv);
    }
}

function hideEmptyState(containerId, recreateCanvas = false) {
    const container = document.getElementById(containerId);
    if (!container) return;

    if (container.tagName.toLowerCase() === 'canvas') {
        const placeholder = container.parentElement.querySelector('.chart-empty-placeholder');
        if (placeholder) {
            placeholder.style.display = 'none';
        }
        
        container.style.display = "block";
        
        if (recreateCanvas) {
            // Destroy the stale Chart.js instance that references the old canvas.
            // If we don't do this, updateChart() sees instance != null and tries to
            // call .update() on the detached (removed) canvas — nothing renders.
            let chartKey = null;
            if (containerId === "pareto-chart") chartKey = "pareto";
            if (chartKey && window.HPOState?.charts?.[chartKey]?.instance) {
                try { window.HPOState.charts[chartKey].instance.destroy(); } catch (e) {}
                window.HPOState.charts[chartKey].instance = null;
            }
            // Recreate canvas to completely reset Chart.js internal context reference
            const newCanvas = document.createElement("canvas");
            newCanvas.id = containerId;
            newCanvas.style.maxHeight = container.style.maxHeight || "100%";
            newCanvas.style.maxWidth = container.style.maxWidth || "100%";
            newCanvas.style.display = "block";
            container.replaceWith(newCanvas);
        }
    }
}

function initHpoMarks() {
    const svg = window.HPOState?.constants?.HPO_MARK_SVG;
    if (!svg) return;
    document.querySelectorAll(".brand-mark").forEach((el) => {
        el.innerHTML = svg;
    });
}

window.showEmptyState = showEmptyState;
window.hideEmptyState = hideEmptyState;
window.initHpoMarks = initHpoMarks;

