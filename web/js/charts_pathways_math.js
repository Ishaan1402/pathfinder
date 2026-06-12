function computeAxesBounds(trials) {
    const paramKeys = [];
    trials.forEach(t => {
        if (t.params) {
            Object.keys(t.params).forEach(k => {
                if (!paramKeys.includes(k)) paramKeys.push(k);
            });
        }
    });
    paramKeys.sort();

    const axes = [];
    paramKeys.forEach(k => {
        axes.push({ key: k, label: k.replace(/_/g, " "), type: "param" });
    });
    axes.push({ key: "dice", label: window.HPOState.data.hpoConfig?.metric_score_label || "Dice", type: "metric" });
    axes.push({ key: "bce", label: window.HPOState.data.hpoConfig?.metric_loss_label || "BCE", type: "metric" });

    axes.forEach(axis => {
        if (axis.type === "param") {
            let spec = window.HPOState.data.activeSearchSpace?.[axis.key] ||
                       window.HPOState.data.hpoConfig?.active_search_space?.[axis.key] ||
                       window.HPOState.data.latestStudyData?.active_search_space?.[axis.key];
            
            if (spec) {
                axis.specType = spec.type;
                if (spec.type === "categorical") {
                    axis.options = spec.options || [];
                } else {
                    axis.min = spec.min !== undefined ? spec.min : Math.min(...trials.map(t => t.params?.[axis.key] ?? 0));
                    axis.max = spec.max !== undefined ? spec.max : Math.max(...trials.map(t => t.params?.[axis.key] ?? 0));
                }
            } else {
                const vals = trials.map(t => t.params?.[axis.key]).filter(v => v !== undefined && v !== null);
                const isNum = vals.every(v => typeof v === "number");
                if (isNum && vals.length > 0) {
                    axis.specType = "float";
                    axis.min = Math.min(...vals);
                    axis.max = Math.max(...vals);
                } else {
                    axis.specType = "categorical";
                    axis.options = Array.from(new Set(vals)).sort();
                }
            }
        } else {
            const vals = trials.map(t => t[axis.key]).filter(v => v !== undefined && v !== null);
            axis.specType = "float";
            axis.min = vals.length > 0 ? Math.min(...vals) : 0;
            axis.max = vals.length > 0 ? Math.max(...vals) : 1;
        }
    });
    return axes;
}

function valToY(axis, val, H, yPad) {
    if (val === undefined || val === null) return null;
    const innerH = H - 2 * yPad;
    if (axis.specType === "categorical") {
        if (!axis.options || axis.options.length === 0) return H - yPad - innerH / 2;
        const idx = axis.options.indexOf(val);
        if (idx === -1) return H - yPad - innerH / 2;
        if (axis.options.length <= 1) return H - yPad - innerH / 2;
        return H - yPad - (idx / (axis.options.length - 1)) * innerH;
    } else {
        const min = axis.min;
        const max = axis.max;
        if (max === min) return H - yPad - innerH / 2;
        
        let ratio;
        if (axis.specType === "float_log" && min > 0 && max > 0) {
            ratio = (Math.log10(val) - Math.log10(min)) / (Math.log10(max) - Math.log10(min));
        } else {
            ratio = (val - min) / (max - min);
        }
        ratio = Math.max(0, Math.min(1, ratio));
        return H - yPad - ratio * innerH;
    }
}

function axisToX(idx, N, W, xPad) {
    if (N <= 1) return xPad;
    return xPad + idx * ((W - 2 * xPad) / (N - 1));
}

function findClosestTrial(mx, my, trials, axes, currentAxisToX, currentValToY, N) {
    let closestTrial = null;
    let minDist = 15;
    
    trials.forEach(t => {
        const pts = [];
        for (let i = 0; i < N; i++) {
            const axis = axes[i];
            const val = axis.type === "param" ? t.params?.[axis.key] : t[axis.key];
            const px = currentAxisToX(i);
            const py = currentValToY(axis, val);
            if (py !== null) pts.push({ x: px, y: py });
        }
        
        for (let i = 0; i < pts.length - 1; i++) {
            const dist = window.distToSegment({ x: mx, y: my }, pts[i], pts[i+1]);
            if (dist < minDist) {
                minDist = dist;
                closestTrial = t;
            }
        }
    });
    return closestTrial;
}

window.computeAxesBounds = computeAxesBounds;
window.valToY = valToY;
window.axisToX = axisToX;
window.findClosestTrial = findClosestTrial;
