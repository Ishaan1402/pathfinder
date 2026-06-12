function updateParallelCoordinates(trials, paretoSet) {
    const canvas = document.getElementById("parallel-coordinates-chart");
    if (!canvas) return;

    canvas.currentTrials = trials;
    canvas.currentParetoSet = paretoSet;

    if (!canvas.dataset.resizeObserved) {
        canvas.dataset.resizeObserved = "true";
        const ro = new ResizeObserver(() => {
            if (canvas.offsetWidth > 0 && canvas.offsetHeight > 0) {
                updateParallelCoordinates(canvas.currentTrials || [], canvas.currentParetoSet || new Set());
            }
        });
        ro.observe(canvas.parentElement || canvas);
    }

    const ctx = canvas.getContext("2d");
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    if (rect.width === 0 || rect.height === 0) return;

    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);
    
    const W = rect.width;
    const H = rect.height;
    
    if (!trials || trials.length === 0) {
        ctx.clearRect(0, 0, W, H);
        ctx.fillStyle = "#7e8c9f";
        ctx.font = "0.85rem Inter, system-ui, sans-serif";
        ctx.textAlign = "center";
        ctx.fillText("Awaiting trial data for parallel coordinates...", W / 2, H / 2);
        return;
    }

    const axes = window.computeAxesBounds(trials);
    const N = axes.length;
    const xPad = 50, yPad = 50;

    const currentValToY = (axis, val) => window.valToY(axis, val, H, yPad);
    const currentAxisToX = (idx) => window.axisToX(idx, N, W, xPad);

    canvas.currentAxes = axes;
    canvas.axisToX = currentAxisToX;
    canvas.valToY = currentValToY;

    if (!canvas.dataset.hoverBound) {
        canvas.dataset.hoverBound = "true";
        canvas.addEventListener("mousemove", (e) => {
            const bounds = canvas.getBoundingClientRect();
            const mx = e.clientX - bounds.left;
            const my = e.clientY - bounds.top;
            const closest = window.findClosestTrial(mx, my, canvas.currentTrials || [], canvas.currentAxes || [], canvas.axisToX, canvas.valToY, (canvas.currentAxes || []).length);
            
            if (window.HPOState.ui.parallelCoordinatesHoveredTrial !== closest) {
                window.HPOState.ui.parallelCoordinatesHoveredTrial = closest;
                canvas.style.cursor = closest ? "pointer" : "default";
                updateParallelCoordinates(canvas.currentTrials || [], canvas.currentParetoSet || new Set());
            }
        });
        canvas.addEventListener("mouseleave", () => {
            window.HPOState.ui.parallelCoordinatesHoveredTrial = null;
            updateParallelCoordinates(canvas.currentTrials || [], canvas.currentParetoSet || new Set());
        });
        canvas.addEventListener("click", () => {
            if (window.HPOState.ui.parallelCoordinatesHoveredTrial) {
                window.openTrialDetails(window.HPOState.ui.parallelCoordinatesHoveredTrial.number);
            }
        });
    }

    ctx.clearRect(0, 0, W, H);
    ctx.lineWidth = 1;
    ctx.font = "bold 0.72rem Inter, system-ui, sans-serif";
    
    axes.forEach((axis, i) => {
        const x = currentAxisToX(i);
        ctx.strokeStyle = "rgba(255, 255, 255, 0.08)";
        ctx.beginPath();
        ctx.moveTo(x, yPad);
        ctx.lineTo(x, H - yPad);
        ctx.stroke();

        ctx.fillStyle = "#94a3b8";
        ctx.textAlign = "center";
        ctx.fillText(axis.label, x, yPad - 20);

        ctx.font = "500 0.65rem JetBrains Mono, monospace";
        ctx.fillStyle = "#64748b";
        if (axis.specType === "categorical") {
            if (axis.options) {
                axis.options.forEach((opt) => {
                    const y = currentValToY(axis, opt);
                    ctx.beginPath();
                    ctx.arc(x, y, 2.5, 0, 2 * Math.PI);
                    ctx.fillStyle = "rgba(255, 255, 255, 0.3)";
                    ctx.fill();
                    ctx.fillStyle = "#64748b";
                    ctx.textAlign = i === 0 ? "left" : (i === N - 1 ? "right" : "center");
                    ctx.fillText(opt, x + (i === 0 ? 6 : (i === N - 1 ? -6 : 0)), y + 3);
                });
            }
        } else {
            ctx.textAlign = "center";
            const minStr = axis.min < 0.01 && axis.min > 0 ? axis.min.toExponential(1) : axis.min.toFixed(2);
            ctx.fillText(minStr, x, H - yPad + 14);
            const maxStr = axis.max < 0.01 && axis.max > 0 ? axis.max.toExponential(1) : axis.max.toFixed(2);
            ctx.fillText(maxStr, x, yPad - 6);
        }
        ctx.font = "bold 0.72rem Inter, system-ui, sans-serif";
    });

    trials.forEach(t => {
        const isPareto = paretoSet.has(t.number);
        const isHovered = window.HPOState.ui.parallelCoordinatesHoveredTrial && window.HPOState.ui.parallelCoordinatesHoveredTrial.number === t.number;
        const pts = [];
        for (let i = 0; i < N; i++) {
            const axis = axes[i];
            const val = axis.type === "param" ? t.params?.[axis.key] : t[axis.key];
            const px = currentAxisToX(i);
            const py = currentValToY(axis, val);
            if (py !== null) pts.push({ x: px, y: py });
        }
        if (pts.length < 2) return;

        ctx.beginPath();
        ctx.moveTo(pts[0].x, pts[0].y);
        for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i].x, pts[i].y);
        ctx.shadowBlur = 0;
        
        if (isHovered) {
            ctx.strokeStyle = window.HPOState.ui.accentColorHex;
            ctx.lineWidth = 3.5;
            ctx.shadowColor = window.HPOState.ui.accentColorHex;
            ctx.shadowBlur = 8;
        } else if (isPareto) {
            ctx.strokeStyle = "rgba(245, 158, 11, 0.95)";
            ctx.lineWidth = 2.5;
            ctx.shadowColor = "rgba(245, 158, 11, 0.4)";
            ctx.shadowBlur = 5;
        } else if (t.state === "FAIL") {
            ctx.strokeStyle = "rgba(239, 68, 68, 0.2)";
            ctx.lineWidth = 1;
            ctx.setLineDash([3, 3]);
        } else if (t.state === "PRUNED") {
            ctx.strokeStyle = "rgba(100, 116, 139, 0.15)";
            ctx.lineWidth = 1;
        } else {
            const score = t.dice !== null ? t.dice : 0;
            const ratio = Math.max(0, Math.min(1, score));
            const grad = ctx.createLinearGradient(xPad, 0, W - xPad, 0);
            grad.addColorStop(0, `hsla(260, 80%, 60%, ${0.15 + ratio * 0.3})`);
            grad.addColorStop(0.5, `hsla(310, 80%, 55%, ${0.15 + ratio * 0.4})`);
            grad.addColorStop(1, `hsla(35, 95%, 55%, ${0.2 + ratio * 0.65})`);
            ctx.strokeStyle = grad;
            ctx.lineWidth = 1.2 + ratio * 1.0;
        }
        ctx.stroke();
        ctx.setLineDash([]);
    });

    if (window.HPOState.ui.parallelCoordinatesHoveredTrial) {
        const t = window.HPOState.ui.parallelCoordinatesHoveredTrial;
        ctx.shadowBlur = 0;
        ctx.fillStyle = "rgba(15, 23, 42, 0.92)";
        ctx.strokeStyle = window.HPOState.ui.accentColorHex;
        ctx.lineWidth = 1;
        
        const lossLabel = window.HPOState.data.hpoConfig?.metric_loss_label || "Loss";
        const scoreLabel = window.HPOState.data.hpoConfig?.metric_score_label || "Score";
        const tooltipText = `Trial #${t.number} [${t.state}] — ${scoreLabel}: ${t.dice !== null ? t.dice.toFixed(4) : "N/A"}, ${lossLabel}: ${t.bce !== null ? t.bce.toFixed(4) : "N/A"}`;
        ctx.font = "600 0.75rem Inter, system-ui, sans-serif";
        const textWidth = ctx.measureText(tooltipText).width;
        const boxW = textWidth + 20, boxH = 28;
        const boxX = Math.max(10, Math.min(W - boxW - 10, W / 2 - boxW / 2));
        const boxY = H - yPad + 18;
        
        ctx.beginPath();
        ctx.roundRect(boxX, boxY, boxW, boxH, 4);
        ctx.fill();
        ctx.stroke();
        
        ctx.fillStyle = window.HPOState.ui.accentColorHex;
        ctx.textAlign = "left";
        ctx.fillText(tooltipText, boxX + 10, boxY + 17);
    }
}

window.updateParallelCoordinates = updateParallelCoordinates;
