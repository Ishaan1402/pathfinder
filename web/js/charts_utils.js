function dist2(v, w) { return (v.x - w.x) ** 2 + (v.y - w.y) ** 2; }
function distToSegment(p, v, w) {
    const l2 = dist2(v, w);
    if (l2 === 0) return dist2(p, v);
    let t = ((p.x - v.x) * (w.x - v.x) + (p.y - v.y) * (w.y - v.y)) / l2;
    t = Math.max(0, Math.min(1, t));
    return Math.sqrt(dist2(p, { x: v.x + t * (w.x - v.x), y: v.y + t * (w.y - v.y) }));
}

function prepareParetoDatasets(trials, paretoSet, directions, isSingleObj, isMinimize, metricLabel, lossLabel, scoreLabel, accentColor) {
    const runningPoints = [];
    let datasets = [];

    if (isSingleObj) {
        const sortedCompleted = trials
            .filter(t => t.state === "COMPLETE" && (isMinimize ? t.bce !== null : t.dice !== null))
            .sort((a, b) => a.number - b.number);
            
        let currentBest = null;
        const bestLinePoints = [];
        
        sortedCompleted.forEach(t => {
            const val = isMinimize ? t.bce : t.dice;
            if (currentBest === null) {
                currentBest = val;
            } else {
                currentBest = isMinimize ? Math.min(currentBest, val) : Math.max(currentBest, val);
            }
            bestLinePoints.push({ x: t.number, y: currentBest, label: "Best Value" });
        });
        
        const completedPoints = sortedCompleted.map(t => ({
            x: t.number,
            y: isMinimize ? t.bce : t.dice,
            label: `Trial ${t.number}`
        }));
        
        const runningTrials = trials.filter(t => t.state === "RUNNING");
        runningTrials.forEach(t => {
            const val = isMinimize ? t.bce : t.dice;
            if (val !== null && val !== undefined) {
                runningPoints.push({ x: t.number, y: val, label: `Trial ${t.number} (running)` });
            }
        });
        
        datasets = [
            {
                label: "All Trials",
                data: completedPoints,
                backgroundColor: "rgba(126, 140, 159, 0.4)",
                borderColor: "rgba(126, 140, 159, 0.5)",
                pointRadius: 6,
                pointHoverRadius: 9,
                hoverBorderWidth: 2,
                type: "scatter"
            },
            {
                label: "Best Value (Progression)",
                data: bestLinePoints,
                backgroundColor: "rgba(16, 185, 129, 0.9)",
                borderColor: "rgba(16, 185, 129, 0.8)",
                pointRadius: 4,
                pointHoverRadius: 6,
                showLine: true,
                fill: false,
                stepped: true,
                type: "line"
            }
        ];
        
        if (runningPoints.length > 0) {
            datasets.push({
                label: "Active Running",
                data: runningPoints,
                backgroundColor: accentColor,
                borderColor: accentColor,
                pointRadius: 8,
                pointHoverRadius: 11,
                hoverBorderWidth: 2,
                type: "scatter"
            });
        }
    } else {
        const scatterPoints = [];
        const paretoPoints = [];
        trials.forEach(t => {
            if (t.bce === null || t.dice === null) return;
            const point = { x: t.bce, y: t.dice, label: `Trial ${t.number}` };
            if (t.state === "RUNNING") {
                runningPoints.push(point);
            } else if (paretoSet.has(t.number)) {
                paretoPoints.push(point);
            } else {
                scatterPoints.push(point);
            }
        });
        
        paretoPoints.sort((a, b) => a.x - b.x);
        
        datasets = [
            {
                label: "Other Trials",
                data: scatterPoints,
                backgroundColor: "rgba(126, 140, 159, 0.4)",
                borderColor: "rgba(126, 140, 159, 0.5)",
                pointRadius: 6,
                pointHoverRadius: 9,
                hoverBorderWidth: 2,
                type: "scatter"
            },
            {
                label: "Pareto Front",
                data: paretoPoints,
                backgroundColor: "rgba(16, 185, 129, 0.9)",
                borderColor: "rgba(16, 185, 129, 0.8)",
                pointRadius: 8,
                pointHoverRadius: 11,
                hoverBorderWidth: 2,
                showLine: true,
                fill: false,
                tension: 0.1,
                type: "line"
            }
        ];
        
        if (runningPoints.length > 0) {
            datasets.push({
                label: "Active Running",
                data: runningPoints,
                backgroundColor: accentColor,
                borderColor: accentColor,
                pointRadius: 8,
                pointHoverRadius: 11,
                hoverBorderWidth: 2,
                type: "scatter"
            });
        }
    }

    const totalPoints = trials.filter(t => (isSingleObj ? (isMinimize ? t.bce !== null : t.dice !== null) : (t.bce !== null && t.dice !== null))).length;
    return { datasets, totalPoints };
}

window.dist2 = dist2;
window.distToSegment = distToSegment;
window.prepareParetoDatasets = prepareParetoDatasets;
