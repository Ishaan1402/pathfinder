function updateAshaTimeline(trials) {
    const container = document.getElementById("asha-progression-timeline");
    if (!container) return;

    const validTrials = trials
        .filter(t => t.state === "COMPLETE" || t.state === "PRUNED" || t.state === "FAIL" || t.state === "RUNNING")
        .sort((a, b) => b.number - a.number);

    if (validTrials.length === 0) {
        window.showEmptyState("asha-progression-timeline", "No trials registered yet. Run training worker.");
        return;
    }

    const scoreLabel = window.HPOState.data.hpoConfig?.metric_score_label || "Dice";
    const lossLabel = window.HPOState.data.hpoConfig?.metric_loss_label || "BCE";

    let html = "";
    validTrials.forEach(t => {
        const epochs = Object.keys(t.intermediate_values || {}).map(Number).sort((a, b) => a - b);
        
        let stateClass = "status-complete-pill";
        let stateLabel = t.state;
        if (t.state === "PRUNED") {
            stateClass = "status-pruned-pill";
        } else if (t.state === "FAIL") {
            stateClass = "status-failed-pill";
            stateLabel = "FAILED";
        } else if (t.state === "RUNNING") {
            stateClass = "status-running-pill";
        }

        let progressionHtml = "";
        const maxEpochs = Math.max(5, ...trials.map(x => Object.keys(x.intermediate_values || {}).length));
        
        for (let ep = 0; ep < maxEpochs; ep++) {
            const hasVal = t.intermediate_values && t.intermediate_values[ep] !== undefined;
            const val = hasVal ? t.intermediate_values[ep] : null;
            
            let nodeClass = "asha-node-pending";
            let nodeTooltip = `Epoch ${ep + 1}: Not reached`;
            let innerValStr = "";

            if (hasVal) {
                innerValStr = val.toFixed(2);
                if (t.state === "PRUNED" && ep === epochs[epochs.length - 1]) {
                    nodeClass = "asha-node-pruned";
                    nodeTooltip = `Epoch ${ep + 1} (Pruned) | ${scoreLabel}: ${val.toFixed(4)}`;
                } else {
                    nodeClass = "asha-node-active";
                    nodeTooltip = `Epoch ${ep + 1} | ${scoreLabel}: ${val.toFixed(4)}`;
                }
            } else if (t.state === "FAIL" && ep === epochs.length) {
                nodeClass = "asha-node-failed";
                nodeTooltip = `Epoch ${ep + 1} (OOM Failure)`;
                innerValStr = "❌";
            }

            progressionHtml += `
                <div class="asha-epoch-node ${nodeClass}" title="${nodeTooltip}">
                    <span class="asha-node-epoch-label">E${ep + 1}</span>
                    <span class="asha-node-val">${innerValStr}</span>
                </div>
                ${ep < maxEpochs - 1 ? `<div class="asha-epoch-connector ${hasVal && ep < epochs.length - 1 ? "active" : ""}"></div>` : ""}
            `;
        }

        const paramsStr = Object.entries(t.params)
            .map(([k, v]) => `${k.replace(/_/g, " ")}: <span style="color:#f1f5f9; font-family:var(--font-mono);">${typeof v === 'number' && v % 1 !== 0 ? v.toFixed(4) : v}</span>`)
            .join("  ·  ");

        html += `
            <div style="background: rgba(255, 255, 255, 0.02); border: 1px solid var(--border-color); border-radius: 0; padding: 10px 14px; display: flex; flex-direction: column; gap: 8px;">
                <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">
                    <div style="display: flex; align-items: center; gap: 10px;">
                        <strong style="font-family: var(--font-mono); font-size: 0.9rem; color: #fff;">Trial #${t.number}</strong>
                        <span class="status-pill ${stateClass}" style="font-size: 0.72rem; padding: 2px 6px;">${stateLabel}</span>
                        <span style="font-size: 0.75rem; color: var(--text-muted);">${paramsStr}</span>
                    </div>
                    <div style="font-family: var(--font-mono); font-size: 0.8rem;">
                        ${t.dice !== null ? `<span style="color:#10b981; font-weight:600;">${scoreLabel}: ${t.dice.toFixed(4)}</span>` : ""}
                        ${t.bce !== null ? `  <span style="color:var(--text-muted);">(${lossLabel}: ${t.bce.toFixed(4)})</span>` : ""}
                    </div>
                </div>
                <div style="display: flex; align-items: center; margin-top: 4px; padding: 0 4px; overflow-x: auto; min-height: 44px;">
                    ${progressionHtml}
                </div>
            </div>
        `;
    });

    container.innerHTML = html;
}

window.updateAshaTimeline = updateAshaTimeline;
