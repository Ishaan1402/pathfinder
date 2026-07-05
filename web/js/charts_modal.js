function openTrialDetails(trialNumber) {
    trialNumber = Number(trialNumber);
    const trials = window.HPOState.data.trials || [];
    const trial = trials.find(t => t.number === trialNumber);
    if (!trial) return;

    document.getElementById("modal-trial-title").textContent = `Trial #${trial.number}`;

    const paramsEl = document.getElementById("modal-params");
    if (paramsEl) {
        const params = trial.params_display || trial.params || {};
        paramsEl.innerHTML = Object.entries(params)
            .map(([k, v]) => `<div class="param-item"><div class="param-label">${k}</div><div class="param-value td-mono">${v}</div></div>`)
            .join("");
    }

    const oomSection = document.getElementById("modal-oom-section");
    if (oomSection) {
        oomSection.style.display = trial.oom_triggered ? "block" : "none";
    }

    const state = trial.state || "";
    const titleEl = document.getElementById("modal-rationale-title");
    if (titleEl) {
        titleEl.textContent = state === "COMPLETE" ? "Result" : "Status";
    }

    const reasoningEl = document.getElementById("modal-reasoning");
    if (reasoningEl) {
        reasoningEl.textContent = state === "COMPLETE"
            ? `Completed with ${trial.loss != null ? `loss ${Number(trial.loss).toFixed(4)}` : "unknown loss"} and ${trial.score != null ? `score ${Number(trial.score).toFixed(4)}` : "unknown score"}.`
            : state === "PRUNED" ? "Pruned by median-rule early stopping." :
            state === "FAIL" ? `Failed${trial.oom_triggered ? " (Out of Memory)" : ""}.` :
            "Awaiting results.";
    }

    const history = trial.history || [];
    updateModalChart(history);

    const modal = document.getElementById("detail-modal");
    if (modal) modal.classList.add("active");
    window.HPOState.ui.isModalOpen = true;
}

function closeModal(event) {
    if (event && event.target !== event.currentTarget) return;
    closeModalDirect();
}

function closeModalDirect() {
    const modal = document.getElementById("detail-modal");
    if (modal) modal.classList.remove("active");
    window.HPOState.ui.isModalOpen = false;
    if (window.HPOState.charts.modalHistory.instance) {
        window.HPOState.charts.modalHistory.instance.destroy();
        window.HPOState.charts.modalHistory.instance = null;
    }
    if (window.HPOState.render.pendingRender) {
        window.HPOState.render.pendingRender = false;
        if (window.renderStudyDetails && window.HPOState.data.latestStudyData) {
            window.renderStudyDetails(window.HPOState.data.latestStudyData);
        }
    }
}

window.openTrialDetails = openTrialDetails;
window.closeModal = closeModal;
window.closeModalDirect = closeModalDirect;

function updateModalChart(history) {
    const ctx = document.getElementById("modal-history-chart")?.getContext("2d");
    if (!ctx) return;
    
    if (!history || history.length === 0) {
        if (window.HPOState.charts.modalHistory.instance) {
            window.HPOState.charts.modalHistory.instance.destroy();
            window.HPOState.charts.modalHistory.instance = null;
        }
        ctx.clearRect(0, 0, ctx.canvas.width || 400, ctx.canvas.height || 200);
        return;
    }

    const labels = history.map(h => `E${h.epoch}`);
    const scoreData = history.map(h => h.score);
    const lossData = history.map(h => h.loss);

    const hasScore = scoreData.some(d => d !== null && d !== undefined);
    const hasLoss = lossData.some(d => d !== null && d !== undefined);

    const lossLabel = window.HPOState.data.hpoConfig?.metric_loss_label || "Loss";
    const scoreLabel = window.HPOState.data.hpoConfig?.metric_score_label || "Score";
    const lossText = lossLabel.toLowerCase().includes("loss") ? lossLabel : `${lossLabel} Loss`;
    const scoreText = scoreLabel.toLowerCase().includes("score") || scoreLabel.toLowerCase().includes("accuracy") ? scoreLabel : `${scoreLabel} Score`;

    const datasets = [];
    const scales = {
        x: {
            ticks: { color: "#7e8c9f" },
            grid: { color: "rgba(255, 255, 255, 0.03)" }
        }
    };

    if (hasScore) {
        datasets.push({
            label: scoreText,
            data: scoreData,
            borderColor: window.HPOState.ui.accentColorHex,
            backgroundColor: "transparent",
            yAxisID: "y-score",
            tension: 0.15,
            borderWidth: 3
        });
        scales["y-score"] = {
            type: "linear",
            position: "left",
            title: { display: true, text: scoreText, color: "#7e8c9f" },
            grid: { color: "rgba(255, 255, 255, 0.03)" },
            ticks: { color: "#7e8c9f" }
        };
    }

    if (hasLoss) {
        datasets.push({
            label: lossText,
            data: lossData,
            borderColor: "#ef4444",
            backgroundColor: "transparent",
            yAxisID: "y-loss",
            tension: 0.15,
            borderWidth: 2,
            borderDash: [4, 4]
        });
        scales["y-loss"] = {
            type: "linear",
            position: hasScore ? "right" : "left",
            title: { display: true, text: lossText, color: "#7e8c9f" },
            grid: { drawOnChartArea: !hasScore },
            ticks: { color: "#7e8c9f" }
        };
    }

    if (window.HPOState.charts.modalHistory.instance) {
        window.HPOState.charts.modalHistory.instance.destroy();
    }

    window.HPOState.charts.modalHistory.instance = new Chart(ctx, {
        type: "line",
        data: {
            labels,
            datasets
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales,
            plugins: {
                legend: { labels: { color: "#e2e8f0" } },
                tooltip: { mode: "index", intersect: false }
            }
        }
    });
}

window.updateModalChart = updateModalChart;
