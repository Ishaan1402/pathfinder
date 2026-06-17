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
    const scoreData = history.map(h => h.score !== undefined ? h.score : h.dice);
    const lossData = history.map(h => h.loss !== undefined ? h.loss : h.bce);

    const hasScore = scoreData.some(d => d !== null && d !== undefined);
    const hasLoss = lossData.some(d => d !== null && d !== undefined);

    const lossLabel = window.HPOState.data.hpoConfig?.metric_loss_label || "BCE";
    const scoreLabel = window.HPOState.data.hpoConfig?.metric_score_label || "Dice";
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
