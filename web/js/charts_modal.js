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
    const diceData = history.map(h => h.dice);
    const bceData = history.map(h => h.bce);

    const lossLabel = window.HPOState.data.hpoConfig?.metric_loss_label || "BCE";
    const scoreLabel = window.HPOState.data.hpoConfig?.metric_score_label || "Dice";
    const lossText = lossLabel.toLowerCase().includes("loss") ? lossLabel : `${lossLabel} Loss`;
    const scoreText = scoreLabel.toLowerCase().includes("score") || scoreLabel.toLowerCase().includes("accuracy") ? scoreLabel : `${scoreLabel} Score`;

    if (window.HPOState.charts.modalHistory.instance) {
        window.HPOState.charts.modalHistory.instance.data.labels = labels;
        window.HPOState.charts.modalHistory.instance.data.datasets[0].data = diceData;
        window.HPOState.charts.modalHistory.instance.data.datasets[0].label = scoreText;
        window.HPOState.charts.modalHistory.instance.data.datasets[0].borderColor = window.HPOState.ui.accentColorHex;
        window.HPOState.charts.modalHistory.instance.data.datasets[1].data = bceData;
        window.HPOState.charts.modalHistory.instance.data.datasets[1].label = lossText;
        if (window.HPOState.charts.modalHistory.instance.options.scales?.["y-dice"]?.title) {
            window.HPOState.charts.modalHistory.instance.options.scales["y-dice"].title.text = scoreText;
        }
        if (window.HPOState.charts.modalHistory.instance.options.scales?.["y-bce"]?.title) {
            window.HPOState.charts.modalHistory.instance.options.scales["y-bce"].title.text = lossText;
        }
        window.HPOState.charts.modalHistory.instance.update("none");
        return;
    }

    window.HPOState.charts.modalHistory.instance = new Chart(ctx, {
        type: "line",
        data: {
            labels,
            datasets: [
                {
                    label: scoreText,
                    data: diceData,
                    borderColor: window.HPOState.ui.accentColorHex,
                    backgroundColor: "transparent",
                    yAxisID: "y-dice",
                    tension: 0.15,
                    borderWidth: 3
                },
                {
                    label: lossText,
                    data: bceData,
                    borderColor: "#ef4444",
                    backgroundColor: "transparent",
                    yAxisID: "y-bce",
                    tension: 0.15,
                    borderWidth: 2,
                    borderDash: [4, 4]
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                "y-dice": {
                    type: "linear",
                    position: "left",
                    title: { display: true, text: scoreText, color: "#7e8c9f" },
                    grid: { color: "rgba(255, 255, 255, 0.03)" },
                    ticks: { color: "#7e8c9f" }
                },
                "y-bce": {
                    type: "linear",
                    position: "right",
                    title: { display: true, text: "BCE Loss", color: "#7e8c9f" },
                    grid: { drawOnChartArea: false },
                    ticks: { color: "#7e8c9f" }
                },
                x: {
                    ticks: { color: "#7e8c9f" },
                    grid: { color: "rgba(255, 255, 255, 0.03)" }
                }
            },
            plugins: {
                legend: { labels: { color: "#e2e8f0" } }
            }
        }
    });
}

window.updateModalChart = updateModalChart;
