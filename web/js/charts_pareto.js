function updateChart(trials, paretoSet, directions) {
    if (!trials || trials.length === 0) {
        window.showEmptyState("pareto-chart", "No trial data yet. Start a worker.");
        return;
    }
    const completedTrials = trials.filter(t => t.state === "COMPLETE");
    if (completedTrials.length < 2) {
        window.showEmptyState("pareto-chart", "Run at least 2 completed trials to see Pareto front");
        return;
    }
    window.hideEmptyState("pareto-chart", true);

    const ctx = document.getElementById("pareto-chart")?.getContext("2d");
    if (!ctx) return;
    
    const isSingleObj = !directions || directions.length === 1;
    const isMinimize = directions && directions[0] === "MINIMIZE";
    
    const lossLabel = window.HPOState.data.hpoConfig?.metric_loss_label || "Loss";
    const scoreLabel = window.HPOState.data.hpoConfig?.metric_score_label || "Score";
    const metricLabel = isMinimize ? lossLabel : scoreLabel;
    const accentColor = window.HPOState.ui.accentColorHex;

    const { datasets, totalPoints } = window.prepareParetoDatasets(
        trials, paretoSet, directions, isSingleObj, isMinimize, metricLabel, lossLabel, scoreLabel, accentColor
    );
    
    const headerSpan = document.getElementById("pareto-chart")?.closest(".card")?.querySelector(".card-header span");
    if (headerSpan) {
        if (isSingleObj) {
            headerSpan.textContent = `Optimization History (${metricLabel})`;
        } else {
            const extraTag = (directions && directions.length > 2) ? ` [${directions.length} Objectives]` : "";
            headerSpan.textContent = `Pareto Frontier (${lossLabel} vs. ${scoreLabel})${extraTag}`;
        }
    }
    
    const getXText = () => isSingleObj ? "Trial Number" : `${lossLabel} (Lower is better)`;
    const getYText = () => isSingleObj ? `${metricLabel} (${isMinimize ? "Lower is better" : "Higher is better"})` : `${scoreLabel} (Higher is better)`;
    
    const getTooltipLabel = (context) => {
        const rawPoint = context.raw;
        const match = rawPoint.label?.match(/Trial (\d+)/);
        if (match) {
            const trialNum = parseInt(match[1], 10);
            const trial = trials.find(t => t.number === trialNum);
            if (trial) {
                const paramsList = Object.entries(trial.params)
                    .map(([k, v]) => `${k.replace(/_/g, " ")}: ${typeof v === 'number' && v % 1 !== 0 ? v.toFixed(4) : v}`)
                    .join(", ");
                
                let labelText = `${rawPoint.label} | `;
                if (isSingleObj) {
                    labelText += `${metricLabel}: ${rawPoint.y.toFixed(4)}`;
                } else {
                    labelText += `${lossLabel}: ${rawPoint.x.toFixed(4)}, ${scoreLabel}: ${rawPoint.y.toFixed(4)}`;
                }
                return [labelText, `Params: ${paramsList}`];
            }
        }
        if (isSingleObj) {
            return `${rawPoint.label} | ${metricLabel}: ${rawPoint.y.toFixed(4)}`;
        } else {
            return `${rawPoint.label} | ${lossLabel}: ${rawPoint.x.toFixed(4)}, ${scoreLabel}: ${rawPoint.y.toFixed(4)}`;
        }
    };
    
    if (window.HPOState.charts.pareto.instance) {
        window.HPOState.charts.pareto.instance.data.datasets = datasets;
        if (window.HPOState.charts.pareto.instance.options.scales?.x?.title) {
            window.HPOState.charts.pareto.instance.options.scales.x.title.text = getXText();
        }
        if (window.HPOState.charts.pareto.instance.options.scales?.y?.title) {
            window.HPOState.charts.pareto.instance.options.scales.y.title.text = getYText();
        }
        if (window.HPOState.charts.pareto.instance.options.plugins?.tooltip?.callbacks) {
            window.HPOState.charts.pareto.instance.options.plugins.tooltip.callbacks.label = getTooltipLabel;
        }
        if (totalPoints > window.HPOState.render.lastParetoPointCount) {
            window.HPOState.charts.pareto.instance.update({ duration: 280, easing: "easeOutQuart" });
        } else {
            window.HPOState.charts.pareto.instance.update("none");
        }
    } else {
        window.HPOState.charts.pareto.instance = new Chart(ctx, {
            data: { datasets },
            options: {
                responsive: true,
                animation: false,
                interaction: { mode: "nearest", intersect: true, axis: "xy" },
                onHover: (evt, elements) => {
                    const target = evt?.native?.target;
                    if (target) target.style.cursor = elements.length ? "pointer" : "default";
                },
                transitions: {
                    active: { animation: { duration: 0 } },
                    resize: { animation: { duration: 0 } },
                },
                scales: {
                    x: {
                        type: "linear",
                        position: "bottom",
                        title: { display: true, text: getXText(), color: "#7e8c9f" },
                        grid: { color: "rgba(255, 255, 255, 0.03)" },
                        ticks: { color: "#7e8c9f" }
                    },
                    y: {
                        title: { display: true, text: getYText(), color: "#7e8c9f" },
                        grid: { color: "rgba(255, 255, 255, 0.03)" },
                        ticks: { color: "#7e8c9f" }
                    }
                },
                plugins: {
                    legend: { labels: { color: "#e2e8f0" } },
                    tooltip: { callbacks: { label: getTooltipLabel } }
                }
            }
        });
    }
    window.HPOState.render.lastParetoPointCount = totalPoints;
}

window.updateChart = updateChart;
