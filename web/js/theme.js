function toggleAccentPanel() {
    const panel = document.getElementById("accent-panel");
    const btn = document.getElementById("accent-toggle");
    if (!panel || !btn) return;
    const open = panel.classList.toggle("open");
    btn.setAttribute("aria-expanded", open ? "true" : "false");

    if (open) {
        const expPanel = document.getElementById("export-panel");
        const expBtn = document.getElementById("export-toggle");
        if (expPanel && expPanel.classList.contains("open")) {
            expPanel.classList.remove("open");
            if (expBtn) expBtn.setAttribute("aria-expanded", "false");
        }
    }
}

function toggleExportPanel() {
    const panel = document.getElementById("export-panel");
    const btn = document.getElementById("export-toggle");
    if (!panel || !btn) return;
    const open = panel.classList.toggle("open");
    btn.setAttribute("aria-expanded", open ? "true" : "false");

    if (open) {
        const accPanel = document.getElementById("accent-panel");
        const accBtn = document.getElementById("accent-toggle");
        if (accPanel && accPanel.classList.contains("open")) {
            accPanel.classList.remove("open");
            if (accBtn) accBtn.setAttribute("aria-expanded", "false");
        }
    }
}

function showToast(message) {
    let toast = document.getElementById("app-toast");
    if (!toast) {
        toast = document.createElement("div");
        toast.id = "app-toast";
        document.body.appendChild(toast);
    }
    toast.textContent = message;
    toast.classList.remove("show");
    void toast.offsetWidth; 
    toast.classList.add("show");
    
    if (window.HPOState.ui.toastTimeout) clearTimeout(window.HPOState.ui.toastTimeout);
    window.HPOState.ui.toastTimeout = setTimeout(() => {
        toast.classList.remove("show");
    }, 2500);
}

function changeAccent(accentName, colorHex, glowStyle) {
    window.HPOState.ui.accentColorHex = colorHex;
    document.documentElement.style.setProperty('--accent-color', colorHex);
    document.documentElement.style.setProperty('--accent-glow', glowStyle);
    const preview = document.getElementById("accent-preview");
    if (preview) preview.style.backgroundColor = colorHex;

    document.querySelectorAll(".accent-dot").forEach(dot => {
        dot.classList.remove("active");
        if (dot.getAttribute("data-accent") === accentName) {
            dot.classList.add("active");
        }
    });

    localStorage.setItem("hpo_accent_name", accentName);
    localStorage.setItem("hpo_accent_color", colorHex);
    localStorage.setItem("hpo_accent_glow", glowStyle);

    if (window.HPOState.charts.pareto.instance && window.HPOState.charts.pareto.instance.data && window.HPOState.charts.pareto.instance.data.datasets) {
        window.HPOState.charts.pareto.instance.data.datasets.forEach(ds => {
            if (ds.label === "Active Running") {
                ds.backgroundColor = colorHex;
                ds.borderColor = colorHex;
            }
        });
        window.HPOState.charts.pareto.instance.update("none");
    }
    
    if (window.HPOState.charts.modalHistory.instance && window.HPOState.charts.modalHistory.instance.data && window.HPOState.charts.modalHistory.instance.data.datasets) {
        window.HPOState.charts.modalHistory.instance.data.datasets.forEach(ds => {
            if (ds.yAxisID === "y-score") {
                ds.borderColor = colorHex;
            }
        });
        window.HPOState.charts.modalHistory.instance.update("none");
    }

    const pathwaysCard = document.getElementById("analysis-pathways-card");
    if (pathwaysCard && pathwaysCard.style.display !== "none") {
        const { trials, paretoSet } = getDisplayTrialsForCurrentStudy();
        updateParallelCoordinates(trials, paretoSet);
    }
}

function setStatusStyle(style) {
    if (style === 'full') {
        document.body.classList.add('use-full-cell-states');
    } else {
        document.body.classList.remove('use-full-cell-states');
    }

    localStorage.setItem("hpo_status_style", style);

    document.querySelectorAll(".style-segment-btn").forEach(btn => {
        btn.classList.remove("active");
    });
    const activeBtn = document.getElementById(`btn-status-${style}`);
    if (activeBtn) activeBtn.classList.add("active");
}

window.toggleAccentPanel = toggleAccentPanel;
window.toggleExportPanel = toggleExportPanel;
window.showToast = showToast;
window.changeAccent = changeAccent;
window.setStatusStyle = setStatusStyle;
