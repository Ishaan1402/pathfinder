function toggleSidebar() {
    const sidebar = document.getElementById("sidebar");
    const icon = document.getElementById("collapse-icon");
    sidebar.classList.toggle("collapsed");
    if (sidebar.classList.contains("collapsed")) {
        icon.innerHTML = `<polyline points="9 18 15 12 9 6"></polyline>`;
    } else {
        icon.innerHTML = `<polyline points="15 18 9 12 15 6"></polyline>`;
    }
}

function switchView(viewId, el, updateHash = true) {
    document.querySelectorAll(".nav-item").forEach(item => item.classList.remove("active"));
    if (el) el.classList.add("active");

    document.querySelectorAll(".content-view").forEach(view => view.classList.remove("active"));
    const targetView = document.getElementById(`${viewId}-view`);
    if (targetView) {
        targetView.classList.add("active");
        targetView.querySelectorAll("table.trial-table").forEach(table => {
            if (window.HPOState.data.trials && window.HPOState.data.trials.length > 0) {
                makeTableResizable(table, true);
            }
        });
    }

    const titles = {
        "dashboard": "Dashboard",
        "analysis": "Study Analysis",
        "search-space": "Search Space & Evaluation",
        "worker-setup": "GPU Training Worker Setup"
    };
    const titleEl = document.getElementById("view-title");
    if (titleEl) titleEl.innerText = titles[viewId] || "Dashboard";

    if (viewId === 'worker-setup') {
        if (typeof window.updateColabSnippet === 'function') {
            window.updateColabSnippet();
        }
    } else if (viewId === 'analysis') {
        const { trials, paretoSet } = getDisplayTrialsForCurrentStudy();
        if (typeof window.updateParallelCoordinates === 'function') {
            window.updateParallelCoordinates(trials, paretoSet);
        }
        if (typeof window.updateAshaTimeline === 'function') {
            window.updateAshaTimeline(trials);
        }
    } else if (viewId === 'dashboard' && window.HPOState.data.latestStudyData) {
        const { trials, paretoSet } = getDisplayTrialsForCurrentStudy();
        if (typeof window.updateChart === 'function') {
            window.updateChart(trials, paretoSet, window.HPOState.data.latestStudyData.study_directions);
        }
    }
    
    if (updateHash) {
        if (viewId === 'analysis') {
            const currentHash = window.location.hash;
            if (!currentHash.startsWith('#analysis-')) {
                window.location.hash = 'analysis-table';
            }
        } else {
            window.location.hash = viewId;
        }
    }
}

function switchAnalysisSubTab(subViewId, updateHash = true) {
    document.querySelectorAll("#analysis-view .tab-btn").forEach(btn => btn.classList.remove("active"));
    const activeBtn = document.getElementById(`analysis-tab-btn-${subViewId}`);
    if (activeBtn) activeBtn.classList.add("active");

    const tableCard = document.getElementById("analysis-table-card");
    const pathwaysCard = document.getElementById("analysis-pathways-card");
    const timelineCard = document.getElementById("analysis-timeline-card");

    if (tableCard) tableCard.style.display = subViewId === "table" ? "flex" : "none";
    if (pathwaysCard) pathwaysCard.style.display = subViewId === "pathways" ? "flex" : "none";
    if (timelineCard) timelineCard.style.display = subViewId === "timeline" ? "flex" : "none";

    const { trials, paretoSet } = getDisplayTrialsForCurrentStudy();
    if (subViewId === "pathways" && pathwaysCard && pathwaysCard.style.display !== "none") {
        if (typeof window.updateParallelCoordinates === 'function') {
            window.updateParallelCoordinates(trials, paretoSet);
        }
    }
    if (subViewId === "timeline" && timelineCard && timelineCard.style.display !== "none") {
        if (typeof window.updateAshaTimeline === 'function') {
            window.updateAshaTimeline(trials);
        }
    }

    if (updateHash) {
        window.location.hash = `analysis-${subViewId}`;
    }
}

function switchWorkerSetupSubTab(subViewId, updateHash = true) {
    document.querySelectorAll("#worker-setup-view .tab-btn").forEach(btn => btn.classList.remove("active"));
    const activeBtn = document.getElementById(`worker-tab-btn-${subViewId}`);
    if (activeBtn) activeBtn.classList.add("active");

    const customCard = document.getElementById("worker-custom-card");
    const colabCard = document.getElementById("worker-colab-card");
    const ideCard = document.getElementById("worker-ide-card");

    if (customCard) customCard.style.display = subViewId === "custom" ? "block" : "none";
    if (colabCard) colabCard.style.display = subViewId === "colab" ? "block" : "none";
    if (ideCard) ideCard.style.display = subViewId === "ide" ? "block" : "none";

    const setupNote = document.getElementById("worker-setup-note");
    if (setupNote) {
        setupNote.classList.toggle("hidden-tab", subViewId === "ide");
    }

    if (updateHash) {
        window.location.hash = `worker-setup-${subViewId}`;
    }
}

function handleRouting() {
    const hash = window.location.hash || "#dashboard";
    const route = hash.substring(1);
    
    if (route.startsWith("analysis-")) {
        const subView = route.replace("analysis-", "");
        const navEl = document.getElementById("nav-analysis");
        switchView("analysis", navEl, false);
        switchAnalysisSubTab(subView, false);
    } else if (route === "analysis") {
        window.location.hash = "analysis-table";
    } else if (route.startsWith("worker-setup-")) {
        const subView = route.replace("worker-setup-", "");
        const navEl = document.getElementById("nav-worker-setup");
        switchView("worker-setup", navEl, false);
        switchWorkerSetupSubTab(subView, false);
    } else if (route === "worker-setup") {
        window.location.hash = "worker-setup-custom";
    } else {
        const navEl = document.getElementById(`nav-${route}`);
        if (navEl) {
            switchView(route, navEl, false);
        } else {
            window.location.hash = "dashboard";
        }
    }
}

window.toggleSidebar = toggleSidebar;
window.switchView = switchView;
window.switchAnalysisSubTab = switchAnalysisSubTab;
window.switchWorkerSetupSubTab = switchWorkerSetupSubTab;
window.handleRouting = handleRouting;
