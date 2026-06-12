const INTERNAL_STUDY_RE = /^(test_)|_test$/i;
function isInternalStudy(name) {
    return INTERNAL_STUDY_RE.test(name || "");
}

async function populateStudyList() {
    try {
        const res = await fetch("/api/studies");
        if (!res.ok) return;
        const data = await res.json();
        if (data.success && data.studies) {
            const select = document.getElementById("study-select");
            if (select) {
                select.innerHTML = "";
                const activeStudy = window.HPOState.session.studyName;
                const userStudies = data.studies.filter((name) => !isInternalStudy(name));
                const studiesToShow = userStudies.slice().sort();
                if (isInternalStudy(activeStudy) && !studiesToShow.includes(activeStudy)) {
                    studiesToShow.unshift(activeStudy);
                }
                studiesToShow.forEach((study) => {
                    const opt = document.createElement("option");
                    opt.value = study;
                    opt.textContent = study;
                    if (study === activeStudy) opt.selected = true;
                    select.appendChild(opt);
                });
            }
        }
    } catch (err) {
        console.error("Error fetching studies:", err);
    }
}

async function fetchStudyDetails(throwOnError = false) {
    try {
        const res = await fetch(`/api/study_details?study_name=${window.HPOState.session.studyName}`);
        if (!res.ok) {
            if (throwOnError) throw new Error("HTTP error " + res.status);
            return;
        }
        const data = await res.json();
        window.HPOState.data.latestStudyData = data;

        const completedCount = data.trials.filter(t => t.state === "COMPLETE" || t.state === "FAIL" || t.state === "PRUNED").length;
        if (window.HPOState.telemetry.lastCompletedCount !== null && completedCount !== window.HPOState.telemetry.lastCompletedCount) {
            window.HPOState.ui.reviewPillDismissed = false;
        }
        window.HPOState.telemetry.lastCompletedCount = completedCount;

        const select = document.getElementById("study-select");
        if (select) select.value = data.study_name;
        document.getElementById("workers-count-val").innerText = data.running_workers;

        window.HPOState.data.trials = data.trials;
        if (data.hpo_config) window.HPOState.data.hpoConfig = data.hpo_config;
        window.HPOState.data.evalInsights = data.eval_insights || null;
        window.HPOState.data.review = data.review || null;
        const nComplete = data.completed_count ?? data.trials.filter((t) => t.state === "COMPLETE").length;
        updateStatConfidenceBanner(data.statistical_confidence || "low", nComplete);
        renderPastReviews(data.past_reviews || []);
        applyReviewUi();
        updateAnalysisStatusTicker();

        syncTelemetryFromTrials([...data.trials].sort((a, b) => b.number - a.number));

        if (window.HPOState.ui.isModalOpen) {
            window.HPOState.render.pendingRender = true;
            return;
        }
        renderStudyDetails(data);
    } catch (err) {
        console.error("Error fetching study details:", err);
        if (throwOnError) throw err;
    }
}

async function fetchHpoConfig() {
    const studyName = window.HPOState.session.studyName || '';
    try {
        const res = await fetch(`/api/hpo_config?study_name=${encodeURIComponent(studyName)}`);
        if (!res.ok) return;
        window.HPOState.data.hpoConfig = await res.json();
        populateEvalProtocolForm(window.HPOState.data.hpoConfig);
        applyAnalysisTableHeaders();
    } catch (err) {
        console.error("Error fetching hpo_config:", err);
    }
}

async function fetchSearchSpace() {
    const studyName = window.HPOState.session.studyName || '';
    try {
        const res = await fetch(`/api/search_space?study_name=${encodeURIComponent(studyName)}`);
        if (!res.ok) return;
        window.HPOState.data.activeSearchSpace = await res.json();
        renderSearchSpace();
        renderDashboardSearchSpaceSummary();
        fetchPendingChanges();
        if (window.HPOState.data.latestStudyData) {
            renderStudyDetails(window.HPOState.data.latestStudyData);
        }
    } catch (err) {
        console.error("Error fetching search space:", err);
    }
}

async function fetchFanova() {
    try {
        const res = await fetch(`/api/fanova?study_name=${window.HPOState.session.studyName}`);
        if (!res.ok) return;
        const data = await res.json();
        const container = document.getElementById("fanova-container");

        if (!data.success) {
            const msg = data.message || "Unable to compute fANOVA.";
            if (window.HPOState.render.lastFanovaRenderKey === msg) return;
            window.HPOState.render.lastFanovaRenderKey = msg;
            container.innerHTML = `<div class="fanova-empty">${msg}</div>`;
            return;
        }

        const sorted = Object.entries(data.importances).sort((a, b) => b[1] - a[1]);
        const renderKey = sorted.map(([k, v]) => `${k}:${v.toFixed(6)}`).join("|");
        if (renderKey === window.HPOState.render.lastFanovaRenderKey) return;
        window.HPOState.render.lastFanovaRenderKey = renderKey;

        let html = "";
        sorted.forEach(([param, value]) => {
            const percentage = (value * 100).toFixed(1);
            const label = fanovaParamLabel(param);
            html += `<div class="fanova-row">
                <div class="fanova-label">
                    <span style="font-weight:500;">${label}</span>
                    <strong style="color: var(--accent-color);">${percentage}%</strong>
                </div>
                <div class="bar-container">
                    <div class="bar-fill" style="width: ${percentage}%"></div>
                </div>
            </div>`;
        });
        container.innerHTML = html;
    } catch (err) {
        console.error("Error fetching fANOVA:", err);
    }
}

window.populateStudyList = populateStudyList;
window.fetchStudyDetails = fetchStudyDetails;
window.fetchHpoConfig = fetchHpoConfig;
window.fetchSearchSpace = fetchSearchSpace;
window.fetchFanova = fetchFanova;
