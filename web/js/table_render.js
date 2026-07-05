function buildStatusTd(state, latestEpoch) {
    let ledClass = "complete";
    let label = "DONE";
    
    if (state === "COMPLETE") {
        ledClass = "complete";
        label = "DONE";
    } else if (state === "FAIL") {
        ledClass = "fail";
        label = "FAILED";
    } else if (state === "PRUNED") {
        ledClass = "pruned";
        label = "PRUNED";
    } else if (state === "RUNNING") {
        ledClass = "running";
        const epochPart = (latestEpoch !== null && latestEpoch !== undefined) ? ` E${latestEpoch}` : "";
        label = `RUNNING${epochPart}`;
    } else {
        return `<td><span style="font-family: var(--font-mono); font-size: 0.72rem; color: var(--text-muted);">${state || 'UNKNOWN'}</span></td>`;
    }
    
    return `<td class="col-state status-cell-${ledClass}" style="padding: 0;">
        <div class="status-cell-wrapper">
            <span class="led-light ${ledClass}"></span>
            <span class="status-text">${label}</span>
        </div>
    </td>`;
}

function buildTrialsSnapshot(trials, paretoSet) {
    const base = trials
        .map(
            (t) =>
                `${t.number}|${t.state}|${t.loss ?? ""}|${t.score ?? ""}|${paretoSet.has(t.number) ? 1 : 0}`
        )
        .sort((a, b) => Number(a.split("|")[0]) - Number(b.split("|")[0]))
        .join(";");
    return `${base}|top15:${window.HPOState.ui.filterTopPerformersOnly ? 1 : 0}`;
}

function buildFanovaSnapshot(trials) {
    return trials
        .filter((t) => t.state === "COMPLETE")
        .map((t) => `${t.number}|${t.loss}|${t.score}`)
        .join(";");
}

function renderDashboardTableBody(displayTrials, paretoSet, data) {
    const tbody = document.getElementById("trials-table-body");
    if (!tbody) return;
    const ev = data.hpo_config?.eval_protocol || {};
    const paramKeys = getParamKeys();

    const piped = applyTablePipeline(displayTrials, "dashboard");
    if (!piped.length) {
        const totalCols = 4 + (ev.enabled ? 1 : 0) + paramKeys.length + 2;
        tbody.innerHTML = `<tr><td colspan="${totalCols}" style="text-align: center; color: var(--text-muted); padding: 40px;">No trials match filters.</td></tr>`;
        return;
    }

    let rowsHtml = "";
    piped.forEach((t) => {
        const isPareto = paretoSet.has(t.number);
        const lossStr = formatMetric(t.loss);
        const scoreStr = formatMetric(t.score);
        let cellsHtml = "";
        paramKeys.forEach((k) => {
            const val = t.params[k];
            let valStr = "-";
            if (val !== undefined && val !== null) {
                if (typeof val === "number") {
                    valStr = val < 0.01 && val > 0 ? val.toExponential(2) : (val % 1 !== 0 ? val.toFixed(4) : val.toString());
                } else {
                    valStr = val.toString();
                }
            }
            cellsHtml += `<td class="td-mono">${valStr}</td>`;
        });
        let trClass = t.state === "RUNNING" ? "running" : (isPareto ? "pareto" : (t.state === "PRUNED" ? "pruned" : (t.state === "FAIL" ? "failed" : "")));
        const statusTdHtml = buildStatusTd(t.state, t.latest_epoch);
        const fixedStr = formatMetric(t.score_eval_fixed);
        let rowHtml = `<tr class="${trClass}" onclick="openTrialDetails(${t.number})">
            <td class="td-mono" style="font-weight:600;">#${t.number}</td>
            ${statusTdHtml}
            <td class="td-mono">${lossStr}</td>
            <td class="td-mono">${scoreStr}</td>`;
        if (ev.enabled) rowHtml += `<td class="td-mono">${fixedStr}</td>`;
        rowHtml += `${cellsHtml}
            <td class="td-mono"><strong style="color:var(--status-complete);">${isPareto ? "★" : ""}</strong></td>
            <td style="text-align: right; white-space: nowrap;">
                <button type="button" class="row-action-btn" onclick="event.stopPropagation(); copyTrialConfigToClipboard(${t.number}, 'json')" title="Copy parameters as JSON">JSON</button>
                <button type="button" class="row-action-btn" onclick="event.stopPropagation(); copyTrialConfigToClipboard(${t.number}, 'cli')" title="Copy parameters as CLI flags">CLI</button>
            </td></tr>`;
        rowsHtml += rowHtml;
    });
    tbody.innerHTML = rowsHtml;
    syncTrialColumnWidth(document.getElementById("dashboard-trial-table"), piped);
}

function getParamKeys() {
    const activeSpace = window.HPOState.data.activeSearchSpace || {};
    let keys = Object.keys(activeSpace).filter(k => !k.startsWith("_") && typeof activeSpace[k] === "object");
    if (keys.length === 0 && window.HPOState.data.trials && window.HPOState.data.trials.length > 0) {
        const keysSet = new Set();
        window.HPOState.data.trials.forEach(t => {
            if (t.params) {
                Object.keys(t.params).forEach(k => keysSet.add(k));
            }
        });
        keys = Array.from(keysSet);
    }
    keys.sort();
    return keys;
}

function renderAnalysisTableBody(displayTrials) {
    const tbody = document.getElementById("analysis-trials-body");
    if (!tbody) return;
    applyAnalysisTableHeaders();
    const ev = window.HPOState.data.hpoConfig?.eval_protocol || {};
    const paramKeys = getParamKeys();
    const piped = applyTablePipeline(displayTrials, "analysis");
    if (!piped.length) {
        const totalCols = 4 + (ev.enabled ? 1 : 0) + paramKeys.length + 1;
        tbody.innerHTML = `<tr><td colspan="${totalCols}" style="text-align:center;color:var(--text-muted);padding:32px;">No trials match filters.</td></tr>`;
        return;
    }
    let html = "";
    piped.forEach((t) => {
        const lossStr = formatMetric(t.loss, "—");
        const scoreStr = formatMetric(t.score, "—");
        const fixedStr = formatMetric(t.score_eval_fixed, "—");
        let cellsHtml = "";
        paramKeys.forEach((k) => {
            const val = t.params[k];
            let valStr = "-";
            if (val !== undefined && val !== null) {
                if (typeof val === "number") {
                    valStr = val < 0.01 && val > 0 ? val.toExponential(2) : (val % 1 !== 0 ? val.toFixed(4) : val.toString());
                } else {
                    valStr = val.toString();
                }
            }
            cellsHtml += `<td class="td-mono">${valStr}</td>`;
        });
        const statusTdHtml = buildStatusTd(t.state, t.latest_epoch);
        html += `<tr onclick="openTrialDetails(${t.number})">
            <td class="td-mono" style="font-weight:600;">#${t.number}</td>
            ${statusTdHtml}
            <td class="td-mono">${lossStr}</td>
            <td class="td-mono">${scoreStr}</td>
            ${ev.enabled ? `<td class="td-mono">${fixedStr}</td>` : ""}
            ${cellsHtml}
            <td style="text-align: right; white-space: nowrap;">
                <button type="button" class="row-action-btn" onclick="event.stopPropagation(); copyTrialConfigToClipboard(${t.number}, 'json')" title="Copy parameters as JSON">JSON</button>
                <button type="button" class="row-action-btn" onclick="event.stopPropagation(); copyTrialConfigToClipboard(${t.number}, 'cli')" title="Copy parameters as CLI flags">CLI</button>
            </td>
        </tr>`;
    });
    tbody.innerHTML = html;
    syncTrialColumnWidth(document.getElementById("analysis-trial-table"), piped);
}

function renderAnalysisTrialsTable(trials) {
    renderAnalysisTableBody(trials || []);
}

function applyAnalysisTableHeaders() {
    const ev = window.HPOState.data.hpoConfig?.eval_protocol || {};
    const paramLabels = window.HPOState.data.hpoConfig?.param_labels || {};
    const paramKeys = getParamKeys();

    const lossLabel = window.HPOState.data.hpoConfig?.metric_loss_label || "Loss";
    const scoreLabel = window.HPOState.data.hpoConfig?.metric_score_label || "Score";
    const evalLabel = ev.score_fixed_label || "Score (eval)";

    const headerSnapshot = `${lossLabel}|${scoreLabel}|${evalLabel}|${ev.enabled}|${paramKeys.join(",")}`;
    if (headerSnapshot === window.HPOState.render.lastAnalysisHeaderSnapshot) {
        return;
    }
    window.HPOState.render.lastAnalysisHeaderSnapshot = headerSnapshot;

    const tableHeader = document.querySelector("#analysis-trial-table thead tr");
    if (tableHeader) {
        let thHtml = buildSortableTh("Trial", "number", "analysis");
        thHtml += buildSortableTh("State", "state", "analysis");
        thHtml += buildSortableTh(lossLabel, "loss", "analysis");
        thHtml += buildSortableTh(scoreLabel, "score", "analysis");
        if (ev.enabled) {
            thHtml += buildSortableTh(evalLabel, "score_eval_fixed", "analysis");
        }
        paramKeys.forEach(k => {
            const label = paramLabels[k] || k.replace(/_/g, " ");
            thHtml += buildSortableTh(label, `param:${k}`, "analysis");
        });
        thHtml += `<th data-col-label="Actions" style="text-align: right;">Actions</th>`;
        tableHeader.innerHTML = thHtml;
        const table = tableHeader.closest("table");
        const baseTrials = window.HPOState.data.trials || [];
        syncTrialColumnWidth(table, baseTrials);
        makeTableResizable(table);
        bindTableHeaderInteractions(table, "analysis", baseTrials);
    }
}

function renderStudyDetails(data) {
    if (!data) return;

    if (data.hpo_config) applyAnalysisTableHeaders();
    applyEvalInsightsUi();

    const paramLabels = data.hpo_config?.param_labels || {};
    const paramKeys = getParamKeys();
    const ev = data.hpo_config?.eval_protocol || {};

    // Render live monitor table headers dynamically
    const tableHeader = document.querySelector("#dashboard-trial-table thead tr");
    if (tableHeader) {
        const lossLabel = data.hpo_config?.metric_loss_label || "Loss";
        const scoreLabel = data.hpo_config?.metric_score_label || "Score";
        const evalLabel = ev.score_fixed_label || "Score (eval)";

        const headerSnapshot = `${lossLabel}|${scoreLabel}|${evalLabel}|${ev.enabled}|${paramKeys.join(",")}`;
        if (headerSnapshot !== window.HPOState.render.lastDashboardHeaderSnapshot) {
            window.HPOState.render.lastDashboardHeaderSnapshot = headerSnapshot;

            let thHtml = buildSortableTh("Trial", "number", "dashboard");
            thHtml += buildSortableTh("State", "state", "dashboard");
            thHtml += buildSortableTh(lossLabel, "loss", "dashboard");
            thHtml += buildSortableTh(scoreLabel, "score", "dashboard");
            if (ev.enabled) {
                thHtml += buildSortableTh(evalLabel, "score_eval_fixed", "dashboard");
            }
            paramKeys.forEach(k => {
                const label = paramLabels[k] || k.replace(/_/g, " ");
                thHtml += buildSortableTh(label, `param:${k}`, "dashboard");
            });
            thHtml += `<th class="col-pareto" data-col-label="Pareto">★</th>`;
            thHtml += `<th data-col-label="Actions" style="text-align: right;">Actions</th>`;
            tableHeader.innerHTML = thHtml;
            const table = tableHeader.closest("table");
            syncTrialColumnWidth(table, data.trials || []);
            makeTableResizable(table);
            bindTableHeaderInteractions(table, "dashboard", data.trials || []);
        }
    }

    const tbody = document.getElementById("trials-table-body");
    const trialsCard = document.getElementById("dashboard-trial-monitor-card");
    const startedCard = document.getElementById("dashboard-getting-started-card");
    const analysisEmpty = document.getElementById("analysis-empty-state");
    const analysisLayout = document.getElementById("analysis-main-layout");

    if (!data.trials || data.trials.length === 0) {
        if (trialsCard) trialsCard.style.display = "none";
        if (startedCard) startedCard.style.display = "block";
        if (analysisEmpty) {
            analysisEmpty.style.display = "flex";
            analysisEmpty.style.alignItems = "center";
            analysisEmpty.style.justifyContent = "center";
        }
        if (analysisLayout) analysisLayout.style.display = "none";

        window.showEmptyState("pareto-chart", "Run at least 2 completed trials to see Pareto front");
        window.showEmptyState("fanova-container", "Need at least 2 completed trials for importance analysis");

        renderAnalysisTrialsTable([]);
        window.HPOState.render.lastTrialsSnapshot = "empty";
        window.HPOState.render.lastFanovaPayload = "empty";
        window.HPOState.render.lastFanovaRenderKey = null;
        return;
    }

    if (trialsCard) trialsCard.style.display = "block";
    if (startedCard) startedCard.style.display = "none";
    if (analysisEmpty) analysisEmpty.style.display = "none";
    if (analysisLayout) analysisLayout.style.display = "flex";

    const paretoSet = new Set(data.pareto_trials);
    const displayTrials = filterTrialsForDisplay(data.trials, paretoSet);

    // Rebuild when trial data or Top-15% filter state changes.
    const snapshot = buildTrialsSnapshot(data.trials, paretoSet);
    if (snapshot === window.HPOState.render.lastTrialsSnapshot) return;
    window.HPOState.render.lastTrialsSnapshot = snapshot;

    renderAnalysisTrialsTable(displayTrials);
    renderDashboardTableBody(displayTrials, paretoSet, data);

    const dashboardView = document.getElementById("dashboard-view");
    const isDashboardActive = dashboardView && dashboardView.classList.contains("active");

    if (isDashboardActive) {
        updateChart(displayTrials, paretoSet, data.study_directions);
    }

    // Parallel Coordinates and ASHA timelines only need update/draw if Analysis view is active
    const analysisView = document.getElementById("analysis-view");
    const isAnalysisActive = analysisView && analysisView.classList.contains("active");
    if (isAnalysisActive) {
        updateParallelCoordinates(displayTrials, paretoSet);
        updateAshaTimeline(displayTrials);
    }

    const fanovaSnap = buildFanovaSnapshot(data.trials);
    if (fanovaSnap !== window.HPOState.render.lastFanovaPayload) {
        window.HPOState.render.lastFanovaPayload = fanovaSnap;
        fetchFanova();
    }
}

function applyEvalInsightsUi() {
    const card = document.getElementById("analysis-resolution-card");
    const grid = document.getElementById("analysis-resolution-grid");
    const deployLabel = document.getElementById("analysis-best-deploy-label");

    if (!card || !grid) return;

    // Restore collapsed state
    const isCollapsed = localStorage.getItem("resolution_card_collapsed") === "true";
    if (isCollapsed) {
        card.classList.add("collapsed");
        const chevron = document.getElementById("resolution-collapse-chevron");
        if (chevron) chevron.style.transform = "rotate(180deg)";
    } else {
        card.classList.remove("collapsed");
        const chevron = document.getElementById("resolution-collapse-chevron");
        if (chevron) chevron.style.transform = "rotate(0deg)";
    }

    // Get train resolution/fidelity param from config
    const ev = window.HPOState.data.hpoConfig?.eval_protocol || {};
    const trainParam = ev.train_resolution_param || "resolution";
    const trainParamLabel = window.paramLabel(trainParam);

    // Update card header text dynamically (e.g., "Resolution breakdown" -> "Fidelity breakdown")
    const headerTitle = card.querySelector(".card-header span:first-child");
    if (headerTitle) {
        const capitalizedLabel = trainParamLabel.charAt(0).toUpperCase() + trainParamLabel.slice(1);
        headerTitle.textContent = `${capitalizedLabel} breakdown`;
    }

    const summary = window.HPOState.data.evalInsights?.resolution_summary || {};
    const keys = Object.keys(summary);
    
    // Show only if evaluation protocol is enabled and we have trials grouped by this parameter
    if (ev.enabled && keys.length) {
        card.style.display = "";
        grid.innerHTML = keys
            .map((res) => {
                const s = summary[res];
                const trainBestVal = s.best_score_train;
                const trainBest = trainBestVal != null ? trainBestVal.toFixed(4) : "—";
                
                const fixedBestVal = s.best_score_fixed;
                const fixedBest = fixedBestVal != null ? fixedBestVal.toFixed(4) : "—";
                
                // Capitalize parameter label for individual chips
                return `<div class="res-summary-chip">${trainParamLabel}: ${res} · n=${s.count} · train ${trainBest} · fixed ${fixedBest}</div>`;
            })
            .join("");
    } else {
        card.style.display = "none";
        grid.innerHTML = "";
    }

    const scoreLabel = window.HPOState.data.hpoConfig?.metric_score_label || "Score";
    if (deployLabel && window.HPOState.data.evalInsights?.best_deploy_trial_number != null) {
        const d = window.HPOState.data.evalInsights.best_deploy_score_fixed;
        deployLabel.textContent = `Best fixed-eval: Trial #${window.HPOState.data.evalInsights.best_deploy_trial_number}${d != null ? ` · ${scoreLabel} ${d.toFixed(4)}` : ""}`;
    } else if (deployLabel) {
        deployLabel.textContent = "";
    }
}

function toggleResolutionCard() {
    const card = document.getElementById("analysis-resolution-card");
    const chevron = document.getElementById("resolution-collapse-chevron");
    if (!card) return;
    card.classList.toggle("collapsed");
    
    const isCollapsed = card.classList.contains("collapsed");
    if (chevron) {
        chevron.style.transform = isCollapsed ? "rotate(180deg)" : "rotate(0deg)";
    }
    
    localStorage.setItem("resolution_card_collapsed", isCollapsed ? "true" : "false");
}

window.buildStatusTd = buildStatusTd;
window.buildTrialsSnapshot = buildTrialsSnapshot;
window.buildFanovaSnapshot = buildFanovaSnapshot;
window.renderDashboardTableBody = renderDashboardTableBody;
window.renderAnalysisTableBody = renderAnalysisTableBody;
window.renderAnalysisTrialsTable = renderAnalysisTrialsTable;
window.applyAnalysisTableHeaders = applyAnalysisTableHeaders;
window.renderStudyDetails = renderStudyDetails;
window.applyEvalInsightsUi = applyEvalInsightsUi;
window.toggleResolutionCard = toggleResolutionCard;
