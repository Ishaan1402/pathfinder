async function fetchPendingChanges(throwOnError = false) {
    try {
        const res = await fetch(`/api/pending_changes?study_name=${window.HPOState.session.studyName}`);
        if (!res.ok) { if (throwOnError) throw new Error("HTTP " + res.status); return; }
        const data = await res.json();
        const banner = document.getElementById("pending-changes-banner");
        const diffContainer = document.getElementById("pending-changes-diff");
        if (!banner || !diffContainer) return;
        if (data && data.proposed_changes) {
            window.HPOState.data.pendingChanges = data.proposed_changes;
            const current = window.HPOState.data.activeSearchSpace || {};
            diffContainer.replaceChildren();
            for (const [key, val] of Object.entries(data.proposed_changes)) {
                const curVal = current[key];
                if (!curVal) continue;
                const infoLine = document.createElement("span");
                infoLine.className = "diff-line info";
                infoLine.textContent = `# Parameter: ${paramLabel(key)} (${key})`;
                diffContainer.appendChild(infoLine);
                if (curVal.type === "categorical") {
                    const curActive = curVal.active || [], newActive = val.active || [];
                    diffContainer.appendChild(Object.assign(document.createElement("span"), { className: "diff-line removed", textContent: `- active: [${curActive.join(", ")}]` }));
                    diffContainer.appendChild(Object.assign(document.createElement("span"), { className: "diff-line added", textContent: `+ active: [${newActive.join(", ")}]` }));
                } else {
                    const curMin = curVal.min, curMax = curVal.max;
                    const newMin = val.min !== undefined ? val.min : curMin;
                    const newMax = val.max !== undefined ? val.max : curMax;
                    if (curMin !== newMin || curMax !== newMax) {
                        diffContainer.appendChild(Object.assign(document.createElement("span"), { className: "diff-line removed", textContent: `- range: [${curMin}, ${curMax}]` }));
                        diffContainer.appendChild(Object.assign(document.createElement("span"), { className: "diff-line added", textContent: `+ range: [${newMin}, ${newMax}]` }));
                    }
                }
            }
            banner.classList.toggle("hidden", diffContainer.childNodes.length === 0);
        } else {
            window.HPOState.data.pendingChanges = null;
            banner.classList.add("hidden");
        }
        renderHealthLatestAction();
    } catch (err) {
        console.error("Error fetching pending changes:", err);
        if (throwOnError) throw err;
    }
}

function populateEvalProtocolForm(config) {
    const ev = config?.eval_protocol || {};
    const set = (id, val) => { const el = document.getElementById(id); if (el) el.value = val ?? ""; };
    const setCheck = (id, val) => { const el = document.getElementById(id); if (el) el.checked = !!val; };
    set("eval-metric-loss-label", config?.metric_loss_label || "Loss");
    set("eval-metric-score-label", config?.metric_score_label || "Score");
    setCheck("desktop-notifs-enabled", !!config?.desktop_notifications_enabled);
    setCheck("eval-enabled", ev.enabled);
    set("eval-fixed-resolution", ev.fixed_resolution ?? "");
    set("eval-train-res-param", ev.train_resolution_param || "resolution");
    set("eval-low-warn", ev.low_train_res_warning ?? "");
    set("eval-score-train-label", ev.dice_train_label || "Score (train)");
    set("eval-score-fixed-label", ev.dice_fixed_label || "Score (eval)");
    setCheck("eval-prune-on-fixed", ev.use_fixed_metric_for_pruning);
    const scoreVal = config?.metric_score_label || "Score";
    const trainInput = document.getElementById("eval-score-train-label");
    const fixedInput = document.getElementById("eval-score-fixed-label");
    if (trainInput) trainInput.placeholder = `${scoreVal} (train)`;
    if (fixedInput) fixedInput.placeholder = `${scoreVal} (eval)`;
}

async function saveEvalProtocol() {
    const fixedRaw = document.getElementById("eval-fixed-resolution")?.value;
    const lowRaw = document.getElementById("eval-low-warn")?.value;
    const payload = {
        ...window.HPOState.data.hpoConfig,
        metric_loss_label: document.getElementById("eval-metric-loss-label")?.value || "Loss",
        metric_score_label: document.getElementById("eval-metric-score-label")?.value || "Score",
        eval_protocol: {
            enabled: document.getElementById("eval-enabled")?.checked || false,
            fixed_resolution: fixedRaw === "" || fixedRaw == null ? null : Number(fixedRaw),
            train_resolution_param: document.getElementById("eval-train-res-param")?.value || "resolution",
            low_train_res_warning: lowRaw === "" || lowRaw == null ? null : Number(lowRaw),
            dice_train_label: document.getElementById("eval-score-train-label")?.value || "Score (train)",
            dice_fixed_label: document.getElementById("eval-score-fixed-label")?.value || "Score (eval)",
            use_fixed_metric_for_pruning: document.getElementById("eval-prune-on-fixed")?.checked || false,
        },
    };
    try {
        const studyName = window.HPOState.session.studyName || '';
        const res = await fetch(`/api/hpo_config?study_name=${encodeURIComponent(studyName)}`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });
        if (!res.ok) { alert("Failed to save eval protocol"); return; }
        const data = await res.json();
        window.HPOState.data.hpoConfig = data.config || data;
        populateEvalProtocolForm(window.HPOState.data.hpoConfig);
        applyAnalysisTableHeaders();
        applyEvalInsightsUi();
        renderAnalysisTrialsTable(getDisplayTrialsForCurrentStudy().trials);
        fetchStudyDetails();
        alert("Eval protocol saved.");
    } catch (err) { alert("Error saving eval protocol: " + err); }
}

async function saveIdeSettings() {
    const payload = {
        ...window.HPOState.data.hpoConfig,
        desktop_notifications_enabled: !!document.getElementById("desktop-notifs-enabled")?.checked,
    };
    try {
        const studyName = window.HPOState.session.studyName || '';
        const res = await fetch(`/api/hpo_config?study_name=${encodeURIComponent(studyName)}`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });
        if (!res.ok) { alert("Failed to save IDE settings"); return; }
        const data = await res.json();
        window.HPOState.data.hpoConfig = data.config || data;
        populateEvalProtocolForm(window.HPOState.data.hpoConfig);
        alert("IDE integration settings saved successfully!");
    } catch (err) { alert("Error saving IDE settings: " + err); }
}

async function applyPendingChanges() {
    try {
        const res = await fetch(`/api/apply_pending_changes?study_name=${window.HPOState.session.studyName}`, { method: "POST" });
        if (!res.ok) { const errData = await res.json(); alert("Error: " + (errData.detail || "Failed to apply changes")); return; }
        const data = await res.json();
        if (data.success) {
            window.HPOState.data.activeSearchSpace = data.space;
            renderSearchSpace();
            renderDashboardSearchSpaceSummary();
            fetchPendingChanges();
        }
    } catch (err) { console.error("Error applying pending changes:", err); }
}

async function discardPendingChanges() {
    try {
        const res = await fetch(`/api/discard_pending_changes?study_name=${window.HPOState.session.studyName}`, { method: "POST" });
        if (!res.ok) { alert("Failed to discard changes"); return; }
        const data = await res.json();
        if (data.success) { fetchPendingChanges(); }
    } catch (err) { console.error("Error discarding pending changes:", err); }
}

function togglePill(param, option, btn) {
    const list = window.HPOState.data.activeSearchSpace[param].active;
    const val = isNaN(option) ? option : Number(option);
    const index = list.indexOf(val);
    if (index > -1) {
        if (list.length > 1) { list.splice(index, 1); btn.classList.remove("active"); }
    } else {
        list.push(val);
        btn.classList.add("active");
    }
    markParamModified(param);
}

async function applySingleParamConstraints(param) {
    const val = window.HPOState.data.activeSearchSpace[param];
    if (val.type === "float" || val.type === "float_log" || val.type === "int") {
        const minEl = document.getElementById(`${param}-min`), maxEl = document.getElementById(`${param}-max`);
        if (minEl && maxEl) { val.min = Number(minEl.value); val.max = Number(maxEl.value); }
    }
    try {
        const res = await fetch(`/api/update_search_space?study_name=${window.HPOState.session.studyName}`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(window.HPOState.data.activeSearchSpace)
        });
        const data = await res.json();
        if (data.success) {
            if (data.space) { window.HPOState.data.activeSearchSpace = data.space; }
            clearParamModified(param);
            renderDashboardSearchSpaceSummary();
        } else { alert("Failed to update constraints: " + data.detail); }
    } catch (err) { alert("Error updating constraints: " + err); }
}

function renderSearchSpace() {
    const container = document.getElementById("search-space-controls");
    if (!container) return;
    let html = "";

    Object.entries(window.HPOState.data.activeSearchSpace).forEach(([param, val]) => {
        const label = paramLabel(param);
        
        if (val.type === "float" || val.type === "float_log" || val.type === "int") {
            html += `<div class="control-group" id="group-${param}" style="display: flex; align-items: center; justify-content: space-between; gap: 12px; padding: 10px 12px; border-bottom: 1px solid var(--border-color); position: relative; transition: all 0.2s ease;">
                <!-- Retro Indicator Strip -->
                <div class="modified-indicator" id="indicator-${param}" style="position: absolute; left: 0; top: 0; bottom: 0; width: 3px; background-color: var(--accent-color); opacity: 0; transition: opacity 0.2s ease;"></div>
                
                <div class="control-label" style="flex: 0 0 200px; margin-right: 24px; font-weight: 500; font-size: 0.85rem; color: var(--text-color); display: flex; flex-direction: column; gap: 4px;">
                    <span>${label}</span>
                    <span style="font-size: 0.72rem; color: var(--text-muted); font-family: var(--font-mono); font-weight: normal;">${val.type}</span>
                </div>
                
                <div style="flex: 1; display: flex; align-items: center; gap: 12px; justify-content: flex-start;">
                    <div class="range-inputs" style="display: flex; align-items: center; gap: 6px;">
                        <input type="number" step="any" class="number-input" id="${param}-min" value="${val.min}" style="width: 110px; padding: 4px 8px; font-family: var(--font-mono); font-size: 0.82rem; background: rgba(255,255,255,0.03); border: 1px solid var(--border-color); border-radius: 0; color: #fff; text-align: right;" oninput="markParamModified('${param}')">
                        <span style="color: var(--text-muted); font-size: 0.8rem;">to</span>
                        <input type="number" step="any" class="number-input" id="${param}-max" value="${val.max}" style="width: 110px; padding: 4px 8px; font-family: var(--font-mono); font-size: 0.82rem; background: rgba(255,255,255,0.03); border: 1px solid var(--border-color); border-radius: 0; color: #fff; text-align: right;" oninput="markParamModified('${param}')">
                    </div>
                    <button id="apply-btn-${param}" class="row-apply-btn" onclick="applySingleParamConstraints('${param}')" style="display: none; padding: 4px 10px; font-size: 0.75rem; background: var(--accent-color); color: #000; border: none; border-radius: 0; font-weight: 600; cursor: pointer; transition: all 0.2s ease;">Apply</button>
                </div>
            </div>`;
        } else if (val.type === "categorical") {
            const pills = val.options.map(opt => {
                const active = val.active.includes(opt);
                return `<button class="pill-btn ${active ? 'active' : ''}" onclick="togglePill('${param}', '${opt}', this)">${opt}</button>`;
            }).join("");

            html += `<div class="control-group" id="group-${param}" style="display: flex; align-items: center; justify-content: space-between; gap: 12px; padding: 10px 12px; border-bottom: 1px solid var(--border-color); position: relative; transition: all 0.2s ease;">
                <!-- Retro Indicator Strip -->
                <div class="modified-indicator" id="indicator-${param}" style="position: absolute; left: 0; top: 0; bottom: 0; width: 3px; background-color: var(--accent-color); opacity: 0; transition: opacity 0.2s ease;"></div>
                
                <div class="control-label" style="flex: 0 0 200px; margin-right: 24px; font-weight: 500; font-size: 0.85rem; color: var(--text-color); display: flex; flex-direction: column; gap: 4px;">
                    <span>${label}</span>
                    <span style="font-size: 0.72rem; color: var(--text-muted); font-family: var(--font-mono); font-weight: normal;">${val.type}</span>
                </div>
                
                <div style="flex: 1; display: flex; align-items: center; gap: 12px; justify-content: flex-start; flex-wrap: wrap;">
                    <div class="pill-container" style="display: flex; gap: 6px; flex-wrap: wrap;">
                        ${pills}
                    </div>
                    <button id="apply-btn-${param}" class="row-apply-btn" onclick="applySingleParamConstraints('${param}')" style="display: none; padding: 4px 10px; font-size: 0.75rem; background: var(--accent-color); color: #000; border: none; border-radius: 0; font-weight: 600; cursor: pointer; transition: all 0.2s ease;">Apply</button>
                </div>
            </div>`;
        }
    });

    container.innerHTML = html;
}

function renderDashboardSearchSpaceSummary() {
    const summaryDiv = document.getElementById("dashboard-search-space-summary-content");
    if (!summaryDiv) return;
    
    let html = "";
    Object.entries(window.HPOState.data.activeSearchSpace).forEach(([param, val]) => {
        const label = paramLabel(param);
        let boundsStr = "";
        if (val.type === "float" || val.type === "float_log" || val.type === "int") {
            const formatNum = (n) => {
                if (n == null) return "any";
                if (Math.abs(n) < 1e-3 && n !== 0) return n.toExponential(2);
                return Number(n.toFixed(4));
            };
            boundsStr = `[${formatNum(val.min)}, ${formatNum(val.max)}]`;
        } else if (val.type === "categorical") {
            const activeCount = val.active ? val.active.length : 0;
            const totalCount = val.options ? val.options.length : 0;
            boundsStr = `${activeCount}/${totalCount} active`;
        }
        
        html += `<div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px dashed rgba(255, 255, 255, 0.05); padding-bottom: 4px; margin-bottom: 4px;">
            <span style="color: var(--text-muted);">${label}</span>
            <span style="color: var(--text-color); font-weight: 500;">${boundsStr}</span>
        </div>`;
    });
    
    if (!html) {
        html = `<div style="text-align: center; color: var(--text-muted); font-size: 0.8rem; padding: 10px;">No parameters configured.</div>`;
    }
    
    summaryDiv.innerHTML = html;
}

function markParamModified(param) {
    window.HPOState.ui.modifiedParams.add(param);
    const indicator = document.getElementById(`indicator-${param}`);
    const btn = document.getElementById(`apply-btn-${param}`);
    if (indicator) indicator.style.opacity = 1;
    if (btn) btn.style.display = "inline-block";
}

function clearParamModified(param) {
    window.HPOState.ui.modifiedParams.delete(param);
    const indicator = document.getElementById(`indicator-${param}`);
    const btn = document.getElementById(`apply-btn-${param}`);
    if (indicator) indicator.style.opacity = 0;
    if (btn) btn.style.display = "none";
}

function switchSettingsTab(tabId) {
    document.querySelectorAll(".tab-btn").forEach((btn) => btn.classList.remove("active"));
    document.querySelectorAll(".tab-panel").forEach((panel) => panel.classList.remove("active"));
    const btn = document.getElementById(`tab-btn-${tabId}`);
    const panel = document.getElementById(`settings-tab-${tabId}`);
    if (btn) btn.classList.add("active");
    if (panel) panel.classList.add("active");
}

window.fetchPendingChanges = fetchPendingChanges;
window.populateEvalProtocolForm = populateEvalProtocolForm;
window.saveEvalProtocol = saveEvalProtocol;
window.saveIdeSettings = saveIdeSettings;
window.applyPendingChanges = applyPendingChanges;
window.discardPendingChanges = discardPendingChanges;
window.togglePill = togglePill;
window.applySingleParamConstraints = applySingleParamConstraints;
window.renderSearchSpace = renderSearchSpace;
window.renderDashboardSearchSpaceSummary = renderDashboardSearchSpaceSummary;
window.markParamModified = markParamModified;
window.clearParamModified = clearParamModified;
window.switchSettingsTab = switchSettingsTab;
