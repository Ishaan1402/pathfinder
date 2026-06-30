function getPrimaryScoreKey() {
    return window.HPOState.data.hpoConfig?.primary_score_key || "score";
}

function filterTrialsForDisplay(trials, paretoSet) {
    if (!window.HPOState.ui.filterTopPerformersOnly || !trials || !trials.length) return trials;
    const scoreKey = getPrimaryScoreKey();
    const completed = trials.filter((t) => t.state === "COMPLETE" && t[scoreKey] != null);
    if (completed.length < 3) return trials;
    const directions = window.HPOState.data.latestStudyData?.study_directions;
    const isSingleObj = !directions || directions.length === 1;
    const dirMultiplier = (isSingleObj && directions && directions[0] === "MINIMIZE") ? 1 : -1;
    const sorted = [...completed].sort((a, b) => ((a[scoreKey] ?? 0) - (b[scoreKey] ?? 0)) * dirMultiplier);
    const n = Math.max(1, Math.ceil(sorted.length * 0.15));
    const keepIds = new Set(sorted.slice(0, n).map((t) => t.number));
    if (paretoSet) paretoSet.forEach((num) => keepIds.add(num));
    trials.filter((t) => t.state === "RUNNING").forEach((t) => keepIds.add(t.number));
    return trials.filter((t) => keepIds.has(t.number));
}

function refreshFilteredVisualizations() {
    if (!window.HPOState.data.latestStudyData) return;
    window.HPOState.render.lastTrialsSnapshot = null;
    if (window.HPOState.ui.isModalOpen) {
        window.HPOState.render.pendingRender = true;
        return;
    }
    renderStudyDetails(window.HPOState.data.latestStudyData);
}

function syncTopPerformersFilterCheckboxes() {
    const a = document.getElementById("filter-top-performers");
    const d = document.getElementById("filter-top-performers-dashboard");
    if (a) a.checked = window.HPOState.ui.filterTopPerformersOnly;
    if (d) d.checked = window.HPOState.ui.filterTopPerformersOnly;
}

function onTopPerformersFilterChange(fromDashboard) {
    const a = document.getElementById("filter-top-performers");
    const d = document.getElementById("filter-top-performers-dashboard");
    const src = fromDashboard ? d : a;
    window.HPOState.ui.filterTopPerformersOnly = !!(src && src.checked);
    sessionStorage.setItem(window.HPOState.constants.HPO_FILTER_TOP_KEY, window.HPOState.ui.filterTopPerformersOnly ? "1" : "0");
    syncTopPerformersFilterCheckboxes();
    refreshFilteredVisualizations();
}

function applyTableFilters(trials, filters) {
    const keys = Object.keys(filters || {});
    if (!keys.length) return trials;
    return trials.filter((t) => {
        for (const colKey of keys) {
            const f = filters[colKey];
            const val = getTrialCellValue(t, colKey);
            if (f.type === "set" && f.values?.length) {
                if (!f.values.includes(val == null ? "" : String(val))) return false;
            } else if (f.type === "range") {
                const n = Number(val);
                if (val == null || Number.isNaN(n) || (f.min != null && n < f.min) || (f.max != null && n > f.max)) return false;
            }
        }
        return true;
    });
}

function applyTableSort(trials, sort) {
    return (sort?.col && sort.dir) ? [...trials].sort((a, b) => compareCellValues(getTrialCellValue(a, sort.col), getTrialCellValue(b, sort.col), sort.dir)) : trials;
}

function applyTablePipeline(trials, tableId) {
    const state = window.HPOState.tables[tableId];
    return applyTableSort(applyTableFilters(trials, state.filters), state.sort);
}

function cycleTableSort(tableId, colKey) {
    const sort = window.HPOState.tables[tableId].sort;
    // Loss defaults to ascending (lower is better).
    // Others (number/Trial, score/eval, parameters) default to descending (higher/latest is better).
    const defaultDir = (colKey === "loss") ? "asc" : "desc";
    
    if (sort.col !== colKey) {
        sort.col = colKey;
        sort.dir = defaultDir;
    } else {
        sort.dir = sort.dir === "asc" ? "desc" : "asc";
    }
    refreshTableForId(tableId);
}

function closeAllFilterPopovers(except) {
    document.querySelectorAll(".col-filter-popover").forEach((p) => { if (p !== except) p.remove(); });
    document.querySelectorAll(".th-filter-btn.active").forEach((b) => b.classList.remove("active"));
    document.body.classList.remove("col-filter-open");
}

function positionFilterPopover(pop, anchorBtn) {
    document.body.appendChild(pop);
    const rect = anchorBtn.getBoundingClientRect();
    pop.style.position = "fixed";
    pop.style.top = `${rect.bottom + 2}px`;
    pop.style.left = `${rect.left}px`;
    const popRect = pop.getBoundingClientRect();
    if (popRect.right > window.innerWidth - 8) pop.style.left = `${Math.max(8, window.innerWidth - popRect.width - 8)}px`;
    if (popRect.bottom > window.innerHeight - 8) pop.style.top = `${Math.max(8, rect.top - popRect.height - 2)}px`;
}

function openColumnFilterPopover(tableId, colKey, anchorBtn, trials) {
    closeAllFilterPopovers();
    anchorBtn.classList.add("active");
    document.body.classList.add("col-filter-open");
    const pop = document.createElement("div");
    pop.className = "col-filter-popover";

    const values = trials.map(t => getTrialCellValue(t, colKey)).filter(v => v != null && v !== "");
    const isNumeric = values.length > 0 && values.every((v) => typeof v === "number" || !Number.isNaN(Number(v)));
    const state = window.HPOState.tables[tableId].filters[colKey];

    if (isNumeric) {
        pop.innerHTML = `<div class="col-filter-popover-title">Filter range</div>
            <div class="col-filter-popover-range">
                <input type="number" class="col-filter-number" id="f-min" placeholder="Min" value="${state?.min ?? ''}">
                <input type="number" class="col-filter-number" id="f-max" placeholder="Max" value="${state?.max ?? ''}">
            </div>`;
        const minInput = pop.querySelector("#f-min"), maxInput = pop.querySelector("#f-max");
        let debounce;
        const apply = () => {
            clearTimeout(debounce);
            debounce = setTimeout(() => {
                const min = minInput.value === "" ? null : Number(minInput.value);
                const max = maxInput.value === "" ? null : Number(maxInput.value);
                if (min == null && max == null) {
                    delete window.HPOState.tables[tableId].filters[colKey];
                    anchorBtn.classList.remove("active");
                } else {
                    window.HPOState.tables[tableId].filters[colKey] = { type: "range", min, max };
                    anchorBtn.classList.add("active");
                }
                refreshTableForId(tableId);
            }, 150);
        };
        minInput.addEventListener("input", apply);
        maxInput.addEventListener("input", apply);
    } else {
        const title = document.createElement("div");
        title.className = "col-filter-popover-title";
        title.textContent = "Filter values";
        pop.appendChild(title);

        const uniq = [...new Set(values.map(String))].sort();
        uniq.forEach((val) => {
            const lbl = document.createElement("label");
            lbl.className = "hpo-toggle hpo-toggle--compact col-filter-option";
            lbl.innerHTML = `<input type="checkbox" value="${val}" ${(!state?.values || state.values.includes(val)) ? 'checked' : ''}>
                <span class="hpo-toggle-box" aria-hidden="true"></span>
                <span class="hpo-toggle-label">${val}</span>`;
            lbl.querySelector("input").addEventListener("change", () => {
                const checked = [...pop.querySelectorAll("input[type=checkbox]:checked")].map((c) => c.value);
                if (!checked.length || checked.length === uniq.length) {
                    delete window.HPOState.tables[tableId].filters[colKey];
                    anchorBtn.classList.remove("active");
                } else {
                    window.HPOState.tables[tableId].filters[colKey] = { type: "set", values: checked };
                    anchorBtn.classList.add("active");
                }
                refreshTableForId(tableId);
            });
            pop.appendChild(lbl);
        });
    }

    positionFilterPopover(pop, anchorBtn);
    document.addEventListener("click", function onDoc(e) {
        if (!pop.contains(e.target) && e.target !== anchorBtn && !anchorBtn.contains(e.target)) {
            pop.remove();
            anchorBtn.classList.remove("active");
            document.body.classList.remove("col-filter-open");
            document.removeEventListener("click", onDoc);
        }
    });
}

function refreshTableForId(tableId) {
    const data = window.HPOState.data.latestStudyData;
    if (!data?.trials) return;
    const paretoSet = new Set(data.pareto_trials || []);
    const base = filterTrialsForDisplay(data.trials, paretoSet);
    if (tableId === "dashboard") renderDashboardTableBody(base, paretoSet, data);
    else renderAnalysisTableBody(base);
}

function getDisplayTrialsForCurrentStudy() {
    const paretoSet = new Set((window.HPOState.data.latestStudyData && window.HPOState.data.latestStudyData.pareto_trials) || []);
    return { trials: filterTrialsForDisplay(window.HPOState.data.trials || [], paretoSet), paretoSet };
}

Object.assign(window, { getPrimaryScoreKey, filterTrialsForDisplay, refreshFilteredVisualizations, syncTopPerformersFilterCheckboxes, onTopPerformersFilterChange, applyTableFilters, applyTableSort, applyTablePipeline, cycleTableSort, closeAllFilterPopovers, positionFilterPopover, openColumnFilterPopover, refreshTableForId, getDisplayTrialsForCurrentStudy });
