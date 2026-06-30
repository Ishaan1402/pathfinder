function syncTrialColumnWidth(table, trials) {
    if (!table || !trials?.length) return;
    const trialTh = table.querySelector("th.col-trial, th[data-col-key='number']");
    if (!trialTh) return;
    const colLabel = trialTh.dataset.colLabel || "Trial";
    if (window.HPOState.ui.columnWidthsUserResized && window.HPOState.ui.columnWidthsUserResized.has(colLabel)) return;

    const maxNum = trials.reduce((m, t) => Math.max(m, t.number ?? 0), 0);
    const digits = Math.max(2, String(maxNum).length);
    const width = Math.max(64, Math.min(90, 44 + digits * 8));
    trialTh.style.width = `${width}px`;
    window.HPOState.ui.columnWidths[colLabel] = width;
}

function autoFitColumnWidth(table, colIndex) {
    if (!table) return;
    const ths = table.querySelectorAll("thead th");
    if (colIndex >= ths.length) return;
    const th = ths[colIndex];
    const colLabel = th.dataset.colLabel || th.textContent.trim().replace(/[\s\r\n\t]+/g, " ");

    const origTableLayout = table.style.tableLayout;
    table.style.tableLayout = "auto";
    th.style.width = "";

    // Force layout / get natural width of the column
    let naturalWidth = th.offsetWidth;

    // Restore table layout
    table.style.tableLayout = origTableLayout;

    // Apply safety padding for sort caret, filter button, and spacing
    const paddedWidth = Math.max(65, Math.min(450, naturalWidth + 28));

    th.style.width = `${paddedWidth}px`;
    window.HPOState.ui.columnWidths[colLabel] = paddedWidth;
    window.HPOState.ui.columnWidthsUserResized = window.HPOState.ui.columnWidthsUserResized || new Set();
    window.HPOState.ui.columnWidthsUserResized.add(colLabel);
}

function makeTableResizable(table, force = false) {
    if (!table) return;
    if (table.offsetWidth === 0) return;

    const oldHandles = table.querySelectorAll(".resize-handle");
    oldHandles.forEach(h => h.remove());

    const cols = table.querySelectorAll("thead th");
    for (let i = 0; i < cols.length - 1; i++) {
        const col = cols[i];
        col.style.position = "sticky";
        
        const handle = document.createElement("div");
        handle.className = "resize-handle";
        col.appendChild(handle);

        const colLabel = col.dataset.colLabel || col.textContent.trim().replace(/[\s\r\n\t]+/g, " ");

        if (window.HPOState.ui.columnWidths[colLabel] !== undefined) {
            col.style.width = `${window.HPOState.ui.columnWidths[colLabel]}px`;
        } else {
            if (force || !col.style.width || col.style.width === "0px") {
                const w = col.offsetWidth;
                if (w > 0) {
                    col.style.width = `${w}px`;
                    window.HPOState.ui.columnWidths[colLabel] = w;
                }
            }
        }

        let startX, startWidth;
        handle.addEventListener("mousedown", (e) => {
            e.preventDefault();
            e.stopPropagation();
            startX = e.clientX;
            startWidth = parseFloat(col.style.width) || col.offsetWidth;
            
            handle.classList.add("resizing");
            document.body.style.cursor = "col-resize";
            
            const onMouseMove = (moveEvent) => {
                const dx = moveEvent.clientX - startX;
                const newWidth = Math.max(50, startWidth + dx);
                col.style.width = `${newWidth}px`;
                window.HPOState.ui.columnWidths[colLabel] = newWidth;
                window.HPOState.ui.columnWidthsUserResized = window.HPOState.ui.columnWidthsUserResized || new Set();
                window.HPOState.ui.columnWidthsUserResized.add(colLabel);
            };

            const onMouseUp = () => {
                handle.classList.remove("resizing");
                document.body.style.cursor = "";
                document.removeEventListener("mousemove", onMouseMove);
                document.removeEventListener("mouseup", onMouseUp);
            };

            document.addEventListener("mousemove", onMouseMove);
            document.addEventListener("mouseup", onMouseUp);
        });

        // Autofit on double-click
        handle.addEventListener("dblclick", (e) => {
            e.preventDefault();
            e.stopPropagation();
            autoFitColumnWidth(table, i);
        });
    }
}

function getTrialCellValue(trial, colKey) {
    if (colKey === "number") return trial.number;
    if (colKey === "state") return trial.state;
    if (colKey === "loss") return trial.loss;
    if (colKey === "score") return trial[getPrimaryScoreKey()] ?? trial.score;
    if (colKey === "score_eval_fixed") return trial.score_eval_fixed;
    if (colKey.startsWith("param:")) return trial.params?.[colKey.slice(6)];
    return null;
}

function compareCellValues(aVal, bVal, dir) {
    const mul = dir === "asc" ? 1 : -1;
    const aNull = aVal == null || aVal === "";
    const bNull = bVal == null || bVal === "";
    if (aNull && bNull) return 0;
    if (aNull) return 1;
    if (bNull) return -1;
    if (typeof aVal === "number" && typeof bVal === "number") {
        return (aVal - bVal) * mul;
    }
    return String(aVal).localeCompare(String(bVal), undefined, { numeric: true }) * mul;
}

function buildSortableTh(label, colKey, tableId) {
    const sort = window.HPOState.tables[tableId].sort;
    let caret = "";
    if (sort.col === colKey) {
        caret = sort.dir === "asc" ? "▲" : "▼";
    }
    const hasFilter = !!window.HPOState.tables[tableId].filters[colKey];
    const colClass = colKey === "number" ? " col-trial" : "";
    
    // Hide filter button on the trial number column
    const filterBtn = colKey === "number" ? "" : `
        <button type="button" class="th-filter-btn${hasFilter ? " active" : ""}" data-filter-col="${colKey}" title="Filter">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" style="width: 10px; height: 10px; display: block; pointer-events: none;">
                <polygon points="22 3 2 3 10 12.46 10 19 14 21 14 12.46 22 3"></polygon>
            </svg>
        </button>
    `;
    
    return `<th class="th-sortable${colClass}" data-col-key="${colKey}" data-col-label="${label.replace(/"/g, "&quot;")}">
        <span class="th-label-wrap">${label}<span class="sort-caret${caret ? " active" : ""}">${caret}</span></span>
        ${filterBtn}
    </th>`;
}

function bindTableHeaderInteractions(table, tableId, baseTrials) {
    if (!table) return;
    table.querySelectorAll(".th-sortable").forEach((th) => {
        th.addEventListener("click", (e) => {
            if (e.target.closest(".th-filter-btn") || e.target.closest(".resize-handle")) return;
            cycleTableSort(tableId, th.dataset.colKey);
        });
    });
    table.querySelectorAll(".th-filter-btn").forEach((btn) => {
        btn.addEventListener("click", (e) => {
            e.stopPropagation();
            openColumnFilterPopover(tableId, btn.dataset.filterCol, btn, baseTrials);
        });
    });
}

window.syncTrialColumnWidth = syncTrialColumnWidth;
window.makeTableResizable = makeTableResizable;
window.getTrialCellValue = getTrialCellValue;
window.compareCellValues = compareCellValues;
window.buildSortableTh = buildSortableTh;
window.bindTableHeaderInteractions = bindTableHeaderInteractions;
