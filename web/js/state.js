window.HPOState = {
    session: {
        studyName: new URLSearchParams(window.location.search).get("study")
            || new URLSearchParams(window.location.search).get("study_name")
            || "bridge_crack_study",
    },
    data: {
        trials: [],
        latestStudyData: null,
        hpoConfig: null,
        evalInsights: null,
        review: null,
        activeSearchSpace: {},
        thoughtLogs: [],
        studyHealthReason: "",
        healthTier: "healthy",
        healthReason: "",
        pastReviews: [],
        pendingChanges: null,
        statisticalConfidence: "low",
        completedCount: 0,
    },
    ui: {
        accentColorHex: "#06b6d4",
        filterTopPerformersOnly: false,
        isModalOpen: false,
        reviewPillDismissed: false,
        reviewPillDismissTimeout: null,
        toastTimeout: null,
        columnWidths: {},
        modifiedParams: new Set(),
        parallelCoordinatesHoveredTrial: null,
    },
    render: {
        pendingRender: false,
        lastTrialsSnapshot: null,
        lastDashboardHeaderSnapshot: "",
        lastAnalysisHeaderSnapshot: "",
        lastFanovaPayload: null,
        lastFanovaRenderKey: null,
        lastParetoPointCount: 0,
    },
    telemetry: {
        trialNumber: null,
        lastCompletedCount: null,
    },
    poll: {
        failures: 0,
        timeoutId: null,
    },
    charts: {
        pareto: { instance: null },
        modalHistory: { instance: null },
    },
    tables: {
        dashboard: { sort: { col: "number", dir: "desc" }, filters: {} },
        analysis: { sort: { col: "number", dir: "desc" }, filters: {} },
    },
    constants: {
        HPO_FILTER_TOP_KEY: "hpo_filter_top",
        DASHBOARD_TABLE_COLS: 9,
        HPO_MARK_SVG: `
    <div class="hpo-mark" role="img" aria-label="Pathfinder">
        <svg class="hpo-mark-svg" viewBox="0 0 64 64" xmlns="http://www.w3.org/2000/svg">
            <g class="hpo-planet-wrap">
                <path class="hpo-water hpo-water-a" d="M32 12c9-1 18 4 22 14 4 11-1 24-12 30-11 6-24 2-30-10-6-12 0-25 11-31 4-2 7-3 9-3z"/>
                <path class="hpo-water hpo-water-b" d="M32 13c8-2 17 3 21 13 5 12 0 25-11 31-12 6-25 1-31-11-6-12 1-24 12-30 3-1 6-2 9-3z"/>
                <path class="hpo-depth" d="M32 30c8 0 14 5 13 12-1 7-8 12-16 11-7-1-12-7-11-13 1-6 7-10 14-10z"/>
                <ellipse class="hpo-sheen" cx="25" cy="24" rx="9" ry="5" transform="rotate(-18 25 24)"/>
                <path class="hpo-surface-line" d="M19 25q7-4 13-2 8 2 15-1"/>
                <circle class="hpo-click-ripple" cx="32" cy="34" r="4"/>
            </g>
            <g class="hpo-forming">
                <path class="hpo-wisp hpo-wisp-1" d="M46 18c3 0 5 2 4 5-1 2-4 3-6 1-2-2-1-5 2-6z"/>
                <path class="hpo-wisp hpo-wisp-2" d="M14 40c3 1 4 4 2 6-2 2-5 1-6-1-1-3 1-5 4-5z"/>
                <path class="hpo-wisp hpo-wisp-3" d="M40 48c2 2 2 5-1 6-3 1-5-1-5-4 0-3 2-4 6-2z"/>
            </g>
        </svg>
    </div>`,
    },
};
window.HPOState.ui.filterTopPerformersOnly =
    sessionStorage.getItem(window.HPOState.constants.HPO_FILTER_TOP_KEY) === "1";
