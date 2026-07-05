window.HPOState = {
    session: {
        studyName: new URLSearchParams(window.location.search).get("study")
            || new URLSearchParams(window.location.search).get("study_name")
            || null,
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
        statisticalConfidence: "low",
        completedCount: 0,
    },
    ui: {
        accentColorHex: "#06b6d4",
        filterTopPerformersOnly: false,
        isModalOpen: false,
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
    },
};
window.HPOState.ui.filterTopPerformersOnly =
    sessionStorage.getItem(window.HPOState.constants.HPO_FILTER_TOP_KEY) === "1";
