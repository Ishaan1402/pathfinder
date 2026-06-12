function paramLabel(key) {
    if (!window.HPOState.data.hpoConfig) return key;
    return window.HPOState.data.hpoConfig.param_labels?.[key] || key;
}

function fanovaParamLabel(param) {
    const legacy = { encoder_name: "model_capacity" };
    return paramLabel(legacy[param] || param);
}

function formatTrainResolution(trial) {
    const key =
        window.HPOState.data.hpoConfig?.eval_protocol?.train_resolution_param || "resolution";
    const val =
        trial.train_resolution ??
        trial.params?.[key] ??
        trial.params?.resolution;
    return val !== undefined && val !== null ? val : "-";
}

function formatMetric(val, empty = "-") {
    return val !== null && val !== undefined ? Number(val).toFixed(4) : empty;
}

function formatLR(value) {
    if (value === undefined || value === null) return "-";
    const n = Number(value);
    if (!Number.isFinite(n)) return "-";
    if (n >= 0.01) return n.toFixed(3);
    if (n >= 0.001) return n.toFixed(4);
    return n.toExponential(0);
}

function formatTime(totalSeconds) {
    const mins = Math.floor(totalSeconds / 60);
    const secs = totalSeconds % 60;
    return `${mins}m ${secs}s`;
}

function copyTrialConfigToClipboard(trialNumber, format) {
    const trial = window.HPOState.data.trials.find(t => t.number === trialNumber);
    if (!trial) return;

    let text = "";
    if (format === 'json') {
        text = JSON.stringify(trial.params, null, 2);
    } else if (format === 'cli') {
        text = Object.entries(trial.params)
            .map(([k, v]) => `--${k} ${v}`)
            .join(" ");
    }

    navigator.clipboard.writeText(text).then(() => {
        showToast(`Trial #${trialNumber} config copied as ${format.toUpperCase()}`);
    }).catch(err => {
        console.error("Failed to copy: ", err);
    });
}

function copyTrialsToJson() {
    if (!window.HPOState.data.trials || window.HPOState.data.trials.length === 0) {
        showToast("No trial data available to copy.");
        return;
    }
    const cleanJson = JSON.stringify(window.HPOState.data.trials, null, 2);
    navigator.clipboard.writeText(cleanJson).then(() => {
        showToast("All trials data copied to clipboard as JSON.");
    }).catch(err => {
        console.error("Failed to copy trials JSON: ", err);
    });
}

function policyActionLabel(action) {
    const labels = {
        no_change: "No change",
        update_search_space: "Search space adjusted",
        enqueue_one_manual_trial: "Manual trial queued",
    };
    return labels[action] || action || "Unknown action";
}

function formatReviewDate(createdAt) {
    if (!createdAt) return "";
    const d = new Date(createdAt);
    if (Number.isNaN(d.getTime())) return String(createdAt);
    return d.toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric" });
}

function formatOutcomeStatus(status) {
    const labels = {
        measured: "Outcome measured",
        pending: "Outcome pending",
        not_applicable: "No change applied",
        inconclusive: "Outcome inconclusive",
    };
    return labels[status] || status || "";
}

function buildHealthBlocks(rating) {
    if (rating == null || rating <= 0) return null;
    const n = Math.max(0, Math.min(5, Math.round(rating)));
    const filled = "■ ".repeat(n).trim();
    const empty = "□ ".repeat(5 - n).trim();
    return (filled + (filled && empty ? " " : "") + empty).trim();
}

function formatReviewTimestamp(createdAt) {
    if (!createdAt) return "";
    const d = new Date(createdAt);
    if (Number.isNaN(d.getTime())) return String(createdAt);
    return d.toISOString().replace("T", " ").slice(0, 16) + " UTC";
}

function getLatestCoordinatorAction(pastReviews, pendingChanges) {
    if (pendingChanges && Object.keys(pendingChanges).length > 0) {
        const keys = Object.keys(pendingChanges).map((k) => window.paramLabel(k)).join(", ");
        return {
            kind: "pending",
            label: "Pending approval",
            detail: `Search space adjustment (${keys})`,
            timestamp: null,
        };
    }
    const actionable = (pastReviews || []).find((r) => r.policy_action && r.policy_action !== "no_change");
    if (actionable) {
        return {
            kind: "review",
            label: policyActionLabel(actionable.policy_action),
            detail: (actionable.summary || "").slice(0, 100),
            timestamp: actionable.created_at,
            reviewId: actionable.id,
        };
    }
    return null;
}

window.paramLabel = paramLabel;
window.fanovaParamLabel = fanovaParamLabel;
window.formatTrainResolution = formatTrainResolution;
window.formatMetric = formatMetric;
window.formatLR = formatLR;
window.formatTime = formatTime;
window.copyTrialConfigToClipboard = copyTrialConfigToClipboard;
window.copyTrialsToJson = copyTrialsToJson;
window.policyActionLabel = policyActionLabel;
window.formatReviewDate = formatReviewDate;
window.formatOutcomeStatus = formatOutcomeStatus;
window.buildHealthBlocks = buildHealthBlocks;
window.formatReviewTimestamp = formatReviewTimestamp;
window.getLatestCoordinatorAction = getLatestCoordinatorAction;

