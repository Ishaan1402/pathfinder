function renderHealthLatestAction() {
    const body = document.getElementById("health-action-body");
    if (!body) return;
    body.replaceChildren();
    const action = window.getLatestCoordinatorAction(window.HPOState.data.pastReviews, window.HPOState.data.pendingChanges);
    if (!action) {
        body.appendChild(Object.assign(document.createElement("div"), { className: "health-action-line health-action-muted", textContent: "No coordinator action recorded" }));
        return;
    }
    body.appendChild(Object.assign(document.createElement("div"), { className: "health-action-line", textContent: action.label }));
    body.appendChild(Object.assign(document.createElement("div"), { className: "health-action-target", textContent: action.detail }));
    if (action.timestamp) {
        body.appendChild(Object.assign(document.createElement("div"), { className: "health-action-time", textContent: window.formatReviewTimestamp(action.timestamp) }));
    }
    if (action.kind === "pending") {
        body.appendChild(Object.assign(document.createElement("a"), { className: "health-action-link", href: "#search-space", textContent: "Review in Search Space →" }));
    }
}

function renderHealthPanelExtras() {
    const tier = window.HPOState.data.healthTier || "healthy", latestReview = (window.HPOState.data.pastReviews || [])[0], rating = latestReview?.health_rating;
    const tierLine = document.getElementById("health-tier-line"), blocksEl = document.getElementById("health-rating-blocks");
    if (tierLine) tierLine.textContent = tier.charAt(0).toUpperCase() + tier.slice(1);
    if (blocksEl) {
        const blocks = window.buildHealthBlocks(rating);
        blocksEl.textContent = blocks ? (blocks + (rating ? `  ${rating}/5` : "")) : "— — — — —";
        blocksEl.classList.toggle("health-blocks-empty", !blocks);
    }
    renderHealthLatestAction();
}

function renderPastReviews(reviews) {
    const details = document.getElementById("audit-history-details"), summary = document.getElementById("audit-history-summary"), list = document.getElementById("past-reviews-list");
    if (!list) return;
    window.HPOState.data.pastReviews = reviews || [];
    list.replaceChildren();
    const count = (reviews || []).length;
    if (summary) summary.textContent = count ? `Show audit history (${count} logs)` : "No audit history";
    if (details) details.style.display = count ? "" : "none";
    if (!count) { renderHealthPanelExtras(); return; }
    reviews.forEach((r) => {
        const row = Object.assign(document.createElement("div"), { className: "audit-history-row" + (r.policy_action === "no_change" ? " no-change" : "") });
        const parts = [`#${r.id}`, window.policyActionLabel(r.policy_action), window.formatReviewDate(r.created_at), (r.summary || "").slice(0, 80)].filter(Boolean);
        row.appendChild(Object.assign(document.createElement("div"), { textContent: parts.join(" · ") }));
        if (r.policy_action && r.policy_action !== "no_change") {
            const btn = Object.assign(document.createElement("button"), { type: "button", className: "audit-flag-btn", textContent: r.quality_flagged ? "Unflag" : "Flag" });
            btn.addEventListener("click", () => toggleReviewFlag(r.id, !r.quality_flagged));
            row.appendChild(btn);
        }
        list.appendChild(row);
    });
    renderHealthPanelExtras();
}

function updateAnalysisStatusTicker() {
    const ticker = document.getElementById("analysis-status-ticker"), track = document.getElementById("analysis-ticker-track");
    if (!ticker || !track) return;
    const tier = window.HPOState.data.healthTier || "healthy", reason = window.HPOState.data.healthReason || window.HPOState.data.studyHealthReason || "", review = window.HPOState.data.review, confidence = window.HPOState.data.statisticalConfidence || "low", nComplete = window.HPOState.data.completedCount || 0;
    const messages = [];
    if (tier === "watch" || tier === "intervene") messages.push(`${tier.charAt(0).toUpperCase() + tier.slice(1)}: ${reason}`);
    if (review?.review_recommended && (review.reasons || []).length && !window.HPOState.ui.reviewPillDismissed) {
        messages.push(`Coordinator review recommended — ` + review.reasons.map((r) => r.message || r.code).join(" · "));
    }
    if (confidence === "low") messages.push(`${nComplete} complete trial${nComplete === 1 ? "" : "s"}. Early signal; treat rankings as indicative.`);
    track.replaceChildren();
    if (!messages.length) {
        ticker.classList.add("hidden");
        ticker.classList.remove("tier-healthy", "tier-watch", "tier-intervene");
        return;
    }
    let displayTier = tier === "intervene" ? "intervene" : (tier === "watch" || (review?.review_recommended && (review.reasons || []).length && !window.HPOState.ui.reviewPillDismissed) || confidence === "low" ? "watch" : "healthy");
    let separator = "  ▲  ";
    if (displayTier === "watch") {
        separator = "  ◆  ";
    } else if (displayTier === "intervene") {
        separator = "  ▼  ";
    }
    const text = messages.map((m) => m.toUpperCase()).join(separator) + separator;
    ticker.className = `analysis-status-ticker tier-${displayTier}`;
    ticker.dataset.tier = displayTier;
    [text, text].forEach((segment) => track.appendChild(Object.assign(document.createElement("span"), { className: "ticker-segment", textContent: segment })));
    track.style.setProperty("--ticker-duration", `${Math.max(20, Math.min(45, text.length * 0.28))}s`);
}

async function toggleReviewFlag(reviewId, flagged) {
    try {
        await fetch(`/api/flag_review?review_id=${reviewId}&flagged=${flagged}`, { method: "POST" });
        fetchStudyDetails();
    } catch (err) { console.error("Failed to flag review:", err); }
}

function applyReviewUi() {
    const recommended = window.HPOState.data.review?.review_recommended, reasons = window.HPOState.data.review?.reasons || [];
    const show = !!(recommended && reasons.length) && !window.HPOState.ui.reviewPillDismissed;
    const pill = document.getElementById("dashboard-review-pill"), pillReasons = document.getElementById("dashboard-review-reasons");
    if (pill) {
        pill.classList.toggle("hidden", !show);
        if (pillReasons) pillReasons.textContent = show ? reasons.map((r) => r.message || r.code).join("  •  ") : "";
    }
    updateAnalysisStatusTicker();
}

async function copyReviewPrompt() {
    const pillAction = document.querySelector("#dashboard-review-pill .review-pill-action");
    try {
        const res = await fetch(`/api/review_packet?study_name=${window.HPOState.session.studyName}`);
        const data = await res.json();
        await navigator.clipboard.writeText(data.review_prompt || "");
        if (pillAction) {
            const prev = pillAction.textContent; pillAction.textContent = "Copied!";
            setTimeout(() => { pillAction.textContent = prev; }, 1600);
        }
        fetch(`/api/dismiss_coordinator_nudge?study_name=${window.HPOState.session.studyName}`, { method: "POST" }).catch(e => {});
        const mark = document.querySelector('.brand-mark'); if (mark) mark.classList.remove("ping-twice");
        if (window.HPOState.ui.reviewPillDismissTimeout) clearTimeout(window.HPOState.ui.reviewPillDismissTimeout);
        window.HPOState.ui.reviewPillDismissTimeout = setTimeout(() => { dismissReviewPill(); }, 120000);
    } catch (err) { console.error("Could not copy review prompt:", err); }
}

async function checkStudyHealth(throwOnError = false) {
    try {
        const res = await fetch(`/api/study_health?study_name=${window.HPOState.session.studyName}`);
        if (!res.ok) { if (throwOnError) throw new Error("HTTP error " + res.status); return; }
        const data = await res.json(), card = document.getElementById("study-health-card"), badge = document.getElementById("health-badge");
        const msg = document.getElementById("health-message"), actionBtn = document.getElementById("health-action-btn"), dismissBtn = document.getElementById("health-dismiss-btn"), mark = document.querySelector('.brand-mark');
        if (!card) return;
        window.HPOState.data.studyHealthReason = window.HPOState.data.healthReason = data.health_reason || "";
        window.HPOState.data.healthTier = data.health_tier || "healthy";
        card.className = "card health-card " + (data.health_tier || "healthy");
        if (mark) mark.classList.remove("ping-watch", "ping-intervene", "ping-twice");
        if (badge) { badge.className = `badge ${data.health_tier || 'healthy'}`; badge.textContent = (data.health_tier || 'healthy').toUpperCase(); }
        const isDismissed = !!data.is_dismissed;
        if (data.health_tier === "healthy") {
            msg.textContent = "No anomalies detected. Search space is healthy.";
            if (actionBtn) actionBtn.style.display = "none";
            if (dismissBtn) dismissBtn.style.display = "none";
        } else {
            msg.textContent = data.health_reason || (data.health_tier === "watch" ? "Watch condition detected." : "Intervention recommended.");
            if (actionBtn) actionBtn.style.display = data.health_tier === "intervene" ? "inline-block" : "none";
            if (dismissBtn) dismissBtn.style.display = isDismissed ? "none" : "inline-block";
            if (mark && !isDismissed) mark.classList.add(data.health_tier === "watch" ? "ping-watch" : "ping-intervene");
        }
        renderHealthPanelExtras(); updateAnalysisStatusTicker();
    } catch (err) { console.error("Error checking study health:", err); if (throwOnError) throw err; }
}

async function dismissHealthAlert() {
    try {
        const res = await fetch(`/api/dismiss_coordinator_nudge?study_name=${window.HPOState.session.studyName}`, { method: "POST" });
        if (res.ok) checkStudyHealth();
    } catch (err) { console.error("Error dismissing health alert:", err); }
}

function copyDiagnosticPrompt() {
    const diagnosticPayload = {
        anomaly: window.HPOState.data.studyHealthReason || "unknown anomaly",
        active_search_space: window.HPOState.data.activeSearchSpace || {},
        recent_trials: window.HPOState.data.trials.slice(0, 3).map(t => ({
            trial_id: t.number, state: t.state, params: t.params, score: t.dice, loss: t.bce
        }))
    };
    const promptText = `I need help debugging my Pathfinder study because it is failing health checks:
ANOMALY: ${diagnosticPayload.anomaly}

ACTIVE SEARCH SPACE:
${JSON.stringify(diagnosticPayload.active_search_space, null, 2)}

RECENT TRIALS:
${JSON.stringify(diagnosticPayload.recent_trials, null, 2)}

Please analyze the failure and suggest one policy action (update_search_space or enqueue_one_manual_trial).`;
    
    navigator.clipboard.writeText(promptText).then(() => {
        const btn = document.getElementById("health-action-btn");
        if (btn) {
            const orig = btn.textContent; btn.textContent = "Copied!";
            setTimeout(() => btn.textContent = orig, 2000);
        }
    }).catch(err => console.error("Could not copy diagnostic prompt:", err));
}

function dismissReviewPill() {
    window.HPOState.ui.reviewPillDismissed = true;
    const pill = document.getElementById("dashboard-review-pill");
    if (pill) pill.classList.add("hidden");
    
    // Clear double ping on dismiss via backend DB
    fetch(`/api/dismiss_coordinator_nudge?study_name=${window.HPOState.session.studyName}`, { method: "POST" })
        .catch(err => console.error("Error dismissing nudge:", err));
        
    const brandMark = document.querySelector('.brand-mark');
    if (brandMark) brandMark.classList.remove("ping-twice");
    
    if (window.HPOState.ui.reviewPillDismissTimeout) {
        clearTimeout(window.HPOState.ui.reviewPillDismissTimeout);
        window.HPOState.ui.reviewPillDismissTimeout = null;
    }
}

function updateStatConfidenceBanner(confidence, nComplete) {
    window.HPOState.data.statisticalConfidence = confidence || "low";
    window.HPOState.data.completedCount = Number.isFinite(nComplete) ? nComplete : 0;
    const count = window.HPOState.data.completedCount;
    const show = confidence === "low";
    const el = document.getElementById("stat-confidence-banner-dashboard");
    if (el) {
        el.replaceChildren();
        if (show) {
            const strong = document.createElement("strong");
            strong.textContent = `${count} complete trial${count === 1 ? "" : "s"}`;
            el.appendChild(strong);
            el.appendChild(document.createTextNode(". Early signal; treat rankings and parameter trends as indicative."));
        }
        el.classList.toggle("hidden", !show);
    }
    updateAnalysisStatusTicker();
}

function syncTelemetryFromTrials(trials) {
    const running = trials
        .filter((t) => t.state === "RUNNING")
        .sort((a, b) => a.number - b.number)[0];

    if (!running) {
        window.HPOState.telemetry.trialNumber = null;
        setWorkerRunIndicator(false, "Worker idle");
        return;
    }

    const epochSuffix =
        running.latest_epoch != null ? ` · E${running.latest_epoch}` : "";
    const label = `Trial #${running.number}${epochSuffix}`;
    if (window.HPOState.telemetry.trialNumber !== running.number) {
        window.HPOState.telemetry.trialNumber = running.number;
    }
    setWorkerRunIndicator(true, label);
}

function setWorkerRunIndicator(isLive, label) {
    const dot = document.getElementById("worker-run-dot");
    const text = document.getElementById("worker-run-label");
    const wrap = document.getElementById("worker-run-indicator");
    if (!dot || !text) return;
    dot.classList.toggle("live", isLive);
    dot.classList.toggle("idle", !isLive);
    text.textContent = label;
    if (wrap) wrap.title = label;
}

window.renderHealthLatestAction = renderHealthLatestAction;
window.renderHealthPanelExtras = renderHealthPanelExtras;
window.renderPastReviews = renderPastReviews;
window.updateAnalysisStatusTicker = updateAnalysisStatusTicker;
window.toggleReviewFlag = toggleReviewFlag;
window.applyReviewUi = applyReviewUi;
window.copyReviewPrompt = copyReviewPrompt;
window.checkStudyHealth = checkStudyHealth;
window.dismissHealthAlert = dismissHealthAlert;
window.copyDiagnosticPrompt = copyDiagnosticPrompt;
window.dismissReviewPill = dismissReviewPill;
window.updateStatConfidenceBanner = updateStatConfidenceBanner;
window.syncTelemetryFromTrials = syncTelemetryFromTrials;
window.setWorkerRunIndicator = setWorkerRunIndicator;
