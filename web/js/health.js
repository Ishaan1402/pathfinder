function renderHealthPanelExtras() {
    const tier = window.HPOState.data.healthTier || "healthy";
    const tierLine = document.getElementById("health-tier-line");
    if (tierLine) tierLine.textContent = tier.charAt(0).toUpperCase() + tier.slice(1);
}

function updateAnalysisStatusTicker() {
    const ticker = document.getElementById("analysis-status-ticker"), track = document.getElementById("analysis-ticker-track");
    if (!ticker || !track) return;
    const tier = window.HPOState.data.healthTier || "healthy", reason = window.HPOState.data.healthReason || window.HPOState.data.studyHealthReason || "", confidence = window.HPOState.data.statisticalConfidence || "low", nComplete = window.HPOState.data.completedCount || 0;
    const messages = [];
    if (tier === "watch" || tier === "intervene") messages.push(`${tier.charAt(0).toUpperCase() + tier.slice(1)}: ${reason}`);
    if (confidence === "low") messages.push(`${nComplete} complete trial${nComplete === 1 ? "" : "s"}. Early signal; treat rankings as indicative.`);
    track.replaceChildren();
    if (!messages.length) {
        ticker.classList.add("hidden");
        ticker.classList.remove("tier-healthy", "tier-watch", "tier-intervene");
        return;
    }
    let displayTier = tier === "intervene" ? "intervene" : (tier === "watch" || confidence === "low" ? "watch" : "healthy");
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

async function checkStudyHealth(throwOnError = false) {
    if (!window.HPOState.session.studyName) return;
    try {
        const res = await fetch(`/api/study_health?study_name=${window.HPOState.session.studyName}`);
        if (!res.ok) { if (throwOnError) throw new Error("HTTP error " + res.status); return; }
        const data = await res.json(), card = document.getElementById("study-health-card"), badge = document.getElementById("health-badge");
        const msg = document.getElementById("health-message"), mark = document.querySelector('.brand-mark');
        if (!card) return;
        window.HPOState.data.studyHealthReason = window.HPOState.data.healthReason = data.health_reason || "";
        window.HPOState.data.healthTier = data.health_tier || "healthy";
        card.className = "card health-card " + (data.health_tier || "healthy");
        if (mark) mark.classList.remove("ping-watch", "ping-intervene", "ping-twice");
        if (badge) { badge.className = `badge ${data.health_tier || 'healthy'}`; badge.textContent = (data.health_tier || 'healthy').toUpperCase(); }
        if (data.health_tier === "healthy") {
            msg.textContent = "No anomalies detected. Search space is healthy.";
        } else {
            msg.textContent = data.health_reason || (data.health_tier === "watch" ? "Watch condition detected." : "Intervention recommended.");
            if (mark) mark.classList.add(data.health_tier === "watch" ? "ping-watch" : "ping-intervene");
        }
        renderHealthPanelExtras(); updateAnalysisStatusTicker();
    } catch (err) { console.error("Error checking study health:", err); if (throwOnError) throw err; }
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

window.renderHealthPanelExtras = renderHealthPanelExtras;
window.updateAnalysisStatusTicker = updateAnalysisStatusTicker;
window.checkStudyHealth = checkStudyHealth;
window.updateStatConfidenceBanner = updateStatConfidenceBanner;
window.syncTelemetryFromTrials = syncTelemetryFromTrials;
window.setWorkerRunIndicator = setWorkerRunIndicator;
