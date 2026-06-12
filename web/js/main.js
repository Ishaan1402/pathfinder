document.addEventListener("DOMContentLoaded", () => {
    window.initHpoMarks();
    window.syncTopPerformersFilterCheckboxes();
    
    const savedAccentName = localStorage.getItem("hpo_accent_name") || "cyan";
    const savedAccentColor = localStorage.getItem("hpo_accent_color") || "#06b6d4";
    const savedAccentGlow = localStorage.getItem("hpo_accent_glow") || "rgba(6, 182, 212, 0.3)";
    window.changeAccent(savedAccentName, savedAccentColor, savedAccentGlow);

    const savedStatusStyle = localStorage.getItem("hpo_status_style") || "full";
    window.setStatusStyle(savedStatusStyle);

    window.fetchSearchSpace();
    window.fetchHpoConfig();
    
    const reviewPill = document.getElementById("dashboard-review-pill");
    if (reviewPill) {
        reviewPill.addEventListener("click", window.copyReviewPrompt);
        reviewPill.addEventListener("keydown", (e) => {
            if (e.key === "Enter" || e.key === " ") {
                e.preventDefault();
                window.copyReviewPrompt();
            }
        });
    }
    const reviewCloseBtn = document.getElementById("dashboard-review-close");
    if (reviewCloseBtn) {
        reviewCloseBtn.addEventListener("click", (e) => {
            e.stopPropagation();
            window.dismissReviewPill();
        });
    }
    window.populateStudyList();
    const studySelect = document.getElementById("study-select");
    if (studySelect) {
        studySelect.addEventListener("change", (e) => {
            const val = e.target.value;
            const url = new URL(window.location);
            url.searchParams.set('study', val);
            window.history.pushState({}, '', url);
            window.HPOState.session.studyName = val;
            window.HPOState.tables.dashboard = { sort: { col: null, dir: null }, filters: {} };
            window.HPOState.tables.analysis = { sort: { col: null, dir: null }, filters: {} };
            window.HPOState.render.lastDashboardHeaderSnapshot = "";
            window.HPOState.render.lastAnalysisHeaderSnapshot = "";
            window.fetchStudyDetails();
            window.fetchFanova();
            if (typeof window.updateColabSnippet === 'function') {
                window.updateColabSnippet();
            }
        });
    }
    fetch("/api/tunnel_url")
        .then(res => res.json())
        .then(data => {
            if (data.success && data.url) {
                const input = document.getElementById("ngrok-url-input");
                if (input) {
                    input.value = data.url;
                    window.updateColabSnippet();
                }
            }
        })
        .catch(err => console.error("Error fetching tunnel URL:", err));

    const scoreLabelInput = document.getElementById("eval-metric-score-label");
    if (scoreLabelInput) {
        scoreLabelInput.addEventListener("input", (e) => {
            const val = e.target.value.trim() || "Score";
            const trainLabelInput = document.getElementById("eval-dice-train-label");
            const fixedLabelInput = document.getElementById("eval-dice-fixed-label");
            if (trainLabelInput) trainLabelInput.placeholder = `${val} (train)`;
            if (fixedLabelInput) fixedLabelInput.placeholder = `${val} (eval)`;
        });
    }

    window.pollData();
    window.handleRouting();
});

async function pollData() {
    if (window.HPOState.poll.timeoutId) {
        clearTimeout(window.HPOState.poll.timeoutId);
        window.HPOState.poll.timeoutId = null;
    }
    try {
        await Promise.all([
            fetchStudyDetails(true),
            fetchPendingChanges(true),
            checkStudyHealth(true)
        ]);
        
        window.HPOState.poll.failures = 0;
        const offlineBanner = document.getElementById("offline-banner");
        if (offlineBanner) {
            offlineBanner.classList.add("hidden");
        }
        
        window.HPOState.poll.timeoutId = setTimeout(pollData, 2000);
    } catch (err) {
        console.warn("Poll attempt failed:", err);
        window.HPOState.poll.failures++;
        
        const delayMs = Math.min(2000 * Math.pow(2, window.HPOState.poll.failures - 1), 32000);
        const delayS = Math.round(delayMs / 1000);
        
        const offlineBanner = document.getElementById("offline-banner");
        const backoffTimer = document.getElementById("offline-backoff-timer");
        if (offlineBanner) {
            offlineBanner.classList.remove("hidden");
        }
        if (backoffTimer) {
            backoffTimer.textContent = delayS;
        }
        
        window.HPOState.poll.timeoutId = setTimeout(pollData, delayMs);
    }
}

function initHpoMarks() {}
function bindHpoMarkTap() {}
function setHpoMarkMode() {}
function updateHpoMarkModes() {}

// Window exports
window.pollData = pollData;
window.initHpoMarks = initHpoMarks;
window.bindHpoMarkTap = bindHpoMarkTap;
window.setHpoMarkMode = setHpoMarkMode;
window.updateHpoMarkModes = updateHpoMarkModes;

window.addEventListener("hashchange", () => window.handleRouting());
