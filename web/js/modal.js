function openTrialDetails(trialNum) {
    const trial = window.HPOState.data.trials.find(t => t.number === trialNum);
    if (!trial) return;

    const titleEl = document.getElementById("modal-trial-title");
    if (titleEl) titleEl.textContent = `Trial ${trial.number} Inspector [ID: ${trial.trial_id}]`;

    const pGrid = document.getElementById("modal-params");
    if (pGrid) {
        pGrid.replaceChildren();
        const displayParams = trial.params_display || trial.params;
        Object.entries(displayParams).forEach(([k, v]) => {
            const displayVal = typeof v === "number" ? (v % 1 === 0 ? String(v) : v.toExponential(2)) : String(v);
            const item = document.createElement("div");
            item.className = "param-item";
            const label = document.createElement("div");
            label.className = "param-label";
            label.textContent = k.replace(/_/g, " ");
            const value = document.createElement("div");
            value.className = "param-value";
            value.textContent = displayVal;
            item.appendChild(label);
            item.appendChild(value);
            pGrid.appendChild(item);
        });
        if (trial.dice_eval_fixed != null) {
            const item = document.createElement("div");
            item.className = "param-item";
            const label = document.createElement("div");
            label.className = "param-label";
            label.textContent = window.HPOState.data.hpoConfig?.eval_protocol?.dice_fixed_label || "Dice (eval)";
            const value = document.createElement("div");
            value.className = "param-value";
            value.textContent = trial.dice_eval_fixed.toFixed(4);
            item.appendChild(label);
            item.appendChild(value);
            pGrid.appendChild(item);
        }
        let ledClass = "complete";
        let stateLabel = trial.state;

        if (trial.state === "COMPLETE") {
            ledClass = "complete";
            stateLabel = "DONE";
        } else if (trial.oom_triggered || trial.failure_tag === "OOM" || trial.state === "FAIL") {
            ledClass = "fail";
            stateLabel = (trial.oom_triggered || trial.failure_tag === "OOM") ? "FAILED (OOM)" : "FAILED";
        } else if (trial.state === "PRUNED") {
            ledClass = "pruned";
            stateLabel = "PRUNED";
        } else if (trial.state === "RUNNING") {
            ledClass = "running";
            stateLabel = "RUNNING";
        }

        const statusItem = document.createElement("div");
        statusItem.className = `param-item status-param-item status-cell-${ledClass}`;
        const statusLabel = document.createElement("div");
        statusLabel.className = "param-label";
        statusLabel.textContent = "State";
        const statusValue = document.createElement("div");
        statusValue.className = "param-value status-cell-wrapper";
        statusValue.style.marginTop = "4px";
        const led = document.createElement("span");
        led.className = `led-light ${ledClass}`;
        const statusText = document.createElement("span");
        statusText.className = "status-text";
        statusText.textContent = stateLabel;
        statusValue.appendChild(led);
        statusValue.appendChild(statusText);
        statusItem.appendChild(statusLabel);
        statusItem.appendChild(statusValue);
        pGrid.appendChild(statusItem);
    }

    const oomSection = document.getElementById("modal-oom-section");
    const oomDetail = document.getElementById("modal-oom-detail");
    const isOom = trial.oom_triggered || trial.failure_tag === 'OOM';
    if (oomSection && oomDetail) {
        if (isOom) {
            oomSection.style.display = "block";
            let oomText = "This trial crashed with an out-of-memory error before completing.";
            if (trial.gpu_model) oomText += ` GPU: ${trial.gpu_model}.`;
            if (trial.max_vram_gb != null) oomText += ` Peak VRAM: ${trial.max_vram_gb.toFixed(1)} GB.`;
            oomText += " TPE will automatically steer future suggestions away from this configuration.";
            oomDetail.innerText = oomText;
        } else {
            oomSection.style.display = "none";
        }
    }

    // Metric validation health warning
    const warnSection = document.getElementById("modal-health-warning-section");
    const warnDetail = document.getElementById("modal-health-warning-detail");
    if (warnSection && warnDetail) {
        if (trial.health_reason) {
            warnSection.style.display = "block";
            warnDetail.textContent = trial.health_reason;
        } else {
            warnSection.style.display = "none";
        }
    }

    // Environment info section
    const envSection = document.getElementById("modal-env-section");
    const envFields = ["python_version", "platform", "hostname", "git_commit", "cuda_version", "dataset_version", "pip_freeze"];
    const hasEnvData = envFields.some(f => trial[f] !== undefined && trial[f] !== null && trial[f] !== "");
    if (envSection) {
        if (hasEnvData) {
            envSection.style.display = "block";
            document.getElementById("modal-env-hostname").textContent = trial.hostname || "—";
            document.getElementById("modal-env-platform").textContent = trial.platform || "—";
            document.getElementById("modal-env-python").textContent = trial.python_version || "—";
            document.getElementById("modal-env-cuda").textContent = trial.cuda_version || "—";
            document.getElementById("modal-env-git").textContent = trial.git_commit || "—";
            document.getElementById("modal-env-dataset").textContent = trial.dataset_version || "—";
            
            const pipTextarea = document.getElementById("modal-env-pip");
            if (pipTextarea) {
                pipTextarea.value = trial.pip_freeze || "No pip freeze recorded.";
                pipTextarea.parentElement.style.display = trial.pip_freeze ? "flex" : "none";
            }
            
            // Start collapsed
            const content = document.getElementById("modal-env-content");
            const chevron = document.getElementById("modal-env-chevron");
            if (content) content.style.display = "none";
            if (chevron) chevron.style.transform = "rotate(180deg)";
        } else {
            envSection.style.display = "none";
        }
    }

    const reasoningEl = document.getElementById("modal-reasoning");
    const rationaleTitle = document.getElementById("modal-rationale-title");
    const logic = window.HPOState.data.thoughtLogs.find(l => l.trial_id === trial.trial_id);
    if (reasoningEl && rationaleTitle) {
        if (logic) {
            rationaleTitle.innerText = "Coordinator Rationale";
            reasoningEl.innerText = `"${logic.predicted_outcome_rationale}"`;
        } else {
            rationaleTitle.innerText = "TPE Suggestion";
            reasoningEl.innerText = `"Optuna TPE sampled this configuration based on the observed score landscape. No coordinator override was active for this trial."`;
        }
    }

    window.HPOState.ui.isModalOpen = true;
    document.getElementById("detail-modal")?.classList.add("active");
    requestAnimationFrame(() => window.updateModalChart(trial.history));
}

function closeModal(event) {
    if (event.target.id === "detail-modal") {
        closeModalDirect();
    }
}

function closeModalDirect() {
    document.getElementById("detail-modal")?.classList.remove("active");
    window.HPOState.ui.isModalOpen = false;
    if (window.HPOState.render.pendingRender) {
        window.HPOState.render.pendingRender = false;
        window.renderStudyDetails(window.HPOState.data.latestStudyData);
    }
}

function toggleModalEnvCard() {
    const content = document.getElementById("modal-env-content");
    const chevron = document.getElementById("modal-env-chevron");
    if (!content || !chevron) return;
    
    const isHidden = content.style.display === "none";
    if (isHidden) {
        content.style.display = "flex";
        chevron.style.transform = "rotate(0deg)";
    } else {
        content.style.display = "none";
        chevron.style.transform = "rotate(180deg)";
    }
}

window.openTrialDetails = openTrialDetails;
window.closeModal = closeModal;
window.closeModalDirect = closeModalDirect;
window.toggleModalEnvCard = toggleModalEnvCard;
