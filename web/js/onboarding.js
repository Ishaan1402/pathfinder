// web/js/onboarding.js — Handles manifest-based onboarding modal lifecycle and validation.

(function () {
    const DEFAULT_TEMPLATE = `study_name: segment_hpo
metrics:
  primary_score: dice
  objectives:
    - name: dice
      direction: maximize
      label: Dice Score
    - name: loss
      direction: minimize
      label: BCE Loss
params:
  - name: lr
    type: float_log
    min: 0.0001
    max: 0.1
  - name: batch_size
    type: categorical
    options: [16, 32, 64]
  - name: optimizer
    type: categorical
    options: ["adam", "sgd"]
  - name: num_epochs
    type: fixed
    value: 10
eval_protocol:
  enabled: true
  fixed_resolution: 512
  train_resolution_param: resolution
worker:
  entrypoint: python train.py --lr {lr} --batch_size {batch_size}
  env:
    CUDA_VISIBLE_DEVICES: "0"`;

    let validationTimeout = null;

    function showNewStudyModal() {
        const modal = document.getElementById("new-study-modal");
        const editor = document.getElementById("manifest-yaml-editor");
        const consoleEl = document.getElementById("manifest-validation-console");
        const forceCheck = document.getElementById("manifest-force-check");
        const submitBtn = document.getElementById("btn-submit-manifest");
        const fileInput = document.getElementById("manifest-file-input");

        if (fileInput) fileInput.value = "";
        if (forceCheck) forceCheck.checked = false;
        if (submitBtn) submitBtn.disabled = true;

        if (editor && (!editor.value || editor.value.trim() === "")) {
            editor.value = DEFAULT_TEMPLATE;
        }

        if (modal) {
            modal.classList.add("active");
            validateManifestOnClient();
        }
    }

    function closeNewStudyModal(event) {
        if (event && event.target.id === "new-study-modal") {
            closeNewStudyModalDirect();
        }
    }

    function closeNewStudyModalDirect() {
        const modal = document.getElementById("new-study-modal");
        if (modal) {
            modal.classList.remove("active");
        }
    }

    async function validateManifestOnClient() {
        const editor = document.getElementById("manifest-yaml-editor");
        const consoleEl = document.getElementById("manifest-validation-console");
        const submitBtn = document.getElementById("btn-submit-manifest");

        if (!editor || !consoleEl) return;

        const yamlContent = editor.value;
        if (!yamlContent || yamlContent.trim() === "") {
            consoleEl.innerHTML = `<div style="color: var(--text-muted); font-style: italic;">No manifest loaded yet. Drag-and-drop or select a file to begin.</div>`;
            if (submitBtn) submitBtn.disabled = true;
            return;
        }

        consoleEl.innerHTML = `<div style="color: var(--text-muted);"><span class="spinner-inline"></span> Running validation check...</div>`;

        try {
            const res = await fetch("/api/validate_manifest", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ yaml: yamlContent })
            });

            if (!res.ok) {
                const data = await res.json();
                throw new Error(data.detail || "Validation request failed");
            }

            const result = await res.json();

            consoleEl.innerHTML = "";

            if (!result.success && result.errors && result.errors.length > 0) {
                if (submitBtn) submitBtn.disabled = true;
                
                const errTitle = document.createElement("div");
                errTitle.style.color = "var(--status-failed, #ef4444)";
                errTitle.style.fontWeight = "bold";
                errTitle.style.marginBottom = "6px";
                errTitle.textContent = `✗ Validation Failed (${result.errors.length} error${result.errors.length > 1 ? 's' : ''}):`;
                consoleEl.appendChild(errTitle);

                result.errors.forEach(err => {
                    const line = document.createElement("div");
                    line.style.color = "#f87171";
                    line.style.paddingLeft = "12px";
                    line.style.textIndent = "-12px";
                    line.style.marginBottom = "4px";
                    line.textContent = `• ${err}`;
                    consoleEl.appendChild(line);
                });
            } else {
                if (submitBtn) submitBtn.disabled = false;
                
                const okLine = document.createElement("div");
                okLine.style.color = "var(--status-complete, #10b981)";
                okLine.style.fontWeight = "bold";
                okLine.style.marginBottom = "6px";
                okLine.textContent = `✓ Manifest validation passed! Ready to initialize.`;
                consoleEl.appendChild(okLine);
            }

            if (result.warnings && result.warnings.length > 0) {
                const warnTitle = document.createElement("div");
                warnTitle.style.color = "var(--status-pruned, #f59e0b)";
                warnTitle.style.fontWeight = "bold";
                warnTitle.style.marginTop = "10px";
                warnTitle.style.marginBottom = "6px";
                warnTitle.textContent = `⚠ Warnings (${result.warnings.length}):`;
                consoleEl.appendChild(warnTitle);

                result.warnings.forEach(warn => {
                    const line = document.createElement("div");
                    line.style.color = "#fbbf24";
                    line.style.paddingLeft = "12px";
                    line.style.textIndent = "-12px";
                    line.style.marginBottom = "4px";
                    line.textContent = `• ${warn}`;
                    consoleEl.appendChild(line);
                });
            }
        } catch (err) {
            if (submitBtn) submitBtn.disabled = true;
            consoleEl.innerHTML = `<div style="color: var(--status-failed, #ef4444); font-weight: bold;">✗ Validation Error:</div>
            <div style="color: #f87171; margin-top: 4px;">${err.message}</div>`;
        }
    }

    function debounceValidation() {
        if (validationTimeout) {
            clearTimeout(validationTimeout);
        }
        validationTimeout = setTimeout(validateManifestOnClient, 300);
    }

    async function submitManifest() {
        const editor = document.getElementById("manifest-yaml-editor");
        const consoleEl = document.getElementById("manifest-validation-console");
        const forceCheck = document.getElementById("manifest-force-check");
        const submitBtn = document.getElementById("btn-submit-manifest");

        if (!editor) return;

        const yamlContent = editor.value;
        const force = forceCheck ? forceCheck.checked : false;

        if (submitBtn) {
            submitBtn.disabled = true;
            submitBtn.textContent = "Initializing...";
        }

        consoleEl.innerHTML = `<div style="color: var(--accent-color); font-weight: bold;">Registering study...</div>`;

        try {
            const res = await fetch(`/api/init_from_manifest?force=${force}`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ yaml: yamlContent })
            });

            const result = await res.json();

            if (!res.ok) {
                throw new Error(result.detail || "Server failed to register the study.");
            }

            if (!result.success) {
                if (result.errors) {
                    throw new Error("Validation errors returned: " + result.errors.join("; "));
                } else {
                    throw new Error(result.message || "Initialization failed");
                }
            }

            consoleEl.innerHTML = `<div style="color: var(--status-complete, #10b981); font-weight: bold;">✓ Study '${result.study_name}' registered successfully!</div>`;
            
            // Refresh study list and switch to the new study
            if (window.populateStudyList) {
                await window.populateStudyList();
            }

            // Select and redirect to the new study
            const select = document.getElementById("study-select");
            if (select) {
                select.value = result.study_name;
                const url = new URL(window.location);
                url.searchParams.set('study', result.study_name);
                window.history.pushState({}, '', url);
                
                window.HPOState.session.studyName = result.study_name;
                window.HPOState.tables.dashboard = { sort: { col: null, dir: null }, filters: {} };
                window.HPOState.tables.analysis = { sort: { col: null, dir: null }, filters: {} };
                window.HPOState.render.lastDashboardHeaderSnapshot = "";
                window.HPOState.render.lastAnalysisHeaderSnapshot = "";

                if (window.fetchStudyDetails) window.fetchStudyDetails();
                if (window.fetchFanova) window.fetchFanova();
                if (window.fetchSearchSpace) window.fetchSearchSpace();
                if (window.fetchHpoConfig) window.fetchHpoConfig();
            }

            setTimeout(() => {
                closeNewStudyModalDirect();
                if (typeof window.showToast === 'function') {
                    window.showToast("Study registered successfully! Go to the 'Worker Setup' tab to run your model.");
                }
            }, 1000);

        } catch (err) {
            consoleEl.innerHTML = `<div style="color: var(--status-failed, #ef4444); font-weight: bold;">✗ Initialization Failed:</div>
            <div style="color: #f87171; margin-top: 4px;">${err.message}</div>`;
        } finally {
            if (submitBtn) {
                submitBtn.disabled = false;
                submitBtn.textContent = "Initialize Study";
            }
        }
    }

    // Set up drag and drop + file input listener once DOM loads
    document.addEventListener("DOMContentLoaded", () => {
        const uploadZone = document.getElementById("manifest-upload-zone");
        const fileInput = document.getElementById("manifest-file-input");
        const editor = document.getElementById("manifest-yaml-editor");

        if (uploadZone && fileInput) {
            uploadZone.addEventListener("click", () => {
                fileInput.click();
            });

            uploadZone.addEventListener("dragover", (e) => {
                e.preventDefault();
                uploadZone.classList.add("dragover");
            });

            uploadZone.addEventListener("dragleave", (e) => {
                e.preventDefault();
                uploadZone.classList.remove("dragover");
            });

            uploadZone.addEventListener("drop", (e) => {
                e.preventDefault();
                uploadZone.classList.remove("dragover");
                if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
                    const file = e.dataTransfer.files[0];
                    readManifestFile(file);
                }
            });

            fileInput.addEventListener("change", (e) => {
                if (e.target.files && e.target.files.length > 0) {
                    const file = e.target.files[0];
                    readManifestFile(file);
                }
            });
        }

        if (editor) {
            editor.addEventListener("input", debounceValidation);
        }

        function readManifestFile(file) {
            const reader = new FileReader();
            reader.onload = (event) => {
                if (editor) {
                    editor.value = event.target.result;
                    validateManifestOnClient();
                }
            };
            reader.readAsText(file);
        }
    });

    // Expose functions globally
    window.showNewStudyModal = showNewStudyModal;
    window.closeNewStudyModal = closeNewStudyModal;
    window.closeNewStudyModalDirect = closeNewStudyModalDirect;
    window.submitManifest = submitManifest;
    window.validateManifestOnClient = validateManifestOnClient;
})();
