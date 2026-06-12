/* === Worker Setup & Colab Integration === */

function updateColabSnippet() {
    // Determine current host base URL
    const protocol = window.location.protocol;
    const host = window.location.host;
    let baseUrl = `${protocol}//${host}`;
    
    // Handle custom inputs
    const inputEl = document.getElementById("ngrok-url-input");
    const colabInputEl = document.getElementById("ngrok-url-input-colab");
    const customInput = inputEl ? inputEl.value.trim() : "";
    
    if (customInput) {
        baseUrl = customInput;
        if (baseUrl.endsWith('/')) {
            baseUrl = baseUrl.slice(0, -1);
        }
    } else {
        // Pre-fill the input with the current browser origin if empty
        if (inputEl) {
            inputEl.value = baseUrl;
        }
    }
    
    // Keep the Colab input field in sync
    if (colabInputEl && colabInputEl.value !== baseUrl) {
        colabInputEl.value = baseUrl;
    }
    
    // Toggle Colab warning if the active broker URL is localhost / 127.0.0.1 / [::1]
    const isLocalhost = baseUrl.includes("localhost") || baseUrl.includes("127.0.0.1") || baseUrl.includes("[::1]");
    const warningBanner = document.getElementById("colab-warning-banner");
    const warningUrlSpan = document.getElementById("colab-warning-url");
    
    if (warningBanner) {
        if (isLocalhost) {
            warningBanner.style.display = "block";
            if (warningUrlSpan) {
                warningUrlSpan.textContent = baseUrl;
            }
        } else {
            warningBanner.style.display = "none";
        }
    }
    
    const tokenLine = isLocalhost
        ? ""
        : `os.environ["HPO_SECRET_TOKEN"] = "paste-same-token-as-broker"\n`;

    // Render the Colab snippet text
    const snippet = `# 1. Install required packages
!pip install -q albumentations opencv-python optuna requests sqlalchemy

# 2. Download worker + client (token header required when broker uses --tunnel)
import requests
import os

broker_url = "${baseUrl}"
os.environ["HPO_BROKER_URL"] = broker_url
${tokenLine}
def _broker_headers():
    h = {}
    if "ngrok-free.app" in broker_url or "ngrok.io" in broker_url:
        h["ngrok-skip-browser-warning"] = "1"
    if os.environ.get("HPO_SECRET_TOKEN"):
        h["X-HPO-Token"] = os.environ["HPO_SECRET_TOKEN"]
    return h

for _name in ("hpo_client.py", "colab_worker.py"):
    _r = requests.get(f"{broker_url}/{_name}", headers=_broker_headers(), timeout=60)
    _r.raise_for_status()
    with open(_name, "w") as _f:
        _f.write(_r.text)

# 3. Import and run the reference worker loop
from colab_worker import train_colab_trial_loop
train_colab_trial_loop("${window.HPOState.session.studyName || 'bridge_crack_study'}", n_trials=12, epochs=15)`;
    
    const snippetTextEl = document.getElementById("colab-snippet-text");
    if (snippetTextEl) {
        snippetTextEl.textContent = snippet;
    }

    const customSnippet = `export HPO_BROKER_URL="${baseUrl}"
export HPO_STUDY_NAME="${window.HPOState.session.studyName}"
python worker_minimal.py`;

    const customSnippetEl = document.getElementById("custom-worker-snippet");
    if (customSnippetEl) {
        customSnippetEl.textContent = customSnippet;
    }
}

function syncColabUrl(val) {
    const mainInputEl = document.getElementById("ngrok-url-input");
    if (mainInputEl) {
        mainInputEl.value = val;
    }
    updateColabSnippet();
}

function copyColabSnippet(btn) {
    const snippetTextEl = document.getElementById("colab-snippet-text");
    if (!snippetTextEl) return;
    const snippetText = snippetTextEl.textContent;
    navigator.clipboard.writeText(snippetText).then(() => {
        const targetBtn = btn || document.querySelector("button[onclick^='copyColabSnippet']");
        if (targetBtn) {
            const originalText = targetBtn.textContent;
            targetBtn.textContent = "Copied!";
            setTimeout(() => {
                targetBtn.textContent = originalText;
            }, 2000);
        }
    }).catch(err => {
        console.error("Could not copy text: ", err);
    });
}

// Window exports
window.updateColabSnippet = updateColabSnippet;
window.syncColabUrl = syncColabUrl;
window.copyColabSnippet = copyColabSnippet;
