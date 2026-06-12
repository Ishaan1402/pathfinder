/* === Dashboard Data Export Functions === */

function exportTrialsToCsv() {
            if (!window.HPOState.data.trials || window.HPOState.data.trials.length === 0) {
                showToast("No trial data available to export.");
                return;
            }

            const ev = window.HPOState.data.hpoConfig?.eval_protocol || {};
            const activeSpace = window.HPOState.data.activeSearchSpace || {};
            const paramKeys = Object.keys(activeSpace).filter(k => !k.startsWith("_") && typeof activeSpace[k] === "object");
            paramKeys.sort();

            const headers = ["Trial", "State", "Loss", "Score"];
            if (ev.enabled) {
                headers.push("Eval Score");
            }
            paramKeys.forEach(k => headers.push(k));

            const rows = [headers.join(",")];

            window.HPOState.data.trials.forEach(t => {
                const row = [
                    `#${t.number}`,
                    t.state,
                    t.bce !== null && t.bce !== undefined ? t.bce.toFixed(6) : "",
                    t.dice !== null && t.dice !== undefined ? t.dice.toFixed(6) : ""
                ];
                if (ev.enabled) {
                    row.push(t.dice_eval_fixed !== null && t.dice_eval_fixed !== undefined ? t.dice_eval_fixed.toFixed(6) : "");
                }
                paramKeys.forEach(k => {
                    const val = t.params[k];
                    row.push(val !== undefined && val !== null ? val : "");
                });
                rows.push(row.map(v => `"${v}"`).join(","));
            });

            const csvContent = "data:text/csv;charset=utf-8," + rows.join("\n");
            const encodedUri = encodeURI(csvContent);
            const link = document.createElement("a");
            link.setAttribute("href", encodedUri);
            link.setAttribute("download", `${window.HPOState.session.studyName}_trials_${new Date().toISOString().slice(0,10)}.csv`);
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            showToast("CSV file download started.");
        }

async function exportParetoFront() {
            try {
                const res = await fetch(`/api/pareto_front?study_name=${window.HPOState.session.studyName}`);
                if (!res.ok) throw new Error("API error");
                const data = await res.json();
                
                if (!data.pareto_front || data.pareto_front.length === 0) {
                    showToast("No Pareto frontier trials available.");
                    return;
                }

                const activeSpace = window.HPOState.data.activeSearchSpace || {};
                const paramKeys = Object.keys(activeSpace).filter(k => !k.startsWith("_") && typeof activeSpace[k] === "object");
                paramKeys.sort();

                const headers = ["Trial", "Loss", "Score"];
                paramKeys.forEach(k => headers.push(k));

                const rows = [headers.join(",")];
                data.pareto_front.forEach(t => {
                    const row = [
                        `#${t.number}`,
                        t.bce !== null && t.bce !== undefined ? t.bce.toFixed(6) : "",
                        t.dice !== null && t.dice !== undefined ? t.dice.toFixed(6) : ""
                    ];
                    paramKeys.forEach(k => {
                        const val = t.params[k];
                        row.push(val !== undefined && val !== null ? val : "");
                    });
                    rows.push(row.map(v => `"${v}"`).join(","));
                });

                const csvContent = "data:text/csv;charset=utf-8," + rows.join("\n");
                const encodedUri = encodeURI(csvContent);
                const link = document.createElement("a");
                link.setAttribute("href", encodedUri);
                link.setAttribute("download", `${window.HPOState.session.studyName}_pareto_front_${new Date().toISOString().slice(0,10)}.csv`);
                document.body.appendChild(link);
                link.click();
                document.body.removeChild(link);
                showToast("Pareto frontier CSV download started.");
            } catch (err) {
                showToast("Error exporting Pareto Frontier: " + err.message);
            }
        }

function exportStudyConfig() {
            if (!window.HPOState.data.hpoConfig) {
                showToast("No study configuration available.");
                return;
            }
            const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(window.HPOState.data.hpoConfig, null, 2));
            const link = document.createElement("a");
            link.setAttribute("href", dataStr);
            link.setAttribute("download", `${window.HPOState.session.studyName}_config_${new Date().toISOString().slice(0,10)}.json`);
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            showToast("Study config JSON download started.");
        }

function exportFullTrialsJson() {
            if (!window.HPOState.data.trials || window.HPOState.data.trials.length === 0) {
                showToast("No trial data available.");
                return;
            }
            const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(window.HPOState.data.trials, null, 2));
            const link = document.createElement("a");
            link.setAttribute("href", dataStr);
            link.setAttribute("download", `${window.HPOState.session.studyName}_full_trials_${new Date().toISOString().slice(0,10)}.json`);
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            showToast("Full trials JSON download started.");
        }


// Window exports
window.exportTrialsToCsv = exportTrialsToCsv;
window.exportParetoFront = exportParetoFront;
window.exportStudyConfig = exportStudyConfig;
window.exportFullTrialsJson = exportFullTrialsJson;
