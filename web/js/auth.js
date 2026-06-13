(function () {
    const originalFetch = window.fetch.bind(window);
    let currentLoginPromise = null;
    let authCancelled = false;

    async function login() {
        if (authCancelled) {
            return false;
        }
        if (currentLoginPromise) {
            return currentLoginPromise;
        }

        currentLoginPromise = (async () => {
            try {
                const token = window.prompt("This Pathfinder dashboard is protected. Enter the access token:");
                if (token === null) {
                    // User clicked Cancel. Stop prompting automatically to prevent annoying loops.
                    authCancelled = true;
                    return false;
                }
                if (!token.trim()) {
                    return false;
                }
                // Call originalFetch directly to bypass the interception middleware
                const res = await originalFetch("/api/login", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ token: token.trim() }),
                });
                if (res.ok) {
                    return true;
                } else {
                    alert("Invalid token. Please check your token and refresh/try again.");
                    // Set authCancelled to true so they aren't immediately spammed by other poll requests.
                    authCancelled = true;
                    return false;
                }
            } catch (err) {
                console.error("Login request failed:", err);
                return false;
            } finally {
                currentLoginPromise = null;
            }
        })();

        return currentLoginPromise;
    }

    window.fetch = async function (resource, config) {
        let res = await originalFetch(resource, config);
        
        // Extract URL to see if it's the login endpoint
        const url = typeof resource === "string" ? resource : (resource?.url || "");
        const isLoginEndpoint = url.includes("/api/login");

        if (res.status === 401 && window.HPO_AUTH_REQUIRED && !isLoginEndpoint) {
            const success = await login();
            if (success) {
                // Retry the original request
                res = await originalFetch(resource, config);
            }
        }
        return res;
    };

    // Auto-login from query parameter '?token=...'
    const urlParams = new URLSearchParams(window.location.search);
    const urlToken = urlParams.get('token');
    if (urlToken) {
        // Strip the token from the URL search parameters immediately to prevent it from leaking in history/referrers
        const cleanUrl = new URL(window.location.href);
        cleanUrl.searchParams.delete('token');
        window.history.replaceState({}, document.title, cleanUrl.toString());

        originalFetch("/api/login", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ token: urlToken.trim() }),
        }).then(res => {
            if (res.ok) {
                // Reload the page to retry all initial API calls with the session cookie set
                window.location.reload();
            } else {
                console.error("Auto-login token was invalid.");
            }
        }).catch(err => {
            console.error("Auto-login failed:", err);
        });
    }
})();
