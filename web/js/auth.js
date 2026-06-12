(function () {
    async function login() {
        const token = window.prompt("This Pathfinder dashboard is protected. Enter the access token:");
        if (!token) return false;
        const res = await fetch("/api/login", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ token }),
        });
        return res.ok;
    }
    const originalFetch = window.fetch.bind(window);
    window.fetch = async function (resource, config) {
        let res = await originalFetch(resource, config);
        if (res.status === 401 && window.HPO_AUTH_REQUIRED) {
            if (await login()) {
                res = await originalFetch(resource, config);
            }
        }
        return res;
    };
})();
