async function requestJSON(url, options = {}) {
    const response = await fetch(url, options);

    if (!response.ok) {
        let detail = "";

        try {
            const body = await response.json();
            detail = body.detail || "";
        } catch (_) {}

        throw new Error(
            `${response.status} ${response.statusText}` +
            (detail ? `: ${detail}` : "")
        );
    }

    return response.json();
}


export async function getMachines() {
    return requestJSON(
        "/api/machines"
    );
}


export async function getDetectors(machineId) {
    return requestJSON(
        `/api/detectors/${encodeURIComponent(machineId)}`
    );
}


export async function getResults(machineId) {
    return requestJSON(
        `/api/results/${encodeURIComponent(machineId)}`
    );
}


export async function createSession(
    machineId,
    activeDetector
) {
    return requestJSON(
        "/api/sessions",
        {
            method: "POST",

            headers: {
                "Content-Type":
                    "application/json",
            },

            body: JSON.stringify({
                machine_id:
                    machineId,

                active_detector:
                    activeDetector,
            }),
        }
    );
}


export async function getSession(
    sessionId
) {
    return requestJSON(
        `/api/sessions/${encodeURIComponent(sessionId)}`
    );
}


export async function switchModel(
    sessionId,
    detector
) {
    return requestJSON(
        `/api/sessions/${encodeURIComponent(sessionId)}/model`,
        {
            method: "POST",

            headers: {
                "Content-Type":
                    "application/json",
            },

            body: JSON.stringify({
                detector,
            }),
        }
    );
}


export async function setThreshold(
    sessionId,
    detector,
    threshold
) {
    return requestJSON(
        `/api/sessions/${encodeURIComponent(sessionId)}/threshold`,
        {
            method: "POST",

            headers: {
                "Content-Type":
                    "application/json",
            },

            body: JSON.stringify({
                detector,
                threshold,
            }),
        }
    );
}


export async function resetThreshold(
    sessionId,
    detector
) {
    return requestJSON(
        `/api/sessions/${encodeURIComponent(sessionId)}/threshold/${encodeURIComponent(detector)}`,
        {
            method: "DELETE",
        }
    );
}


export function openSessionStream(
    sessionId
) {
    return new EventSource(
        `/api/sessions/${encodeURIComponent(sessionId)}/stream?delay=0.02`
    );
}