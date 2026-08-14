import {
    createSession,
    getDetectors,
    openSessionStream,
    resetThreshold,
    setThreshold,
    switchModel,
} from "./api.js";


const LABELS = {
    random: "Random Baseline",
    zscore: "Z-Score",
    pca: "PCA",
    iforest: "Isolation Forest",
    lstm_autoencoder: "LSTM Autoencoder",
};


const ORDER = [
    "random",
    "zscore",
    "pca",
    "iforest",
    "lstm_autoencoder",
];


const MAX_POINTS = 150;
const LATENCY_WINDOW = 100;


let machineId = null;
let sessionId = null;
let activeDetector = null;

let eventSource = null;
let running = false;

let currentTimestep = 0;
let eventsProcessed = 0;
let alertsTriggered = 0;

let activeThreshold = null;
let thresholdSource = null;

let recentLatencies = [];
let recentNormalizedScores = [];

let switchShapes = [];
let switchAnnotations = [];


/* ============================================================
   Utility
============================================================ */

function detectorLabel(name) {
    return LABELS[name] || name;
}


function showError(message) {
    const banner =
        document.getElementById("error-banner");

    banner.textContent = message;
    banner.style.display = "block";

    console.error(message);
}


function clearError() {
    const banner =
        document.getElementById("error-banner");

    banner.textContent = "";
    banner.style.display = "none";
}


function percentile(values, q) {
    if (!values.length) {
        return null;
    }

    const sorted = [...values]
        .sort((a, b) => a - b);

    const index =
        (sorted.length - 1) * q;

    const lower =
        Math.floor(index);

    const upper =
        Math.ceil(index);

    if (lower === upper) {
        return sorted[lower];
    }

    const weight =
        index - lower;

    return (
        sorted[lower] * (1 - weight) +
        sorted[upper] * weight
    );
}


function formatLatency(value) {
    if (value == null) {
        return "—";
    }

    if (value < 1) {
        return `${value.toFixed(3)} ms`;
    }

    return `${value.toFixed(2)} ms`;
}


function formatScore(value) {
    if (value == null) {
        return "—";
    }

    if (Math.abs(value) < 0.001) {
        return value.toExponential(3);
    }

    return value.toFixed(4);
}


function thresholdPolicyLabel(source) {
    switch (source) {
        case "score_file":
            return "Calibrated";
        case "results_json":
            return "Calibrated";
        case "p99_of_offline_scores":
            return "Training P99";
        case "manual_override":
            return "Manual override";
        default:
            return "Calibrated";
    }
}


/* ============================================================
   Chart
============================================================ */

function chartLayout() {
    return {
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(0,0,0,0)",

        font: {
            family: "Inter",
            color: "#8b93a7",
            size: 11,
        },

        margin: {
            l: 58,
            r: 20,
            t: 18,
            b: 45,
        },

        xaxis: {
            title: {
                text: "timestep",
                font: {
                    size: 10,
                    color: "#5f6676",
                },
            },

            gridcolor: "#151820",
            zeroline: false,
        },

        yaxis: {
            title: {
                text: "normalized anomaly score",
                font: {
                    size: 10,
                    color: "#5f6676",
                },
            },

            gridcolor: "#151820",
            zeroline: false,

            range: [
                0,
                1.35,
            ],
        },

        shapes: [
            {
                type: "line",

                xref: "paper",
                x0: 0,
                x1: 1,

                yref: "y",
                y0: 1,
                y1: 1,

                line: {
                    color: "#64748b",
                    width: 1,
                    dash: "dot",
                },
            },

            ...switchShapes,
        ],

        annotations: [
            ...switchAnnotations,
        ],

        showlegend: false,

        hovermode: "x unified",
    };
}


function initializeChart() {
    switchShapes = [];
    switchAnnotations = [];
    recentNormalizedScores = [];

    Plotly.newPlot(
        "chart",
        [
            {
                x: [],
                y: [],

                mode: "lines",

                name: "Normalized score",

                line: {
                    color: "#22d3ee",
                    width: 1.8,
                },

                hovertemplate:
                    "normalized: %{y:.3f}<extra></extra>",
            },

            {
                x: [],
                y: [],

                mode: "markers",

                name: "Alert",

                marker: {
                    color: "#f43f5e",
                    size: 7,
                },

                hovertemplate:
                    "ALERT<br>normalized: %{y:.3f}<extra></extra>",
            },
        ],
        chartLayout(),
        {
            displayModeBar: false,
            responsive: true,
        }
    );
}


function updateYAxis() {
    if (!recentNormalizedScores.length) {
        return;
    }

    const maxScore =
        Math.max(
            1,
            ...recentNormalizedScores
        );

    const upper =
        Math.max(
            1.25,
            maxScore * 1.15
        );

    Plotly.relayout(
        "chart",
        {
            "yaxis.range": [
                0,
                upper,
            ],
        }
    );
}


function addModelSwitchMarker(
    timestep,
    previous,
    next
) {
    switchShapes.push({
        type: "line",

        xref: "x",
        x0: timestep,
        x1: timestep,

        yref: "paper",
        y0: 0,
        y1: 1,

        line: {
            color: "#8b93a7",
            width: 1,
            dash: "dash",
        },
    });


    switchAnnotations.push({
        x: timestep,

        y: 1,

        xref: "x",
        yref: "paper",

        text:
            `${detectorLabel(previous)} → ${detectorLabel(next)}`,

        showarrow: false,

        yshift: 12,

        font: {
            size: 9,
            color: "#8b93a7",
        },

        bgcolor: "#0d0f14",

        bordercolor: "#1b1e27",
        borderwidth: 1,

        borderpad: 4,
    });


    Plotly.relayout(
        "chart",
        {
            shapes: [
                {
                    type: "line",

                    xref: "paper",
                    x0: 0,
                    x1: 1,

                    yref: "y",
                    y0: 1,
                    y1: 1,

                    line: {
                        color: "#64748b",
                        width: 1,
                        dash: "dot",
                    },
                },

                ...switchShapes,
            ],

            annotations:
                switchAnnotations,
        }
    );
}


/* ============================================================
   KPI / Service State
============================================================ */

function setConnectionState(connected) {
    const badge =
        document.getElementById("stream-badge");

    const liveLabel =
        document.getElementById("live-label");

    const streamKpi =
        document.getElementById("k-stream");

    const serviceStream =
        document.getElementById("service-stream");


    if (connected) {
        badge.classList.add("live");

        liveLabel.textContent =
            "Streaming";

        streamKpi.textContent =
            "LIVE";

        streamKpi.classList.add("live");

        serviceStream.textContent =
            "Connected";

        serviceStream.classList.add(
            "positive"
        );

    } else {
        badge.classList.remove("live");

        liveLabel.textContent =
            "Idle";

        streamKpi.textContent =
            "IDLE";

        streamKpi.classList.remove("live");

        serviceStream.textContent =
            "Disconnected";

        serviceStream.classList.remove(
            "positive"
        );
    }
}


function updateLatency(latencyMs) {
    if (latencyMs == null) {
        return;
    }


    recentLatencies.push(
        latencyMs
    );


    if (
        recentLatencies.length >
        LATENCY_WINDOW
    ) {
        recentLatencies.shift();
    }


    const p95 =
        percentile(
            recentLatencies,
            0.95
        );


    document.getElementById(
        "k-latency"
    ).textContent =
        formatLatency(p95);


    document.getElementById(
        "service-p95"
    ).textContent =
        formatLatency(p95);
}


function updateThresholdUI(
    threshold,
    source
) {
    activeThreshold =
        threshold;

    thresholdSource =
        source;


    document.getElementById(
        "service-threshold"
    ).textContent =
        threshold == null
            ? "—"
            : formatScore(threshold);


    document.getElementById(
        "service-threshold-policy"
    ).textContent =
        thresholdPolicyLabel(source);


    const input =
        document.getElementById(
            "threshold-input"
        );


    input.placeholder =
        threshold == null
            ? "Enter override"
            : formatScore(threshold);
}


/* ============================================================
   Event Log
============================================================ */

function clearEventLog() {
    document.getElementById(
        "event-list"
    ).innerHTML = `
        <div class="empty-state">
            Inference events will appear here when the stream starts.
        </div>
    `;
}


function addInferenceEvent(message) {
    const list =
        document.getElementById(
            "event-list"
        );


    if (
        list.querySelector(
            ".empty-state"
        )
    ) {
        list.innerHTML = "";
    }


    const row =
        document.createElement(
            "div"
        );


    if (message.warmup) {
        row.className =
            "event-row";

        row.innerHTML = `
            <span>t = ${message.t}</span>

            <span class="event-score">
                —
            </span>

            <span class="event-model">
                ${detectorLabel(message.detector)}
            </span>

            <span class="event-status warmup">
                WARM-UP
            </span>
        `;

    } else {
        row.className =
            message.is_alert
                ? "event-row alert-row"
                : "event-row";


        row.innerHTML = `
            <span>
                t = ${message.t}
            </span>

            <span class="event-score">
                ${formatScore(message.score)}
            </span>

            <span class="event-model">
                ${detectorLabel(message.detector)}
            </span>

            <span class="event-status ${
                message.is_alert
                    ? "anomaly"
                    : "normal"
            }">
                ${
                    message.is_alert
                        ? "ALERT"
                        : "NORMAL"
                }
            </span>
        `;
    }


    list.prepend(row);


    while (
        list.children.length > 10
    ) {
        list.removeChild(
            list.lastChild
        );
    }


    document.getElementById(
        "event-count"
    ).textContent =
        `${eventsProcessed.toLocaleString()} events`;
}


function addModelSwitchEvent(
    timestep,
    previous,
    next
) {
    const list =
        document.getElementById(
            "event-list"
        );


    if (
        list.querySelector(
            ".empty-state"
        )
    ) {
        list.innerHTML = "";
    }


    const row =
        document.createElement(
            "div"
        );


    row.className =
        "event-row switch-row";


    row.innerHTML = `
        <span>
            t = ${timestep}
        </span>

        <span class="event-score">
            —
        </span>

        <span class="event-model">
            ${detectorLabel(previous)}
            →
            ${detectorLabel(next)}
        </span>

        <span class="event-status switch">
            MODEL SWITCH
        </span>
    `;


    list.prepend(row);
}


/* ============================================================
   Stream Processing
============================================================ */

function handleStreamMessage(message) {
    if (message.meta) {
        currentTimestep =
            message.current_timestep || 0;

        return;
    }


    if (message.error) {
        showError(
            message.error
        );

        stopStream();

        return;
    }


    if (message.done) {
        running = false;

        setConnectionState(false);

        document.getElementById(
            "chart-status"
        ).textContent =
            `Replay complete · ${message.current_timestep.toLocaleString()} events`;

        return;
    }


    currentTimestep =
        message.t;


    eventsProcessed += 1;


    document.getElementById(
        "k-events"
    ).textContent =
        eventsProcessed.toLocaleString();


    updateThresholdUI(
        message.threshold,
        message.threshold_source
    );


    updateLatency(
        message.latency_ms
    );


    if (message.is_alert) {
        alertsTriggered += 1;

        document.getElementById(
            "k-alerts"
        ).textContent =
            alertsTriggered.toLocaleString();
    }


    addInferenceEvent(
        message
    );


    document.getElementById(
        "chart-status"
    ).textContent =
        `Processing t = ${message.t}`;


    if (
        message.score == null ||
        message.threshold == null ||
        message.threshold === 0
    ) {
        return;
    }


    /*
     * Important:
     *
     * Raw detector scores are not directly comparable.
     *
     * Normalizing by each detector's active threshold means
     * the decision boundary is always y = 1.
     */

    const normalized =
        message.score /
        message.threshold;


    recentNormalizedScores.push(
        normalized
    );


    if (
        recentNormalizedScores.length >
        MAX_POINTS
    ) {
        recentNormalizedScores.shift();
    }


    Plotly.extendTraces(
        "chart",
        {
            x: [
                [message.t],
                message.is_alert
                    ? [message.t]
                    : [],
            ],

            y: [
                [normalized],
                message.is_alert
                    ? [normalized]
                    : [],
            ],
        },
        [0, 1],
        MAX_POINTS
    );


    if (message.t > MAX_POINTS) {
        Plotly.relayout(
            "chart",
            {
                "xaxis.range": [
                    message.t - MAX_POINTS,
                    message.t + 5,
                ],
            }
        );
    }


    updateYAxis();
}


/* ============================================================
   Session / Stream
============================================================ */

async function createLiveSession() {
    stopStream();

    recentLatencies = [];
    recentNormalizedScores = [];

    eventsProcessed = 0;
    alertsTriggered = 0;
    currentTimestep = 0;

    activeThreshold = null;

    clearEventLog();
    initializeChart();


    document.getElementById(
        "k-events"
    ).textContent = "0";

    document.getElementById(
        "k-alerts"
    ).textContent = "0";

    document.getElementById(
        "k-latency"
    ).textContent = "—";


    const result =
        await createSession(
            machineId,
            activeDetector
        );


    sessionId =
        result.session_id;


    document.getElementById(
        "service-machine"
    ).textContent =
        machineId;


    document.getElementById(
        "service-model"
    ).textContent =
        detectorLabel(activeDetector);


    startStream();
}


function startStream() {
    if (!sessionId) {
        return;
    }


    stopStream();


    clearError();


    running = true;


    eventSource =
        openSessionStream(
            sessionId
        );


    setConnectionState(true);


    const button =
        document.getElementById(
            "play-btn"
        );


    button.textContent = "Ⅱ";
    button.classList.add("running");


    eventSource.onmessage =
        event => {
            const message =
                JSON.parse(
                    event.data
                );

            handleStreamMessage(
                message
            );
        };


    eventSource.onerror =
        () => {
            if (!running) {
                return;
            }

            showError(
                "Stream disconnected — check the uvicorn server logs."
            );

            stopStream();
        };
}


function stopStream() {
    running = false;


    if (eventSource) {
        eventSource.close();

        eventSource = null;
    }


    setConnectionState(false);


    const button =
        document.getElementById(
            "play-btn"
        );


    if (button) {
        button.textContent = "▶";
        button.classList.remove(
            "running"
        );
    }
}


/* ============================================================
   Model Hot-Swap
============================================================ */

async function handleModelSwitch(
    nextDetector
) {
    if (
        !sessionId ||
        nextDetector === activeDetector
    ) {
        return;
    }


    clearError();


    const previous =
        activeDetector;


    const result =
        await switchModel(
            sessionId,
            nextDetector
        );


    activeDetector =
        result.active_detector;


    const switchTimestep =
        result.current_timestep;


    document.getElementById(
        "service-model"
    ).textContent =
        detectorLabel(activeDetector);


    document.getElementById(
        "chart-title"
    ).textContent =
        `${detectorLabel(activeDetector)} · live inference`;


    /*
     * Threshold will be automatically replaced by the
     * new detector's calibrated threshold on its next SSE event.
     */

    updateThresholdUI(
        null,
        null
    );


    addModelSwitchMarker(
        switchTimestep,
        previous,
        activeDetector
    );


    addModelSwitchEvent(
        switchTimestep,
        previous,
        activeDetector
    );
}


/* ============================================================
   Manual Threshold Override
============================================================ */

async function applyThresholdOverride() {
    if (
        !sessionId ||
        !activeDetector
    ) {
        return;
    }


    const input =
        document.getElementById(
            "threshold-input"
        );


    const value =
        Number(input.value);


    if (
        !Number.isFinite(value) ||
        value <= 0
    ) {
        showError(
            "Threshold override must be a positive number."
        );

        return;
    }


    clearError();


    await setThreshold(
        sessionId,
        activeDetector,
        value
    );


    input.value = "";


    updateThresholdUI(
        value,
        "manual_override"
    );
}


async function resetThresholdOverride() {
    if (
        !sessionId ||
        !activeDetector
    ) {
        return;
    }


    const result =
        await resetThreshold(
            sessionId,
            activeDetector
        );


    updateThresholdUI(
        result.threshold,
        result.source
    );


    document.getElementById(
        "threshold-input"
    ).value = "";
}


/* ============================================================
   Public Init
============================================================ */

export async function initLiveView(
    selectedMachine
) {
    machineId =
        selectedMachine;


    const response =
        await getDetectors(
            machineId
        );


    const detectors =
        response.detectors;


    detectors.sort(
        (a, b) => {
            const ai =
                ORDER.indexOf(a);

            const bi =
                ORDER.indexOf(b);

            return (
                (ai === -1 ? 99 : ai) -
                (bi === -1 ? 99 : bi)
            );
        }
    );


    const select =
        document.getElementById(
            "model-select"
        );


    select.innerHTML =
        detectors
            .map(detector => `
                <option value="${detector}">
                    ${detectorLabel(detector)}
                </option>
            `)
            .join("");


    activeDetector =
        detectors.includes("pca")
            ? "pca"
            : detectors[0];


    select.value =
        activeDetector;


    select.onchange =
        async () => {
            try {
                await handleModelSwitch(
                    select.value
                );
            } catch (error) {
                showError(
                    error.message
                );

                select.value =
                    activeDetector;
            }
        };


    document.getElementById(
        "service-machine"
    ).textContent =
        machineId;


    document.getElementById(
        "service-model"
    ).textContent =
        detectorLabel(activeDetector);


    document.getElementById(
        "chart-title"
    ).textContent =
        `${detectorLabel(activeDetector)} · live inference`;


    initializeChart();


    await createLiveSession();
}


/* ============================================================
   Controls
============================================================ */

document.getElementById(
    "play-btn"
).onclick =
    () => {
        if (running) {
            stopStream();
        } else {
            startStream();
        }
    };


document.getElementById(
    "restart-btn"
).onclick =
    async () => {
        try {
            await createLiveSession();
        } catch (error) {
            showError(
                error.message
            );
        }
    };


document.getElementById(
    "threshold-apply"
).onclick =
    async () => {
        try {
            await applyThresholdOverride();
        } catch (error) {
            showError(
                error.message
            );
        }
    };


document.getElementById(
    "threshold-reset"
).onclick =
    async () => {
        try {
            await resetThresholdOverride();
        } catch (error) {
            showError(
                error.message
            );
        }
    };