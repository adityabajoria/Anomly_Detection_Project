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


const WINDOW_SIZE = 150;


const DARK = {

    paper_bgcolor:
        "rgba(0,0,0,0)",

    plot_bgcolor:
        "rgba(0,0,0,0)",

    font: {
        color: "#8b93a7",
        family: "Inter",
        size: 11,
    },

    margin: {
        l: 55,
        r: 20,
        t: 18,
        b: 45,
    },
};


let current = null;
let machine = null;

let evt = null;
let streaming = false;

let modelThreshold = null;
let activeThreshold = null;

let manualThreshold = false;

let eventsProcessed = 0;
let alertsTriggered = 0;

let eventLogCount = 0;


/* ============================================================
   Errors
============================================================ */

function showError(message) {

    const element =
        document.getElementById(
            "error-banner"
        );

    element.textContent =
        message;

    element.style.display =
        "block";

    console.error(message);
}


function clearError() {

    const element =
        document.getElementById(
            "error-banner"
        );

    element.style.display =
        "none";

    element.textContent =
        "";
}


/* ============================================================
   HTTP
============================================================ */

async function getJSON(url) {

    const response =
        await fetch(url);

    if (!response.ok) {

        let detail = "";

        try {

            const body =
                await response.json();

            detail =
                body.detail || "";

        } catch (_) {}


        throw new Error(
            `${url} → HTTP ${response.status}` +
            `${detail ? ": " + detail : ""}`
        );
    }

    return response.json();
}


/* ============================================================
   Initialization
============================================================ */

async function init() {

    const { machines } =
        await getJSON(
            "/api/machines"
        );


    if (!machines.length) {

        throw new Error(
            "No machines found."
        );
    }


    const machineSelect =
        document.getElementById(
            "machine-select"
        );


    machineSelect.innerHTML =
        machines
            .map(machineName => (
                `<option value="${machineName}">
                    ${machineName}
                </option>`
            ))
            .join("");


    machineSelect.onchange = () => {

        machine =
            machineSelect.value;

        loadMachine()
            .catch(error =>
                showError(
                    error.message
                )
            );
    };


    machine =
        machines[0];


    await loadMachine();
}


/* ============================================================
   Machine / Detector
============================================================ */

async function loadMachine() {

    stopStream();
    clearError();


    const detectorSelect =
        document.getElementById(
            "detector-select"
        );


    const response =
        await getJSON(
            `/api/detectors/${machine}`
        );


    const detectors =
        response.detectors;


    detectors.sort((a, b) => {

        const aIndex =
            ORDER.indexOf(a);

        const bIndex =
            ORDER.indexOf(b);


        return (
            (aIndex === -1 ? 99 : aIndex) -
            (bIndex === -1 ? 99 : bIndex)
        );
    });


    detectorSelect.innerHTML =
        detectors
            .map(detector => (
                `<option value="${detector}">
                    ${LABELS[detector] || detector}
                </option>`
            ))
            .join("");


    detectorSelect.onchange =
        () => {
            selectDetector(
                detectorSelect.value
            );
        };


    const firstDetector =
        detectors.includes("pca")
            ? "pca"
            : detectors[0];


    detectorSelect.value =
        firstDetector;


    selectDetector(
        firstDetector
    );
}


/* ============================================================
   Detector Selection
============================================================ */

function selectDetector(detector) {

    current =
        detector;


    manualThreshold =
        false;


    modelThreshold =
        null;


    activeThreshold =
        null;


    stopStream();

    resetDashboard();

    resetChart();


    const label =
        LABELS[detector] ||
        detector;


    document.getElementById(
        "view-title"
    ).textContent =
        `${label} anomaly score`;


    document.getElementById(
        "service-machine"
    ).textContent =
        machine;


    document.getElementById(
        "service-detector"
    ).textContent =
        label;
}


/* ============================================================
   Dashboard Reset
============================================================ */

function resetDashboard() {

    eventsProcessed = 0;

    alertsTriggered = 0;

    eventLogCount = 0;


    document.getElementById(
        "k-events"
    ).textContent =
        "0";


    document.getElementById(
        "k-alerts"
    ).textContent =
        "0";


    document.getElementById(
        "k-latency"
    ).textContent =
        "—";


    document.getElementById(
        "event-count"
    ).textContent =
        "0 events";


    document.getElementById(
        "event-list"
    ).innerHTML = `
        <div class="empty-state">
            Inference events will appear here when the stream starts.
        </div>
    `;


    setLive(false);
}


/* ============================================================
   Live State
============================================================ */

function setLive(isLive) {

    const badge =
        document.getElementById(
            "stream-badge"
        );

    const label =
        document.getElementById(
            "live-label"
        );

    const statusValue =
        document.getElementById(
            "k-stream"
        );

    const serviceStream =
        document.getElementById(
            "service-stream"
        );


    if (isLive) {

        badge.classList.add(
            "live"
        );

        label.textContent =
            "Streaming";

        statusValue.textContent =
            "LIVE";

        statusValue.classList.add(
            "live"
        );

        serviceStream.textContent =
            "Connected";

    } else {

        badge.classList.remove(
            "live"
        );

        label.textContent =
            "Idle";

        statusValue.textContent =
            "IDLE";

        statusValue.classList.remove(
            "live"
        );

        serviceStream.textContent =
            "Disconnected";
    }
}


/* ============================================================
   Chart
============================================================ */

function getThresholdShape() {

    if (activeThreshold === null) {
        return [];
    }


    return [
        {
            type: "line",

            xref: "paper",

            x0: 0,
            x1: 1,

            yref: "y",

            y0: activeThreshold,
            y1: activeThreshold,

            line: {
                color: "#64748b",
                width: 1,
                dash: "dot",
            },
        },
    ];
}


function baseLayout() {

    return {

        ...DARK,

        height:
            410,

        xaxis: {

            title: {
                text: "timestep",

                font: {
                    size: 10,
                    color: "#5f6676",
                },
            },

            gridcolor:
                "#151820",

            zeroline:
                false,

            tickfont: {
                size: 10,
            },
        },


        yaxis: {

            title: {
                text: "anomaly score",

                font: {
                    size: 10,
                    color: "#5f6676",
                },
            },

            gridcolor:
                "#151820",

            zeroline:
                false,

            tickfont: {
                size: 10,
            },
        },


        shapes:
            getThresholdShape(),

        showlegend:
            false,

        hovermode:
            "x unified",
    };
}


function resetChart() {

    Plotly.newPlot(

        "chart",

        [

            {
                x: [],
                y: [],

                mode:
                    "lines",

                name:
                    "Anomaly score",

                line: {
                    color:
                        "#22d3ee",

                    width:
                        1.7,
                },

                hovertemplate:
                    "score: %{y:.4f}<extra></extra>",
            },


            {
                x: [],
                y: [],

                mode:
                    "markers",

                name:
                    "Alert",

                marker: {
                    color:
                        "#f43f5e",

                    size:
                        7,
                },

                hovertemplate:
                    "ALERT<br>score: %{y:.4f}<extra></extra>",
            },

        ],

        baseLayout(),

        {
            displayModeBar:
                false,

            responsive:
                true,
        }
    );


    document.getElementById(
        "status"
    ).textContent =
        "Waiting for stream";
}


/* ============================================================
   Rolling Window
============================================================ */

function updateRollingWindow(timestep) {

    if (timestep < WINDOW_SIZE) {
        return;
    }


    const start =
        timestep -
        WINDOW_SIZE;


    Plotly.relayout(
        "chart",
        {
            "xaxis.range":
                [
                    start,
                    timestep + 5
                ],
        }
    );
}


/* ============================================================
   Event Log
============================================================ */

function addInferenceEvent(
    timestep,
    score,
    isAlert
) {

    const list =
        document.getElementById(
            "event-list"
        );


    eventLogCount += 1;


    if (eventLogCount === 1) {
        list.innerHTML = "";
    }


    const row =
        document.createElement(
            "div"
        );


    row.className =
        isAlert
            ? "event-row alert-row"
            : "event-row";


    row.innerHTML = `

        <span>
            t = ${timestep}
        </span>

        <span class="event-score">
            ${score.toFixed(4)}
        </span>

        <span class="event-status ${
            isAlert
                ? "anomaly"
                : "normal"
        }">
            ${
                isAlert
                    ? "ANOMALY"
                    : "NORMAL"
            }
        </span>
    `;


    list.prepend(row);


    while (
        list.children.length > 8
    ) {

        list.removeChild(
            list.lastChild
        );
    }


    document.getElementById(
        "event-count"
    ).textContent =
        `${eventLogCount.toLocaleString()} events`;
}


/* ============================================================
   Threshold
============================================================ */

function updateThresholdUI() {

    const input =
        document.getElementById(
            "threshold-input"
        );

    const defaultLabel =
        document.getElementById(
            "threshold-default"
        );


    if (activeThreshold !== null) {

        input.value =
            activeThreshold.toFixed(4);
    }


    if (modelThreshold !== null) {

        defaultLabel.textContent =
            `Model default: ${modelThreshold.toFixed(4)}`;

    } else {

        defaultLabel.textContent =
            "Model default: —";
    }
}


function setManualThreshold() {

    const input =
        document.getElementById(
            "threshold-input"
        );


    const value =
        Number(input.value);


    if (
        Number.isNaN(value)
    ) {

        showError(
            "Threshold must be a valid number."
        );

        return;
    }


    clearError();


    activeThreshold =
        value;


    manualThreshold =
        true;


    Plotly.relayout(
        "chart",
        {
            shapes:
                getThresholdShape(),
        }
    );


    /*
     * Restart so all alerts in this replay use
     * one consistent threshold.
     */

    startStream();
}


function resetThreshold() {

    if (
        modelThreshold === null
    ) {

        return;
    }


    activeThreshold =
        modelThreshold;


    manualThreshold =
        false;


    updateThresholdUI();


    Plotly.relayout(
        "chart",
        {
            shapes:
                getThresholdShape(),
        }
    );


    startStream();
}


/* ============================================================
   Stop Stream
============================================================ */

function stopStream() {

    if (evt) {

        evt.close();

        evt = null;
    }


    streaming =
        false;


    const playButton =
        document.getElementById(
            "play-btn"
        );


    playButton.innerHTML =
        "▶";


    playButton.classList.remove(
        "running"
    );


    playButton.title =
        "Start stream";


    setLive(false);
}


/* ============================================================
   Start Stream
============================================================ */

function startStream() {

    stopStream();

    resetDashboard();

    resetChart();

    clearError();


    streaming =
        true;


    const playButton =
        document.getElementById(
            "play-btn"
        );


    playButton.innerHTML =
        "Ⅱ";


    playButton.classList.add(
        "running"
    );


    playButton.title =
        "Stop stream";


    document.getElementById(
        "restart-btn"
    ).disabled =
        false;


    setLive(true);


    const delay =
        document.getElementById(
            "speed-select"
        ).value;


    const buffer = {

        x: [],
        y: [],

        flaggedX: [],
        flaggedY: [],
    };


    let lastTimestep = 0;

    let done = false;


    function flush() {

        if (
            !streaming &&
            !done
        ) {
            return;
        }


        if (
            buffer.x.length
        ) {

            Plotly.extendTraces(

                "chart",

                {
                    x: [
                        buffer.x,
                        buffer.flaggedX,
                    ],

                    y: [
                        buffer.y,
                        buffer.flaggedY,
                    ],
                },

                [0, 1],

                WINDOW_SIZE
            );


            buffer.x = [];
            buffer.y = [];

            buffer.flaggedX = [];
            buffer.flaggedY = [];


            updateRollingWindow(
                lastTimestep
            );


            document.getElementById(
                "status"
            ).textContent =
                done
                    ? `Replay complete · ${lastTimestep + 1} events`
                    : `Processing t = ${lastTimestep}`;
        }


        if (!done) {

            requestAnimationFrame(
                flush
            );
        }
    }


    evt =
        new EventSource(
            `/api/stream/${machine}/${current}?delay=${delay}`
        );


    evt.onmessage =
        event => {

            const message =
                JSON.parse(
                    event.data
                );


            /*
             * Stream metadata.
             */

            if (
                message.meta
            ) {

                modelThreshold =
                    message.threshold;


                /*
                 * Only overwrite the active threshold
                 * if the user has not manually selected one.
                 */

                if (
                    !manualThreshold
                ) {

                    activeThreshold =
                        modelThreshold;
                }


                updateThresholdUI();


                document.getElementById(
                    "alert-footer"
                ).textContent =
                    `${message.n_segments} true anomaly segments`;


                Plotly.relayout(
                    "chart",
                    {
                        shapes:
                            getThresholdShape(),
                    }
                );


                requestAnimationFrame(
                    flush
                );

                return;
            }


            if (
                message.error
            ) {

                stopStream();


                showError(
                    `Stream error: ${message.error}`
                );


                return;
            }


            if (
                message.done
            ) {

                done = true;

                streaming = false;


                stopStream();

                flush();

                return;
            }


            lastTimestep =
                message.t;


            eventsProcessed += 1;


            document.getElementById(
                "k-events"
            ).textContent =
                eventsProcessed.toLocaleString();


            /*
             * LSTM warm-up can emit null.
             */

            if (
                message.score === null
            ) {

                return;
            }


            const isAlert =
                activeThreshold !== null &&
                message.score >=
                    activeThreshold;


            buffer.x.push(
                message.t
            );


            buffer.y.push(
                message.score
            );


            if (isAlert) {

                buffer.flaggedX.push(
                    message.t
                );


                buffer.flaggedY.push(
                    message.score
                );


                alertsTriggered += 1;


                document.getElementById(
                    "k-alerts"
                ).textContent =
                    alertsTriggered.toLocaleString();
            }


            addInferenceEvent(
                message.t,
                message.score,
                isAlert
            );
        };


    evt.onerror =
        () => {

            stopStream();


            if (!done) {

                showError(
                    "Stream disconnected — check the uvicorn server logs."
                );
            }
        };
}


/* ============================================================
   Controls
============================================================ */

document.getElementById(
    "play-btn"
).onclick =
    () => {

        if (streaming) {

            stopStream();

        } else {

            startStream();
        }
    };


document.getElementById(
    "restart-btn"
).onclick =
    () => {

        startStream();
    };


document.getElementById(
    "threshold-apply"
).onclick =
    () => {

        setManualThreshold();
    };


document.getElementById(
    "threshold-reset"
).onclick =
    () => {

        resetThreshold();
    };


document.getElementById(
    "threshold-input"
).addEventListener(
    "keydown",
    event => {

        if (
            event.key === "Enter"
        ) {

            setManualThreshold();
        }
    }
);


/* ============================================================
   Startup
============================================================ */

init()
    .catch(error =>
        showError(
            "Initialization failed: " +
            error.message
        )
    );