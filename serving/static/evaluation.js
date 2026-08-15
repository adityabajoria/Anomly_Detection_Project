import {
    getResults,
    getEvaluationTrace,
} from "./api.js";


const LABELS = {
    random:
        "Random Baseline",

    zscore:
        "Z-Score",

    pca:
        "PCA",

    iforest:
        "Isolation Forest",

    lstm_autoencoder:
        "LSTM Autoencoder",
};


const ORDER = [
    "random",
    "zscore",
    "pca",
    "iforest",
    "lstm_autoencoder",
];


let machineId = null;

let results = {};

let selectedDetector = null;


/* ============================================================
   Formatting
============================================================ */

function detectorLabel(name) {
    return LABELS[name] || name;
}


function formatMetric(
    value,
    digits = 3
) {
    if (
        value === null ||
        value === undefined ||
        !Number.isFinite(
            Number(value)
        )
    ) {
        return "—";
    }

    return Number(value)
        .toFixed(digits);
}


function formatThreshold(value) {
    if (
        value === null ||
        value === undefined ||
        !Number.isFinite(
            Number(value)
        )
    ) {
        return "—";
    }


    const number =
        Number(value);


    if (
        Math.abs(number) > 0 &&
        Math.abs(number) < 0.001
    ) {
        return number.toExponential(3);
    }


    return number.toFixed(4);
}


function formatSeconds(value) {
    if (
        value === null ||
        value === undefined ||
        !Number.isFinite(
            Number(value)
        )
    ) {
        return "—";
    }


    const number =
        Number(value);


    if (number < 0.001) {
        return `${
            (
                number * 1000
            ).toFixed(3)
        } ms`;
    }


    return `${
        number.toFixed(4)
    } s`;
}


function formatThroughput(value) {
    if (
        value === null ||
        value === undefined ||
        !Number.isFinite(
            Number(value)
        )
    ) {
        return "—";
    }


    return `${
        Math.round(
            Number(value)
        ).toLocaleString()
    } pts/s`;
}


function metric(
    result,
    group,
    key
) {
    return (
        result?.[group]?.[key] ??
        null
    );
}


/* ============================================================
   Available Detectors
============================================================ */

function getAvailableDetectors() {
    return Object.keys(results)
        .filter(name =>
            ORDER.includes(name)
        )
        .sort(
            (a, b) =>
                ORDER.indexOf(a) -
                ORDER.indexOf(b)
        );
}


/* ============================================================
   Detector Selection
============================================================ */

async function selectDetector(
    detector
) {
    selectedDetector =
        detector;


    renderDetectorTabs();

    renderDetectorTable();

    renderSelectedDetector();


    await renderEvaluationChart();
}


/* ============================================================
   Detector Selector
============================================================ */

function renderDetectorTabs() {
    const container =
        document.getElementById(
            "evaluation-detectors"
        );


    const detectors =
        getAvailableDetectors();


    container.innerHTML =
        detectors
            .map(detector => `
                <button
                    class="
                        detector-tab
                        ${
                            detector ===
                            selectedDetector
                                ? "active"
                                : ""
                        }
                    "
                    data-detector="${detector}"
                >
                    ${detectorLabel(detector)}
                </button>
            `)
            .join("");


    container
        .querySelectorAll(
            ".detector-tab"
        )
        .forEach(button => {

            button.onclick =
                async () => {

                    await selectDetector(
                        button.dataset.detector
                    );
                };
        });
}


/* ============================================================
   Registry Table
============================================================ */

function renderDetectorTable() {
    const tbody =
        document.getElementById(
            "evaluation-table-body"
        );


    const detectors =
        getAvailableDetectors();


    tbody.innerHTML =
        detectors
            .map(detector => {

                const result =
                    results[detector];


                const precision =
                    metric(
                        result,
                        "honest",
                        "precision"
                    );


                const recall =
                    metric(
                        result,
                        "honest",
                        "recall"
                    );


                const f1 =
                    metric(
                        result,
                        "honest",
                        "f1"
                    );


                const prAuc =
                    result?.pr_auc ??
                    null;


                const threshold =
                    result?.threshold ??
                    null;


                const throughput =
                    result
                        ?.throughput_pts_per_sec ??
                    null;


                const selected =
                    detector ===
                    selectedDetector;


                return `
                    <tr
                        class="${
                            selected
                                ? "selected"
                                : ""
                        }"

                        data-detector="${detector}"
                    >

                        <td>
                            <span class="detector-name">
                                ${detectorLabel(detector)}
                            </span>
                        </td>


                        <td class="numeric">
                            ${formatMetric(precision)}
                        </td>


                        <td class="numeric">
                            ${formatMetric(recall)}
                        </td>


                        <td class="numeric metric-strong">
                            ${formatMetric(f1)}
                        </td>


                        <td class="numeric">
                            ${formatMetric(prAuc)}
                        </td>


                        <td class="numeric threshold-value">
                            ${formatThreshold(threshold)}
                        </td>


                        <td class="numeric throughput-value">
                            ${formatThroughput(throughput)}
                        </td>


                        <td>
                            <span class="artifact-ready">
                                Ready
                            </span>
                        </td>

                    </tr>
                `;
            })
            .join("");


    tbody
        .querySelectorAll("tr")
        .forEach(row => {

            row.onclick =
                async () => {

                    await selectDetector(
                        row.dataset.detector
                    );
                };
        });
}


/* ============================================================
   Selected Artifact
============================================================ */

function renderSelectedDetector() {
    if (!selectedDetector) {
        return;
    }


    const result =
        results[selectedDetector];


    if (!result) {
        return;
    }


    const precision =
        metric(
            result,
            "honest",
            "precision"
        );


    const recall =
        metric(
            result,
            "honest",
            "recall"
        );


    const honestF1 =
        metric(
            result,
            "honest",
            "f1"
        );


    const adjustedF1 =
        metric(
            result,
            "point_adjusted",
            "f1"
        );


    const prAuc =
        result?.pr_auc ??
        null;


    const threshold =
        result?.threshold ??
        null;


    const meanDelay =
        result
            ?.detection_delay
            ?.mean_delay ??
        null;


    const segmentsDetected =
        result
            ?.detection_delay
            ?.segments_detected ??
        null;


    const segmentsTotal =
        result
            ?.detection_delay
            ?.segments_total ??
        null;


    const fitSeconds =
        result?.fit_seconds ??
        null;


    const scoreSeconds =
        result?.score_seconds ??
        null;


    const throughput =
        result
            ?.throughput_pts_per_sec ??
        null;


    document.getElementById(
        "evaluation-selected-name"
    ).textContent =
        detectorLabel(
            selectedDetector
        );


    document.getElementById(
        "diag-status"
    ).textContent =
        "Ready for serving";


    document.getElementById(
        "diag-threshold"
    ).textContent =
        formatThreshold(
            threshold
        );


    document.getElementById(
        "diag-f1"
    ).textContent =
        formatMetric(
            honestF1
        );


    document.getElementById(
        "diag-precision"
    ).textContent =
        formatMetric(
            precision
        );


    document.getElementById(
        "diag-recall"
    ).textContent =
        formatMetric(
            recall
        );


    document.getElementById(
        "diag-pr-auc"
    ).textContent =
        formatMetric(
            prAuc
        );


    document.getElementById(
        "diag-delay"
    ).textContent =
        meanDelay == null
            ? "—"
            : `${
                Number(meanDelay)
                    .toFixed(2)
            } steps`;


    document.getElementById(
        "diag-segments"
    ).textContent =
        (
            segmentsDetected == null ||
            segmentsTotal == null
        )
            ? "—"
            : `${segmentsDetected} / ${segmentsTotal}`;


    document.getElementById(
        "diag-fit-time"
    ).textContent =
        formatSeconds(
            fitSeconds
        );


    document.getElementById(
        "diag-score-time"
    ).textContent =
        formatSeconds(
            scoreSeconds
        );


    document.getElementById(
        "diag-throughput"
    ).textContent =
        formatThroughput(
            throughput
        );


    document.getElementById(
        "diag-adjusted-f1"
    ).textContent =
        formatMetric(
            adjustedF1
        );


    const gapElement =
        document.getElementById(
            "diag-adjustment-gap"
        );


    if (
        honestF1 != null &&
        adjustedF1 != null
    ) {
        const gap =
            Number(adjustedF1) -
            Number(honestF1);


        gapElement.textContent =
            `+${gap.toFixed(3)}`;

    } else {

        gapElement.textContent =
            "—";
    }
}


/* ============================================================
   Ground-Truth Segments
============================================================ */

function buildGroundTruthShapes(
    labels
) {
    const shapes = [];

    let start = null;


    for (
        let i = 0;
        i < labels.length;
        i++
    ) {

        if (
            labels[i] === 1 &&
            start === null
        ) {
            start = i;
        }


        const segmentEnded =
            labels[i] === 0 &&
            start !== null;


        if (segmentEnded) {

            shapes.push({
                type: "rect",

                xref: "x",

                yref: "paper",

                x0: start,

                x1: i - 1,

                y0: 0,

                y1: 1,

                fillcolor:
                    "rgba(244, 63, 94, 0.08)",

                line: {
                    width: 0,
                },

                layer: "below",
            });


            start = null;
        }
    }


    if (start !== null) {

        shapes.push({
            type: "rect",

            xref: "x",

            yref: "paper",

            x0: start,

            x1:
                labels.length - 1,

            y0: 0,

            y1: 1,

            fillcolor:
                "rgba(244, 63, 94, 0.08)",

            line: {
                width: 0,
            },

            layer: "below",
        });
    }


    return shapes;
}


/* ============================================================
   Evaluation Chart
============================================================ */

async function renderEvaluationChart() {
    const chart =
        document.getElementById(
            "evaluation-chart"
        );


    const status =
        document.getElementById(
            "evaluation-chart-status"
        );


    const title =
        document.getElementById(
            "evaluation-chart-title"
        );


    if (
        !chart ||
        !selectedDetector
    ) {
        return;
    }


    title.textContent =
        `${detectorLabel(selectedDetector)} · evaluation replay`;


    status.textContent =
        "Loading evaluation trace…";


    try {

        const trace =
            await getEvaluationTrace(
                machineId,
                selectedDetector
            );


        const scores =
            trace.scores;


        const labels =
            trace.labels;


        const threshold =
            Number(
                trace.threshold
            );


        const timesteps =
            scores.map(
                (_, i) => i
            );


        const alertX = [];

        const alertY = [];


        for (
            let i = 0;
            i < scores.length;
            i++
        ) {

            if (
                Number.isFinite(
                    scores[i]
                ) &&
                scores[i] >= threshold
            ) {

                alertX.push(i);

                alertY.push(
                    scores[i]
                );
            }
        }


        const groundTruthShapes =
            buildGroundTruthShapes(
                labels
            );


        /*
         * Add the calibrated threshold on top of
         * the ground-truth region shapes.
         */

        const shapes = [
            ...groundTruthShapes,

            {
                type: "line",

                xref: "paper",

                x0: 0,

                x1: 1,

                yref: "y",

                y0: threshold,

                y1: threshold,

                line: {
                    color: "#64748b",
                    width: 1.2,
                    dash: "dot",
                },
            },
        ];


        const scoreTrace = {
            type: "scattergl",

            mode: "lines",

            x: timesteps,

            y: scores,

            name: "Score",

            line: {
                color: "#22d3ee",

                width: 1.2,
            },

            hovertemplate:
                "t=%{x}<br>" +
                "score=%{y:.6g}" +
                "<extra></extra>",
        };


        const alertTrace = {
            type: "scattergl",

            mode: "markers",

            x: alertX,

            y: alertY,

            name: "Predicted alert",

            marker: {
                color: "#f43f5e",

                size: 4,
            },

            hovertemplate:
                "ALERT<br>" +
                "t=%{x}<br>" +
                "score=%{y:.6g}" +
                "<extra></extra>",
        };


        const layout = {

            paper_bgcolor:
                "rgba(0,0,0,0)",

            plot_bgcolor:
                "rgba(0,0,0,0)",


            font: {
                family: "Inter",

                color: "#8b93a7",

                size: 10,
            },


            margin: {
                l: 64,

                r: 20,

                t: 20,

                b: 50,
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

                rangeslider: {
                    visible: false,
                },
            },


            yaxis: {

                title: {
                    text:
                        "raw anomaly score",

                    font: {
                        size: 10,

                        color: "#5f6676",
                    },
                },

                gridcolor: "#151820",

                zeroline: false,

                autorange: true,
            },


            shapes,

            showlegend: false,

            hovermode:
                "closest",
        };


        await Plotly.react(
            chart,

            [
                scoreTrace,
                alertTrace,
            ],

            layout,

            {
                displayModeBar: false,

                responsive: true,
            }
        );


        status.textContent =
            `${trace.n_points.toLocaleString()} points · ` +
            `threshold ${formatThreshold(threshold)}`;


    } catch (error) {

        status.textContent =
            "Unable to load evaluation trace";


        chart.innerHTML = `
            <div class="evaluation-chart-error">
                ${error.message}
            </div>
        `;
    }
}


/* ============================================================
   States
============================================================ */

function setLoading() {
    document.getElementById(
        "evaluation-table-body"
    ).innerHTML = `
        <tr>
            <td
                colspan="8"
                class="evaluation-empty"
            >
                Loading detector artifacts…
            </td>
        </tr>
    `;
}


function showEvaluationError(
    message
) {
    document.getElementById(
        "evaluation-table-body"
    ).innerHTML = `
        <tr>
            <td
                colspan="8"
                class="
                    evaluation-empty
                    evaluation-error
                "
            >
                ${message}
            </td>
        </tr>
    `;
}


/* ============================================================
   Public Init
============================================================ */

export async function initEvaluationView(
    selectedMachine
) {
    machineId =
        selectedMachine;


    document.getElementById(
        "evaluation-machine"
    ).textContent =
        machineId;


    setLoading();


    try {

        results =
            await getResults(
                machineId
            );


        const detectors =
            getAvailableDetectors();


        if (!detectors.length) {

            showEvaluationError(
                "No evaluation results are available for this machine."
            );

            return;
        }


        selectedDetector =
            detectors.includes("pca")
                ? "pca"
                : detectors[0];


        renderDetectorTabs();

        renderDetectorTable();

        renderSelectedDetector();


        await renderEvaluationChart();


    } catch (error) {

        showEvaluationError(
            `Unable to load evaluation results: ${error.message}`
        );
    }
}