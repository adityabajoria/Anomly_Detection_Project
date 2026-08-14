import {
    getResults,
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
   Detector Selector
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
                () => {

                    selectedDetector =
                        button.dataset.detector;


                    renderDetectorTabs();

                    renderDetectorTable();

                    renderSelectedDetector();
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
                () => {

                    selectedDetector =
                        row.dataset.detector;


                    renderDetectorTabs();

                    renderDetectorTable();

                    renderSelectedDetector();
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


function showEvaluationError(message) {
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


    } catch (error) {

        showEvaluationError(
            `Unable to load evaluation results: ${error.message}`
        );
    }
}