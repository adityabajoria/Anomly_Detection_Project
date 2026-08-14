import {
    getResults,
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


let machineId = null;
let results = {};
let selectedDetector = null;


/* ============================================================
   Helpers
============================================================ */

function detectorLabel(name) {
    return LABELS[name] || name;
}


function formatMetric(value, digits = 3) {
    if (
        value === null ||
        value === undefined ||
        !Number.isFinite(Number(value))
    ) {
        return "—";
    }

    return Number(value).toFixed(digits);
}


function formatThreshold(value) {
    if (
        value === null ||
        value === undefined ||
        !Number.isFinite(Number(value))
    ) {
        return "—";
    }

    const number = Number(value);

    if (
        Math.abs(number) > 0 &&
        Math.abs(number) < 0.001
    ) {
        return number.toExponential(3);
    }

    return number.toFixed(4);
}


function metric(result, group, key) {
    if (!result) {
        return null;
    }

    const section =
        result[group];

    if (!section) {
        return null;
    }

    return section[key] ?? null;
}


/* ============================================================
   Detector Table
============================================================ */

function renderDetectorTable() {
    const tbody =
        document.getElementById(
            "evaluation-table-body"
        );


    const detectors =
        Object.keys(results)
            .filter(name =>
                ORDER.includes(name)
            )
            .sort(
                (a, b) =>
                    ORDER.indexOf(a) -
                    ORDER.indexOf(b)
            );


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


                const threshold =
                    result?.threshold ??
                    null;


                const isSelected =
                    detector ===
                    selectedDetector;


                return `
                    <tr
                        class="${
                            isSelected
                                ? "selected"
                                : ""
                        }"
                        data-detector="${detector}"
                    >

                        <td>
                            <div class="detector-name">
                                ${detectorLabel(detector)}
                            </div>
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

                        <td class="numeric threshold-value">
                            ${formatThreshold(threshold)}
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

                    renderDetectorTable();

                    renderSelectedDetector();
                };
        });
}


/* ============================================================
   Selected Detector
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


    const threshold =
        result.threshold ??
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
        "diag-f1"
    ).textContent =
        formatMetric(
            honestF1
        );


    document.getElementById(
        "diag-adjusted-f1"
    ).textContent =
        formatMetric(
            adjustedF1
        );


    /*
     * Point-adjusted F1 is deliberately presented as
     * secondary diagnostic information rather than a
     * primary serving metric.
     */

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
   Detector Selector Pills
============================================================ */

function renderDetectorTabs() {
    const container =
        document.getElementById(
            "evaluation-detectors"
        );


    const detectors =
        Object.keys(results)
            .filter(name =>
                ORDER.includes(name)
            )
            .sort(
                (a, b) =>
                    ORDER.indexOf(a) -
                    ORDER.indexOf(b)
            );


    container.innerHTML =
        detectors
            .map(detector => `
                <button
                    class="detector-tab ${
                        detector ===
                        selectedDetector
                            ? "active"
                            : ""
                    }"
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
   Loading / Empty States
============================================================ */

function setLoading() {
    document.getElementById(
        "evaluation-table-body"
    ).innerHTML = `
        <tr>
            <td
                colspan="6"
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
                colspan="6"
                class="evaluation-empty evaluation-error"
            >
                ${message}
            </td>
        </tr>
    `;
}


/* ============================================================
   Public Initialization
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
            Object.keys(results)
                .filter(name =>
                    ORDER.includes(name)
                )
                .sort(
                    (a, b) =>
                        ORDER.indexOf(a) -
                        ORDER.indexOf(b)
                );


        if (!detectors.length) {
            showEvaluationError(
                "No detector evaluation results are available for this machine."
            );

            return;
        }


        /*
         * Prefer PCA as the initial artifact simply to keep
         * the initial state consistent with Live Inference.
         */

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