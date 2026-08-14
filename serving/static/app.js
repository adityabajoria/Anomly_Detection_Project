import {
    getMachines,
} from "./api.js";

import {
    initLiveView,
} from "./live.js";


let currentMachine = null;


async function init() {
    const response =
        await getMachines();


    if (
        !response.machines.length
    ) {
        throw new Error(
            "No machines available."
        );
    }


    const machineSelect =
        document.getElementById(
            "machine-select"
        );


    machineSelect.innerHTML =
        response.machines
            .map(machine => `
                <option value="${machine}">
                    ${machine}
                </option>
            `)
            .join("");


    currentMachine =
        response.machines[0];


    machineSelect.value =
        currentMachine;


    machineSelect.onchange =
        async () => {
            currentMachine =
                machineSelect.value;

            await initLiveView(
                currentMachine
            );
        };


    setupTabs();


    await initLiveView(
        currentMachine
    );
}


function setupTabs() {
    const buttons =
        document.querySelectorAll(
            ".top-tab"
        );


    buttons.forEach(
        button => {
            button.onclick =
                () => {
                    const target =
                        button.dataset.view;


                    buttons.forEach(
                        item =>
                            item.classList.remove(
                                "active"
                            )
                    );


                    button.classList.add(
                        "active"
                    );


                    document.getElementById(
                        "live-view"
                    ).classList.toggle(
                        "hidden",
                        target !== "live"
                    );


                    document.getElementById(
                        "evaluation-view"
                    ).classList.toggle(
                        "hidden",
                        target !== "evaluation"
                    );
                };
        }
    );
}


init().catch(
    error => {
        const banner =
            document.getElementById(
                "error-banner"
            );

        banner.textContent =
            `Initialization failed: ${error.message}`;

        banner.style.display =
            "block";
    }
);