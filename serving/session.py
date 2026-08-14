from collections import deque
from dataclasses import dataclass, field
from uuid import uuid4


@dataclass
class StreamSession:
    machine_id: str
    active_detector: str

    session_id: str = field(
        default_factory=lambda: uuid4().hex
    )

    current_timestep: int = 0

    # Optional manual threshold override per detector.
    threshold_overrides: dict[str, float] = field(
        default_factory=dict
    )

    # One rolling buffer per detector.
    # This matters for stateful models like the LSTM autoencoder.
    detector_buffers: dict[str, deque] = field(
        default_factory=dict
    )

    # Keep a history of hot-swaps so the frontend can
    # eventually draw model-switch markers on the graph.
    switch_history: list[dict] = field(
        default_factory=list
    )

    stopped: bool = False

    def switch_detector(self, detector_name: str):
        """
        Change the active detector without resetting the telemetry timeline.
        """

        if detector_name == self.active_detector:
            return

        previous = self.active_detector

        self.active_detector = detector_name

        self.switch_history.append({
            "t": self.current_timestep,
            "from": previous,
            "to": detector_name,
        })

    def set_threshold(
        self,
        detector_name: str,
        threshold: float
    ):
        """
        Store a manual threshold override for a detector.
        """

        self.threshold_overrides[
            detector_name
        ] = float(threshold)

    def clear_threshold(
        self,
        detector_name: str
    ):
        """
        Return a detector to its calibrated/model threshold.
        """

        self.threshold_overrides.pop(
            detector_name,
            None
        )

    def get_threshold_override(
        self,
        detector_name: str
    ):
        return self.threshold_overrides.get(
            detector_name
        )

    def get_buffer(
        self,
        detector_name: str,
        maxlen: int
    ):
        """
        Return the persistent rolling context buffer for a detector.

        Stateless detectors will usually use maxlen=1.
        The LSTM uses its window_size.
        """

        if detector_name not in self.detector_buffers:
            self.detector_buffers[
                detector_name
            ] = deque(maxlen=maxlen)

        return self.detector_buffers[
            detector_name
        ]

    def advance(self):
        """
        Advance the global telemetry cursor by one timestep.
        """

        self.current_timestep += 1

    def reset(self):
        """
        Reset the entire telemetry session.

        This is intentionally different from switching models.
        Model switching must never reset current_timestep.
        """

        self.current_timestep = 0

        self.detector_buffers.clear()

        self.switch_history.clear()

        self.stopped = False
