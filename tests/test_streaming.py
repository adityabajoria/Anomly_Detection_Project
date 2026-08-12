import numpy as np
from detector.z_detector import ZScoreDetector

def test_zscore_streaming_matches_batch():
    """
    Streaming (score_one per timestep) must produce identical results to batch (score).
    """
    rng = np.random.default_rng(0)
    train = rng.normal(0, 1, (500, 10))
    test = rng.normal(0, 1, (300, 10))

    det = ZScoreDetector().fit(train)

    batch_scores = det.score(test)
    stream_scores = np.array([det.score_one(row) for row in test])

    assert np.allclose(stream_scores, batch_scores), "streaming path diverges from batch"