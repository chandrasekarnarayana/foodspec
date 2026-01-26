from __future__ import annotations

import numpy as np

from foodspec.monitoring import StreamingMonitor


def test_streaming_monitor_updates():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(5, 20))
    monitor = StreamingMonitor()
    event = monitor.update(X)
    assert event.health_mean >= 0.0
    assert len(monitor.history) == 1
