import pytest
from pydantic import ValidationError

from gmv.buffer import BoundedQueue
from gmv.bus import EventBus
from gmv.models import AnalysisConfig, Status


def test_config_rejects_invalid_windows_and_bands():
    with pytest.raises(ValidationError):
        AnalysisConfig(window_ms=20, hop_ms=50)
    with pytest.raises(ValidationError):
        AnalysisConfig(band_edges=(1, 2, 2, 4, 5, 6, 7, 8))


def test_backpressure_and_close():
    q = BoundedQueue(2)
    for i in range(5):
        q.put(i)
    assert q.dropped == 3 and q.high_water == 2
    q.close()
    assert [q.get(), q.get(), q.get()] == [3, 4, None]
    assert not q.put(6)


def test_slow_subscriber_is_isolated():
    bus = EventBus()
    slow, fast = bus.subscribe(1), bus.subscribe(10)
    for i in range(4):
        bus.publish(
            Status(
                session_id="test",
                source="test",
                timestamp=i,
                detected_monotonic=0,
                state="test",
                detail="test",
            )
        )
    assert slow.dropped == 3 and fast.dropped == 0
    assert fast.get().sequence == 1 and slow.get().sequence == 4
    bus.unsubscribe(slow)
    assert slow.closed
