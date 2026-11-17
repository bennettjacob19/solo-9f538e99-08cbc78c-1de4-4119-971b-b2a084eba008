"""Unit tests for the per-device consecutive-anomaly state machine.

These drive the tracker directly on the event loop, controlling when each
admitted ticket's future resolves so the ordering / reset / version-isolation
guarantees are deterministic.
"""

from __future__ import annotations

import asyncio

import pytest

from app.alarm_tracker import DEFAULT_ALARM_LIMIT, DeviceAlarmTracker

pytestmark = pytest.mark.asyncio(loop_scope="function")

V1 = "v1"
V2 = "v2"
NORMAL = "pass"
OTHER = "dent"


async def _wait_settled(tracker, device_id, count):
    """Wait until ``count`` tickets have been applied or discarded."""
    deadline = asyncio.get_running_loop().time() + 5
    while asyncio.get_running_loop().time() < deadline:
        view = tracker.get_device(device_id)
        if (
            view is not None
            and view["total_applied"] + view["total_discarded"] >= count
        ):
            return view
        await asyncio.sleep(0.01)
    raise AssertionError("tickets did not settle in time")


async def test_admit_returns_none_without_device_or_label():
    tracker = DeviceAlarmTracker()
    assert tracker.admit("", V1, NORMAL) is None
    assert tracker.admit("d1", V1, "") is None  # type: ignore[arg-type]
    assert tracker.list_devices() == []
    await tracker.aclose()


async def test_three_consecutive_anomalies_then_alarm():
    tracker = DeviceAlarmTracker()
    for _ in range(3):
        a = tracker.admit("dev-1", V1, NORMAL)
        assert a is not None
        DeviceAlarmTracker.succeed(a, OTHER, 0.1)
    view = await _wait_settled(tracker, "dev-1", 3)
    assert view["version"] == V1
    assert view["normal_label"] == NORMAL
    assert view["consecutive_anomalies"] == 3
    assert view["alarm"] is True
    assert view["alarm_threshold"] == DEFAULT_ALARM_LIMIT
    assert view["last_outcome"] == "anomaly"
    assert view["last_label"] == OTHER
    assert view["pending"] == 0
    await tracker.aclose()


async def test_normal_result_resets_streak_and_clears_alarm():
    tracker = DeviceAlarmTracker()
    tickets = [tracker.admit("dev-1", V1, NORMAL) for _ in range(3)]
    for a in tickets[:2]:
        DeviceAlarmTracker.succeed(a, OTHER, 0.1)
    DeviceAlarmTracker.succeed(tickets[2], NORMAL, 0.9)
    view = await _wait_settled(tracker, "dev-1", 3)
    assert view["consecutive_anomalies"] == 0
    assert view["alarm"] is False
    assert view["last_outcome"] == "normal"

    # Two anomalies alone must not alarm; a third would.
    for _ in range(2):
        a = tracker.admit("dev-1", V1, NORMAL)
        DeviceAlarmTracker.succeed(a, OTHER, 0.1)
    view = await _wait_settled(tracker, "dev-1", 5)
    assert view["consecutive_anomalies"] == 2
    assert view["alarm"] is False
    await tracker.aclose()


async def test_computation_failure_resets_streak():
    tracker = DeviceAlarmTracker()
    a1 = tracker.admit("dev-1", V1, NORMAL)
    a2 = tracker.admit("dev-1", V1, NORMAL)
    DeviceAlarmTracker.succeed(a1, OTHER, 0.1)
    DeviceAlarmTracker.fail(a2, RuntimeError("worker exploded"))
    view = await _wait_settled(tracker, "dev-1", 2)
    assert view["consecutive_anomalies"] == 0
    assert view["alarm"] is False
    assert view["last_outcome"] == "error"
    assert view["last_label"] is None

    # After the reset the next anomaly starts again from one.
    a3 = tracker.admit("dev-1", V1, NORMAL)
    DeviceAlarmTracker.succeed(a3, OTHER, 0.1)
    view = await _wait_settled(tracker, "dev-1", 3)
    assert view["consecutive_anomalies"] == 1
    assert view["alarm"] is False
    await tracker.aclose()


async def test_completion_order_cannot_overtake_admission_order():
    tracker = DeviceAlarmTracker()
    first = tracker.admit("dev-1", V1, NORMAL)
    second = tracker.admit("dev-1", V1, NORMAL)
    third = tracker.admit("dev-1", V1, NORMAL)

    # The two later-admitted tickets finish first; the first ticket is slow.
    DeviceAlarmTracker.succeed(second, OTHER, 0.1)
    DeviceAlarmTracker.succeed(third, OTHER, 0.1)
    await asyncio.sleep(0.05)
    blocked = tracker.get_device("dev-1")
    # Nothing is applied while the predecessor is unsettled.
    assert blocked["consecutive_anomalies"] == 0
    assert blocked["pending"] == 3
    assert blocked["total_applied"] == 0

    # The slow first ticket returns a *normal* result: it must reset the
    # streak before the two anomalies behind it are applied, so no alarm.
    DeviceAlarmTracker.succeed(first, NORMAL, 0.9)
    view = await _wait_settled(tracker, "dev-1", 3)
    assert view["total_applied"] == 3
    assert view["consecutive_anomalies"] == 2
    assert view["alarm"] is False
    assert view["last_outcome"] == "anomaly"
    assert view["pending"] == 0
    await tracker.aclose()


async def test_admission_order_raises_alarm_when_three_anomalies_apply():
    tracker = DeviceAlarmTracker()
    tickets = [tracker.admit("dev-1", V1, NORMAL) for _ in range(3)]
    # Resolve out of order on purpose.
    DeviceAlarmTracker.succeed(tickets[2], OTHER, 0.1)
    DeviceAlarmTracker.succeed(tickets[1], OTHER, 0.1)
    DeviceAlarmTracker.succeed(tickets[0], OTHER, 0.1)
    view = await _wait_settled(tracker, "dev-1", 3)
    assert view["consecutive_anomalies"] == 3
    assert view["alarm"] is True
    await tracker.aclose()


async def test_version_activation_clears_alarm_and_generation():
    tracker = DeviceAlarmTracker()
    tickets = [tracker.admit("dev-1", V1, NORMAL) for _ in range(3)]
    for a in tickets:
        DeviceAlarmTracker.succeed(a, OTHER, 0.1)
    view = await _wait_settled(tracker, "dev-1", 3)
    assert view["alarm"] is True
    gen_before = view["generation"]

    tracker.activate_version("v2", "ok")
    view = tracker.get_device("dev-1")
    assert view["version"] == "v2"
    assert view["normal_label"] == "ok"
    assert view["alarm"] is False
    assert view["consecutive_anomalies"] == 0
    assert view["generation"] == gen_before + 1
    await tracker.aclose()


async def test_late_old_version_results_are_discarded_after_swap():
    tracker = DeviceAlarmTracker()
    old = tracker.admit("dev-1", V1, NORMAL)
    # The old-version call is still running when the swap happens.
    tracker.activate_version(V2, NORMAL)
    fresh = tracker.admit("dev-1", V2, NORMAL)

    # First the old ticket settles (e.g. long after a 504), then the new one.
    DeviceAlarmTracker.succeed(old, OTHER, 0.1)
    DeviceAlarmTracker.succeed(fresh, OTHER, 0.1)
    view = await _wait_settled(tracker, "dev-1", 2)
    assert view["version"] == V2
    assert view["total_discarded"] == 1
    assert view["total_applied"] == 1
    # The discarded anomaly does not count against the new version.
    assert view["consecutive_anomalies"] == 1
    assert view["alarm"] is False
    await tracker.aclose()


async def test_old_version_results_cannot_trigger_new_version_alarm():
    tracker = DeviceAlarmTracker()
    old_tickets = [tracker.admit("dev-1", V1, NORMAL) for _ in range(3)]
    tracker.activate_version(V2, NORMAL)
    for a in old_tickets:  # all settle late, after the swap
        DeviceAlarmTracker.succeed(a, OTHER, 0.1)
    view = await _wait_settled(tracker, "dev-1", 3)
    assert view["version"] == "v2"
    assert view["consecutive_anomalies"] == 0
    assert view["alarm"] is False
    assert view["total_discarded"] == 3
    await tracker.aclose()


async def test_devices_are_independent():
    tracker = DeviceAlarmTracker()
    for device in ("d1", "d2"):
        for _ in range(3):
            a = tracker.admit(device, V1, NORMAL)
            DeviceAlarmTracker.succeed(a, OTHER, 0.1)
    # d3 sends a normal result only.
    a = tracker.admit("d3", V1, NORMAL)
    DeviceAlarmTracker.succeed(a, NORMAL, 0.9)

    await _wait_settled(tracker, "d1", 3)
    await _wait_settled(tracker, "d2", 3)
    await _wait_settled(tracker, "d3", 1)
    listed = {d["device_id"]: d for d in tracker.list_devices()}
    assert sorted(listed) == ["d1", "d2", "d3"]
    assert listed["d1"]["alarm"] is True
    assert listed["d2"]["alarm"] is True
    assert listed["d3"]["alarm"] is False
    assert listed["d3"]["consecutive_anomalies"] == 0

    assert tracker.get_device("unknown") is None
    await tracker.aclose()


async def test_custom_alarm_limit():
    tracker = DeviceAlarmTracker(alarm_limit=2)
    a1 = tracker.admit("dev-1", V1, NORMAL)
    DeviceAlarmTracker.succeed(a1, OTHER, 0.1)
    await _wait_settled(tracker, "dev-1", 1)
    assert tracker.get_device("dev-1")["alarm"] is False
    a2 = tracker.admit("dev-1", V1, NORMAL)
    DeviceAlarmTracker.succeed(a2, OTHER, 0.1)
    view = await _wait_settled(tracker, "dev-1", 2)
    assert view["consecutive_anomalies"] == 2
    assert view["alarm"] is True
    assert view["alarm_threshold"] == 2
    await tracker.aclose()


async def test_many_concurrent_inference_settle_in_admission_order():
    tracker = DeviceAlarmTracker()
    n = 50
    tickets = [tracker.admit("dev-1", V1, NORMAL) for _ in range(n)]

    async def resolve_later(a, delay, label):
        await asyncio.sleep(delay)
        DeviceAlarmTracker.succeed(a, label, 0.1)

    # Reverse-ish completion order via staggered delays.
    await asyncio.gather(
        *[
            resolve_later(a, (n - i) * 0.002, OTHER if i % 4 else NORMAL)
            for i, a in enumerate(tickets)
        ]
    )
    view = await _wait_settled(tracker, "dev-1", n)
    assert view["total_applied"] == n
    assert view["pending"] == 0
    # Streak must reflect the admission sequence (labels at indices 0,4,8...
    # are normal), never the completion order.
    expected_streak = 0
    for i in range(n):
        expected_streak = 0 if i % 4 == 0 else expected_streak + 1
    assert view["consecutive_anomalies"] == expected_streak
    await tracker.aclose()
