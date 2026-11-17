"""Integration tests against a real API process and real worker processes.

These exercise the guarantees that only show up with real OS processes:
version pinning, serving during candidate load, load-failure fallback, the
two-process cap, and the fact that a client timeout does *not* free the slot
until the worker has actually finished.
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
import time
import uuid
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parent.parent


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class ApiServer:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.port = _free_port()
        self.proc: subprocess.Popen | None = None
        self.log = open(data_dir / "api.log", "w", encoding="utf-8")

    @property
    def base(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def __enter__(self) -> "ApiServer":
        env = {**os.environ, "DATA_DIR": str(self.data_dir), "LOG_LEVEL": "WARNING"}
        self.proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "app.main:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(self.port),
                "--log-level",
                "warning",
            ],
            cwd=str(ROOT),
            env=env,
            stdout=self.log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        deadline = time.time() + 30
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError("API exited during startup:\n" + self._tail())
            try:
                if httpx.get(f"{self.base}/healthz", timeout=1.0).status_code == 200:
                    return self
            except httpx.HTTPError:
                pass
            time.sleep(0.05)
        raise TimeoutError("API did not start:\n" + self._tail())

    def __exit__(self, *exc) -> None:
        if self.proc is not None and self.proc.poll() is None:
            os.killpg(os.getpgid(self.proc.pid), 15)
            try:
                self.proc.wait(10)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(self.proc.pid), 9)
                self.proc.wait(10)
        # Any model workers belong to their own process groups; the API stops
        # them on shutdown, but reap stragglers that could poison later tests.
        subprocess.run(
            ["pkill", "-f", f"{self.data_dir}/workers"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self.log.close()

    def _tail(self) -> str:
        self.log.flush()
        return self.log.name


@pytest.fixture(scope="module")
def api(tmp_path_factory):
    data_dir = tmp_path_factory.mktemp("qcdata")
    with ApiServer(data_dir) as server:
        yield server
    shutil.rmtree(data_dir, ignore_errors=True)


def artifact(version: str, weights, bias, labels, **extra):
    return {"version": version, "weights": weights, "bias": bias, "labels": labels, **extra}


W_A = [[1.0, 0.0], [0.0, 1.0]]
W_B = [[0.0, 1.0], [1.0, 0.0]]
LABELS = ["defect-A", "defect-B"]


def upload(api: ApiServer, payload, timeout=40.0):
    return httpx.post(f"{api.base}/admin/models", json=payload, timeout=timeout)


def infer(api: ApiServer, features, timeout_ms=None, slow_ms=0, timeout=40.0, device_id=None):
    body = {"features": features, "slow_ms": slow_ms}
    if timeout_ms is not None:
        body["timeout_ms"] = timeout_ms
    if device_id is not None:
        body["device_id"] = device_id
    return httpx.post(f"{api.base}/infer", json=body, timeout=timeout)


def status(api: ApiServer):
    return httpx.get(f"{api.base}/admin/status", timeout=5.0).json()


def alarms(api: ApiServer):
    return httpx.get(f"{api.base}/admin/device-alarms", timeout=5.0).json()


def alarm_view(api: ApiServer, device_id: str):
    return httpx.get(f"{api.base}/admin/device-alarms/{device_id}", timeout=5.0)


def wait_alarm_settled(
    api: ApiServer,
    device_id: str,
    admitted: int,
    timeout=15.0,
    require_empty_pending=True,
):
    """Wait until ``admitted`` tickets have all been applied or discarded.

    By default the device must also show no pending tickets. When other
    devices' slow tickets can still be in flight (e.g. right after a worker is
    killed), pass ``require_empty_pending=False`` and check the settled counts
    directly.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        resp = alarm_view(api, device_id)
        if resp.status_code == 200:
            view = resp.json()
            settled = view["total_applied"] + view["total_discarded"]
            if settled >= admitted and (
                not require_empty_pending or view["pending"] == 0
            ):
                return view
        time.sleep(0.05)
    raise AssertionError("alarm tickets did not settle in time")


# Two-feature model: argmax over [+x, -x] -> NORMAL_LABEL for positive x,
# NORM_OTHER for negative x, so a single-feature vector drives either label.
NORMAL_LABEL = "pass"
NORM_OTHER = "defect-A"
NORM_LABELS = [NORMAL_LABEL, NORM_OTHER]
W_NORM = [[1.0, 0.0], [-1.0, 0.0]]


def wait_for(predicate, timeout=10.0, interval=0.05):
    deadline = time.time() + timeout
    while time.time() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(interval)
    raise AssertionError("condition not met in time")


def slot_states(api: ApiServer):
    return [(s["slot_id"], s["version"], s["process_state"]) for s in status(api)["slots"]]


def test_01_initial_load_and_inference(api):
    resp = upload(api, artifact("v1", W_A, [0.0, 0.0], LABELS))
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["previous_version"] is None
    assert body["state"] == "active"

    result = infer(api, [0.9, 0.1])
    assert result.status_code == 200, result.text
    payload = result.json()
    assert payload["version"] == "v1"
    assert payload["label"] == "defect-A"
    assert payload["score"] == pytest.approx(0.9)

    st = status(api)
    assert st["active_version"] == "v1"
    assert st["slots_in_use"] == 1
    assert st["total_in_flight"] == 0
    assert st["total_completed"] >= 1


def test_02_feature_width_mismatch(api):
    resp = infer(api, [1.0, 2.0, 3.0])
    assert resp.status_code == 422
    assert resp.json()["detail"]["error"] == "feature_dimension_mismatch"
    assert status(api)["total_in_flight"] == 0


def test_03_load_failure_keeps_old_version(api):
    before = status(api)["slots_in_use"]
    resp = upload(
        api,
        artifact("bad-load", W_B, [0.0, 0.0], LABELS, simulate_load_failure=True),
    )
    assert resp.status_code == 502, resp.text
    detail = resp.json()["detail"]
    assert detail["error"] == "candidate_load_failed"
    assert detail["fallback_version"] == "v1"

    st = status(api)
    assert st["active_version"] == "v1"
    assert st["slots_in_use"] == before  # candidate slot released
    failed = next(v for v in st["versions"] if v["version"] == "bad-load")
    assert failed["state"] == "failed"
    assert "simulated" in failed["error"]
    # Old version still answers with the old weights.
    assert infer(api, [0.9, 0.1]).json()["label"] == "defect-A"


def test_04_invalid_artifact_rejected(api):
    resp = upload(api, artifact("broken", [[1.0, 2.0]], [0.0, 0.0], ["a"]))
    assert resp.status_code == 422
    assert resp.json()["detail"]["error"] == "invalid_artifact"


def test_05_old_version_serves_during_candidate_load(api):
    observations = []
    resp = {}

    def do_upload():
        resp["r"] = upload(
            api, artifact("v2", W_B, [0.0, 0.0], LABELS, slow_load_ms=1200)
        )

    import threading

    t = threading.Thread(target=do_upload)
    t.start()
    deadline = time.time() + 1.0
    while time.time() < deadline:
        r = infer(api, [0.9, 0.1])
        if r.status_code == 200:
            observations.append(r.json()["version"])
        time.sleep(0.1)
    t.join()
    assert resp["r"].status_code == 201, resp["r"].text
    assert observations, "old version did not answer during candidate load"
    assert set(observations) <= {"v1", "v2"}
    assert "v1" in observations

    st = status(api)
    assert st["active_version"] == "v2"
    v1 = next(v for v in st["versions"] if v["version"] == "v1")
    assert v1["state"] == "retired"


def test_06_timeout_does_not_release_slot(api):
    # Let the drain of v1 settle first so both slots are known.
    wait_for(lambda: status(api)["slots_in_use"] == 1)

    # Long enough that the upload's bounded drain-wait (3s) expires.
    resp = infer(api, [0.9, 0.1], timeout_ms=150, slow_ms=6000)
    assert resp.status_code == 504, resp.text
    detail = resp.json()["detail"]
    assert detail["error"] == "inference_timeout"
    assert detail["version"] == "v2"

    # Immediately after the client gave up the work is still running.
    st = status(api)
    assert st["total_in_flight"] == 1
    assert st["active_version"] == "v2"

    # Swap commits while the old request is still computing.
    resp2 = upload(api, artifact("v3", W_A, [0.0, 0.0], LABELS))
    assert resp2.status_code == 201, resp2.text
    st = status(api)
    assert st["active_version"] == "v3"
    states = dict((s["version"], s["process_state"]) for s in st["slots"] if s["version"])
    assert states.get("v2") == "draining"
    assert st["slots_in_use"] == 2

    # Third load must be refused while the drain is still held by real work.
    resp3 = upload(api, artifact("v4", W_B, [0.0, 0.0], LABELS))
    assert resp3.status_code == 429, resp3.text
    assert resp3.json()["detail"]["error"] == "capacity_exhausted"
    assert status(api)["active_version"] == "v3"  # never stopped the current version

    # New traffic already goes to the new version.
    assert infer(api, [0.9, 0.1]).json()["version"] == "v3"

    # Once the slow work actually finishes, the drained slot is released.
    wait_for(lambda: status(api)["slots_in_use"] == 1, timeout=15)
    st = status(api)
    v2 = next(v for v in st["versions"] if v["version"] == "v2")
    assert v2["state"] == "retired"
    assert v2["process_state"] == "idle"
    assert st["total_in_flight"] == 0


def test_07_third_version_loads_after_release(api):
    resp = upload(api, artifact("v4", W_B, [0.0, 0.0], LABELS))
    assert resp.status_code == 201, resp.text
    st = status(api)
    assert st["active_version"] == "v4"
    in_use = [s for s in st["slots"] if s["process_state"] != "idle"]
    assert len(in_use) == 2  # v4 active + v3 draining
    assert infer(api, [0.9, 0.1]).json()["version"] == "v4"


def test_08_concurrent_swaps_one_winner(api):
    wait_for(lambda: status(api)["slots_in_use"] == 1, timeout=10)
    import threading

    results = []

    def go(version, weights, slow):
        results.append(
            (
                version,
                upload(
                    api,
                    artifact(version, weights, [0.0, 0.0], LABELS, slow_load_ms=slow),
                ).status_code,
            )
        )

    t1 = threading.Thread(target=go, args=("c1", W_A, 1200))
    t2 = threading.Thread(target=go, args=("c2", W_B, 0))
    t1.start()
    time.sleep(0.25)
    t2.start()
    t1.join()
    t2.join()
    winners = [v for v, code in results if code == 201]
    losers = [v for v, code in results if code == 409]
    assert len(winners) == 1, results
    assert len(losers) == 1, results
    assert status(api)["active_version"] == winners[0]


def test_08b_duplicate_id_concurrent_uploads(api):
    """Two simultaneous uploads of the same id: exactly one wins, one 409."""
    wait_for(lambda: status(api)["slots_in_use"] == 1, timeout=10)
    import threading

    codes = []

    def go():
        codes.append(upload(api, artifact("dup", W_A, [0.0, 0.0], LABELS)).status_code)

    threads = [threading.Thread(target=go) for _ in range(2)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert sorted(codes) == [201, 409], codes
    assert status(api)["active_version"] == "dup"
    # The loser must not leave a stale swap claim behind: another swap works.
    resp = upload(api, artifact("dup-next", W_B, [0.0, 0.0], LABELS))
    assert resp.status_code == 201, resp.text


def test_09_status_reports_processes_and_counts(api):
    st = status(api)
    assert st["capacity"] == 2
    assert 1 <= st["slots_in_use"] <= 2
    for slot in st["slots"]:
        assert slot["process_state"] in {"loading", "active", "draining", "idle"}
        if slot["version"] is not None:
            assert slot["alive"] is True
            assert slot["pid"] is not None
    active_slots = [s for s in st["slots"] if s["process_state"] == "active"]
    assert len(active_slots) == 1
    assert active_slots[0]["version"] == st["active_version"]


def test_10_duplicate_version_is_conflict(api):
    active = status(api)["active_version"]
    resp = upload(api, artifact(active, W_A, [0.0, 0.0], LABELS))
    assert resp.status_code == 409
    assert resp.json()["detail"]["error"] == "version_exists"


def test_11_stale_expected_version(api):
    resp = upload(
        api,
        artifact(
            f"stale-{uuid.uuid4().hex[:6]}",
            W_A,
            [0.0, 0.0],
            LABELS,
            expected_active_version="does-not-exist",
        ),
    )
    assert resp.status_code == 409
    assert resp.json()["detail"]["error"] == "swap_conflict"


# ---------------------------------------------------------------------------
# Device alarm behaviour (normal_label + device_id)
# ---------------------------------------------------------------------------


def test_13_normal_label_must_belong_to_version_labels(api):
    resp = upload(
        api,
        artifact(
            "al-v1",
            W_NORM,
            [0.0, 0.0],
            NORM_LABELS,
            normal_label="not-a-real-label",
        ),
    )
    assert resp.status_code == 422, resp.text
    assert resp.json()["detail"]["error"] == "invalid_artifact"
    # The version was never created.
    assert all(v["version"] != "al-v1" for v in status(api)["versions"])


def test_14_anonymous_inference_is_not_tracked(api):
    resp = upload(
        api,
        artifact("al-v1", W_NORM, [0.0, 0.0], NORM_LABELS, normal_label=NORMAL_LABEL),
    )
    assert resp.status_code == 201, resp.text
    assert resp.json()["normal_label"] == NORMAL_LABEL

    # No device_id -> original behaviour, no device record is created.
    r = infer(api, [0.9, 0.1])
    assert r.status_code == 200
    assert r.json()["label"] == NORMAL_LABEL
    listed = alarms(api)
    assert listed["alarm_threshold"] == 3
    assert all(d["device_id"] != "ghost" for d in listed["devices"])
    assert alarm_view(api, "ghost").status_code == 404


def test_15_three_consecutive_anomalies_alarm_then_normal_resets(api):
    device = "line-3-camera-7"
    # Negative first feature -> NORM_OTHER (anomaly vs normal_label pass).
    for _ in range(3):
        r = infer(api, [-0.9, 0.1], device_id=device)
        assert r.status_code == 200
        assert r.json()["label"] == NORM_OTHER
    view = wait_alarm_settled(api, device, 3)
    assert view["version"] == "al-v1"
    assert view["normal_label"] == NORMAL_LABEL
    assert view["consecutive_anomalies"] == 3
    assert view["alarm"] is True
    assert view["pending"] == 0

    # A normal result clears both the streak and the alarm.
    r = infer(api, [0.9, 0.1], device_id=device)
    assert r.json()["label"] == NORMAL_LABEL
    view = wait_alarm_settled(api, device, 4)
    assert view["consecutive_anomalies"] == 0
    assert view["alarm"] is False
    assert view["last_outcome"] == "normal"


def test_16_two_anomalies_do_not_alarm_and_are_independent_per_device(api):
    d1, d2 = "dev-a", "dev-b"
    for _ in range(2):
        assert infer(api, [-0.9, 0.1], device_id=d1).status_code == 200
    view = wait_alarm_settled(api, d1, 2)
    assert view["consecutive_anomalies"] == 2
    assert view["alarm"] is False

    # d2 stays untouched.
    assert alarm_view(api, d2).status_code == 404

    # Third anomaly for d1 alarms.
    assert infer(api, [-0.9, 0.1], device_id=d1).status_code == 200
    view = wait_alarm_settled(api, d1, 3)
    assert view["alarm"] is True


def test_17_timeout_result_still_counts_and_ordering_is_admission_order(api):
    device = "slow-cam"
    # Admit two slow anomaly requests with an impatient client; both 504 but
    # the worker keeps computing and the tickets must still settle.
    for _ in range(2):
        r = infer(api, [-0.9, 0.1], timeout_ms=150, slow_ms=1500, device_id=device)
        assert r.status_code == 504, r.text
    mid = alarms(api)["devices"]
    mid = next(d for d in mid if d["device_id"] == device)
    # Admitted but not settled yet (the first worker call is still running).
    assert mid["pending"] == 2
    assert mid["total_applied"] == 0

    # A third, fast anomaly after the slow ones finish must apply *after* them
    # in admission order; wait for the slow work to settle first.
    wait_alarm_settled(api, device, 2, timeout=15)
    r = infer(api, [-0.9, 0.1], device_id=device)
    assert r.status_code == 200
    view = wait_alarm_settled(api, device, 3)
    assert view["consecutive_anomalies"] == 3
    assert view["alarm"] is True
    assert view["pending"] == 0


def test_18_swap_clears_alarm_and_old_version_late_results_are_discarded(api):
    device = "swap-cam"
    # Bring the device into alarm on al-v1.
    for _ in range(3):
        assert infer(api, [-0.9, 0.1], device_id=device).status_code == 200
    view = wait_alarm_settled(api, device, 3)
    assert view["alarm"] is True

    # An old-version request is admitted, then made slow so it is still
    # running when the swap commits.
    slow = infer(
        api,
        [-0.9, 0.1],
        timeout_ms=150,
        slow_ms=2500,
        device_id=device,
    )
    assert slow.status_code == 504
    before = next(d for d in alarms(api)["devices"] if d["device_id"] == device)
    assert before["pending"] == 1

    # New version (different weights, same normal label) activates; alarm clears.
    resp = upload(
        api,
        artifact(
            "al-v2",
            W_NORM,
            [0.1, -0.1],
            NORM_LABELS,
            normal_label=NORMAL_LABEL,
        ),
    )
    assert resp.status_code == 201, resp.text
    after_swap = alarm_view(api, device).json()
    assert after_swap["version"] == "al-v2"
    assert after_swap["alarm"] is False
    assert after_swap["consecutive_anomalies"] == 0

    # The late old-version anomaly settles but must not pollute the new
    # version.  Wait for the 4 admitted tickets (3 fast + 1 slow) to drain.
    view = wait_alarm_settled(api, device, 4, timeout=15)
    assert view["total_discarded"] == 1
    assert view["total_applied"] == 3  # only the pre-swap fast ones
    assert view["consecutive_anomalies"] == 0
    assert view["alarm"] is False

    # Fresh inference on the new version counts from scratch.
    assert infer(api, [-0.9, 0.1], device_id=device).status_code == 200
    view = wait_alarm_settled(api, device, 5)
    assert view["total_applied"] == 4
    assert view["total_discarded"] == 1
    assert view["consecutive_anomalies"] == 1
    assert view["version"] == "al-v2"


def test_19_model_without_normal_label_keeps_original_behaviour(api):
    device = "unmonitored-cam"
    resp = upload(api, artifact("plain-v1", W_A, [0.0, 0.0], LABELS))
    assert resp.status_code == 201, resp.text
    assert resp.json()["normal_label"] is None
    for _ in range(4):
        assert infer(api, [0.9, 0.1], device_id=device).status_code == 200
    # Even with device_id, an unconfigured model never creates device state.
    assert alarm_view(api, device).status_code == 404


def test_20_disconnected_client_result_still_settles(api):
    # Ensure a model with a normal label is active without depending on order.
    active_version = status(api)["active_version"]
    active_info = next(
        (v for v in status(api)["versions"] if v["version"] == active_version),
        None,
    )
    if not active_info or not active_info.get("normal_label"):
        resp = upload(
            api,
            artifact("disc-v1", W_NORM, [0.0, 0.0], NORM_LABELS, normal_label=NORMAL_LABEL),
        )
        assert resp.status_code == 201, resp.text
    device = "disconnect-cam"

    # Fire a slow device inference with a very short client timeout built into
    # httpx: the client gives up (ReadTimeout) while the worker keeps computing.
    with httpx.Client(base_url=api.base, timeout=0.1) as impatient:
        with pytest.raises(httpx.ReadTimeout):
            impatient.post(
                "/infer",
                json={
                    "features": [-0.9, 0.1],
                    "slow_ms": 1500,
                    "device_id": device,
                },
            )

    # The ticket was admitted; it must settle into the alarm state once the
    # shielded worker call finishes, even though no client is waiting.
    view = wait_alarm_settled(api, device, 1, timeout=15)
    assert view["total_applied"] == 1
    assert view["total_discarded"] == 0
    assert view["consecutive_anomalies"] == 1
    assert view["alarm"] is False
    assert view["last_outcome"] == "anomaly"


def test_21_failed_computation_resets_the_streak(api):
    # Ensure a monitored model is active and reachable.
    active_version = status(api)["active_version"]
    active_info = next(
        (v for v in status(api)["versions"] if v["version"] == active_version),
        None,
    )
    if not active_info or not active_info.get("normal_label"):
        resp = upload(
            api,
            artifact("fail-v1", W_NORM, [0.0, 0.0], NORM_LABELS, normal_label=NORMAL_LABEL),
        )
        assert resp.status_code == 201, resp.text

    device = "failure-cam"
    # Build a streak of two anomalies.
    for _ in range(2):
        assert infer(api, [-0.9, 0.1], device_id=device).status_code == 200
    view = wait_alarm_settled(api, device, 2)
    assert view["consecutive_anomalies"] == 2
    assert view["alarm"] is False

    # Make sure no earlier slow inference is still queued in the worker before
    # we kill it, so this device's next ticket is the only one in flight.
    drain_deadline = time.time() + 15
    while time.time() < drain_deadline:
        if status(api)["total_in_flight"] == 0:
            break
        time.sleep(0.1)
    assert status(api)["total_in_flight"] == 0

    active_slot = next(
        s for s in status(api)["slots"] if s["process_state"] == "active"
    )
    pid = active_slot["pid"]
    assert pid

    # Admit a slow device inference first so the alarm ticket exists, then kill
    # the worker while the computation is in flight.  The shielded call fails
    # and the ticket must settle as an error (reset), never as a third anomaly.
    with httpx.Client(base_url=api.base, timeout=30.0) as client:
        import threading

        holder: dict = {}

        def fire():
            holder["resp"] = client.post(
                "/infer",
                json={
                    "features": [-0.9, 0.1],
                    "device_id": device,
                    "slow_ms": 3000,
                },
            )

        t = threading.Thread(target=fire)
        t.start()
        # Wait until the request is admitted (and the worker picked it up).
        wait_for(lambda: status(api)["total_in_flight"] >= 1, timeout=10)
        admitted = alarm_view(api, device).json()
        assert admitted["total_admitted"] == 3
        os.kill(pid, 9)
        t.join(timeout=30)
        assert holder["resp"].status_code in (502, 503), holder["resp"].status_code

    # The failed ticket settles as an error-outcome and clears the streak.
    # Other devices' slow tickets may still be pending, so key off the settled
    # counts for this device rather than a global empty-pending condition.
    deadline2 = time.time() + 15
    view = None
    while time.time() < deadline2:
        candidate = alarm_view(api, device)
        if candidate.status_code == 200:
            view = candidate.json()
            if view["total_applied"] + view["total_discarded"] >= 3:
                break
        time.sleep(0.05)
    assert view is not None and (
        view["total_applied"] + view["total_discarded"] >= 3
    )
    assert view["consecutive_anomalies"] == 0
    assert view["alarm"] is False
    assert view["last_outcome"] == "error"


def test_12_no_model_after_shutdown(api):
    # Sanity: the service still answers health and exposes /docs.
    assert httpx.get(f"{api.base}/healthz", timeout=5).status_code == 200
    assert httpx.get(f"{api.base}/docs", timeout=5).status_code == 200
    assert httpx.get(f"{api.base}/openapi.json", timeout=5).status_code == 200
