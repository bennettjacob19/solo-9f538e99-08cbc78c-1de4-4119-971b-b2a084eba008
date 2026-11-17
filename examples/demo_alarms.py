#!/usr/bin/env python3
"""End-to-end demo of the per-device on-call alarm rule, against a live API.

Usage::

    python examples/demo_alarms.py [BASE_URL]

It walks through everything a production engineer relies on:

1. upload a model that saves ``normal_label`` (and prove a label outside the
   version's label set is rejected with 422);
2. three consecutive anomalous results for one device raise the alarm, a
   normal result clears it, and devices are independent;
3. a slow request whose client gives up (504) still settles into the count;
4. a hot swap clears the alarm and the old version's late result is discarded
   instead of polluting the new version;
5. inference without ``device_id`` (or against a model without
   ``normal_label``) never creates device state.
"""

from __future__ import annotations

import json
import sys
import time

import httpx

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"

# Two-feature linear model: positive first feature -> "pass", negative -> "defect".
WEIGHTS = [[1.0, 0.0], [-1.0, 0.0]]
LABELS = ["pass", "defect"]
NORMAL = "pass"
NORMAL_FEATURES = [0.9, 0.1]
DEFECT_FEATURES = [-0.9, 0.1]

client = httpx.Client(base_url=BASE, timeout=60.0)


def show(label, resp):
    try:
        body = resp.json()
    except Exception:
        body = resp.text
    print(f"  {label:<46} {resp.status_code}  {json.dumps(body)[:200]}")
    return resp


def upload(version, bias=None, **extra):
    payload = {
        "version": version,
        "weights": WEIGHTS,
        "bias": bias or [0.0, 0.0],
        "labels": LABELS,
        "normal_label": NORMAL,
        **extra,
    }
    return client.post("/admin/models", json=payload)


def infer(device, features, **extra):
    body = {"features": features, "device_id": device, **extra}
    return client.post("/infer", json=body)


def view(device):
    resp = client.get(f"/admin/device-alarms/{device}")
    if resp.status_code == 200:
        return resp.json()
    return None


def wait_settled(device, admitted, timeout=20.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        v = view(device)
        if v and v["pending"] == 0 and v["total_admitted"] >= admitted:
            return v
        time.sleep(0.1)
    raise AssertionError(f"{device}: tickets did not settle in time")


def print_view(title, device):
    v = view(device)
    if v is None:
        print(f"  {title}: no device state")
        return
    print(
        f"  {title:<46} version={v['version']} streak={v['consecutive_anomalies']} "
        f"alarm={v['alarm']} pending={v['pending']} applied={v['total_applied']} "
        f"discarded={v['total_discarded']}"
    )


def main() -> int:
    print("\n== 1. upload a monitored model ==")
    show("upload monitor-v1", upload("monitor-v1"))
    show(
        "normal_label outside label set -> 422",
        upload("monitor-bad", normal_label="not-a-label"),
    )

    print("\n== 2. three anomalies alarm, a normal result clears ==")
    device = "line-3-camera-7"
    for i in range(3):
        show(f"defect inference #{i + 1}", infer(device, DEFECT_FEATURES))
    wait_settled(device, 3)
    print_view("after three anomalies", device)

    show("normal inference", infer(device, NORMAL_FEATURES))
    wait_settled(device, 4)
    print_view("after a normal result", device)

    print("\n== 3. devices are independent ==")
    other = "line-5-camera-2"
    show("defect on the other device", infer(other, DEFECT_FEATURES))
    wait_settled(other, 1)
    print_view("first device", device)
    print_view("other device", other)
    print("  all devices:")
    for d in client.get("/admin/device-alarms").json()["devices"]:
        print(f"    - {d['device_id']}: alarm={d['alarm']} streak={d['consecutive_anomalies']}")

    print("\n== 4. a 504 still settles into the count ==")
    slow_device = "slow-cam"
    resp = infer(slow_device, DEFECT_FEATURES, slow_ms=2000, timeout_ms=150)
    show("slow defect inference (client gives up)", resp)
    print_view("immediately after the 504", slow_device)
    wait_settled(slow_device, 1)
    print_view("after the shielded work finishes", slow_device)

    print("\n== 5. hot swap clears alarms and isolates late old results ==")
    for i in range(3):
        show(f"bring device to alarm #{i + 1}", infer(device, DEFECT_FEATURES))
    wait_settled(device, 7)
    print_view("alarming before swap", device)

    late = infer(device, DEFECT_FEATURES, slow_ms=2500, timeout_ms=150)
    show("old-version request still running at swap time", late)
    show("upload monitor-v2", upload("monitor-v2", bias=[0.1, -0.1]))
    print_view("right after the swap", device)

    # Wait for the late old-version ticket, then one fresh new-version result.
    deadline = time.time() + 20
    while time.time() < deadline:
        v = view(device)
        if v and v["total_discarded"] >= 1 and v["pending"] == 0:
            break
        time.sleep(0.1)
    show("new-version defect inference", infer(device, DEFECT_FEATURES))
    wait_settled(device, 9)
    print_view("final state", device)

    print("\n== 6. anonymous inference stays unmonitored ==")
    show(
        "inference without device_id",
        client.post("/infer", json={"features": DEFECT_FEATURES}),
    )
    anon = client.get("/admin/device-alarms/ghost")
    show("unknown device -> 404", anon)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
