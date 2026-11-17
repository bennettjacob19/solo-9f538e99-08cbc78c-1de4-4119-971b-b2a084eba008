"""Per-device consecutive-anomaly alarm state.

This module implements the on-call rule the production engineers use to decide
whether a device's line must be stopped:

* Only requests that carry a ``device_id`` *and* hit a model version that
  configured ``normal_label`` participate.  Everything else is ignored, so the
  original service behaviour is unchanged for unconfigured models and for
  anonymous inference.
* Results are counted in **admission order**, not completion order.  Inference
  stays fully concurrent: a request that finishes later than a request admitted
  after it cannot skip the unsettled predecessor.  Each device owns a queue
  consumed by a single asyncio task; the queue entries are futures resolved by
  the worker call, so the consumer blocks until the predecessor settles.
* A result whose label differs from the version's ``normal_label`` extends the
  consecutive-anomaly count; a normal result or a failed computation resets it
  to zero.  When the count reaches the configured limit (default three) the
  device enters the alarming state.
* Timeouts and client disconnects do *not* settle anything: the shielded
  worker task keeps running and resolves the ticket when it finally returns,
  so 504-era results still take part in the judgement.
* On every model activation each device is reset and moved to a new
  generation.  Late results of the old version carry the old generation and
  are discarded, so they can never pollute the new version's counters.

The tracker lives only in the API process (the single owner of the service
state); no cross-process plumbing is needed.  All methods run on the event
loop thread, so plain dictionaries need no extra locking.
"""

from __future__ import annotations

import asyncio
import itertools
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger("qc.alarms")

# How many consecutive non-normal results raise the alarm.
DEFAULT_ALARM_LIMIT = 3


@dataclass
class DeviceState:
    device_id: str
    version: Optional[str]
    normal_label: Optional[str]
    generation: int = 0
    consecutive: int = 0
    alarm: bool = False
    # Tickets admitted but not yet applied in admission order.  Includes
    # tickets whose worker call has finished but which are waiting for an
    # earlier admitted ticket to settle.
    pending: int = 0
    total_admitted: int = 0
    total_applied: int = 0
    total_discarded: int = 0
    last_label: Optional[str] = None
    last_outcome: Optional[str] = None
    updated_at: Optional[float] = None
    _queue: Optional["asyncio.Queue[Optional[Admission]]"] = field(
        repr=False, default=None
    )
    _consumer: Optional[asyncio.Task] = field(default=None, repr=False)


@dataclass
class Admission:
    """A ticket handed out at inference admission time.

    The worker call resolves ``future`` exactly once -- with a ``(label,
    score)`` tuple on success, or with an exception when the computation
    failed.  The per-device consumer awaits these in admission order.
    """

    device_id: str
    version: str
    normal_label: str
    generation: int
    seq: int
    future: "asyncio.Future[tuple[str, float]]"


class DeviceAlarmTracker:
    """In-process, per-device consecutive-anomaly state machine."""

    def __init__(self, alarm_limit: int = DEFAULT_ALARM_LIMIT) -> None:
        self._devices: Dict[str, DeviceState] = {}
        self._seq = itertools.count(1)
        self._alarm_limit = int(alarm_limit)
        self._closed = False

    # ------------------------------------------------------------------
    # Admission / settlement
    # ------------------------------------------------------------------
    def admit(
        self, device_id: str, version: str, normal_label: str
    ) -> Optional[Admission]:
        """Register a device-scoped inference at admission time.

        Returns ``None`` when there is nothing to track (anonymous request or
        a model without ``normal_label``).  The returned admission must be
        resolved by the worker call via :meth:`succeed` / :meth:`fail`.
        """
        if self._closed or not device_id or not normal_label:
            return None

        state = self._devices.get(device_id)
        if state is None:
            state = DeviceState(
                device_id=device_id,
                version=version,
                normal_label=normal_label,
            )
            state._queue = asyncio.Queue()
            self._devices[device_id] = state
            state._consumer = asyncio.create_task(
                self._consume(state), name=f"alarm-consumer-{device_id}"
            )
        elif state.version != version or state.normal_label != normal_label:
            # Defensive: an activation should have reset the device already.
            # Bring the state to the new version without leaving stale counts.
            self._reset_state(state, version, normal_label)

        admission = Admission(
            device_id=device_id,
            version=version,
            normal_label=normal_label,
            generation=state.generation,
            seq=next(self._seq),
            future=asyncio.get_running_loop().create_future(),
        )
        state.pending += 1
        state.total_admitted += 1
        state._queue.put_nowait(admission)
        return admission

    @staticmethod
    def succeed(admission: Admission, label: str, score: float) -> None:
        if not admission.future.done():
            admission.future.set_result((label, score))

    @staticmethod
    def fail(admission: Admission, error: BaseException) -> None:
        if not admission.future.done():
            admission.future.set_exception(error)

    async def _consume(self, state: DeviceState) -> None:
        """Apply one device's tickets strictly in admission order."""
        queue = state._queue
        assert queue is not None
        while True:
            admission = await queue.get()
            if admission is None:  # shutdown sentinel
                queue.task_done()
                return
            try:
                # Block here until this exact ticket's worker call settles.
                # A later-admitted ticket that finished earlier is queued
                # behind this one and cannot be applied first.
                outcome = await self._resolve(admission)
                self._apply(state, admission, outcome)
            except asyncio.CancelledError:
                raise
            except Exception:  # pragma: no cover - defensive
                logger.exception("alarm consumer for %s crashed", state.device_id)
            finally:
                state.pending = max(0, state.pending - 1)
                queue.task_done()

    @staticmethod
    async def _resolve(admission: Admission) -> Dict[str, Any]:
        try:
            label, _score = await admission.future
            return {"ok": True, "label": label}
        except BaseException as exc:  # computation failure -> reset streak
            return {"ok": False, "label": None, "error": f"{type(exc).__name__}: {exc}"}

    def _apply(
        self, state: DeviceState, admission: Admission, outcome: Dict[str, Any]
    ) -> None:
        # Version guard: a swap moved this device to a new generation.  Late
        # results of the retired version are discarded without touching the
        # new version's counters.
        if (
            admission.generation != state.generation
            or admission.version != state.version
        ):
            state.total_discarded += 1
            state.updated_at = time.time()
            logger.info(
                "discard stale result for device %s: ticket version %s (gen %d), "
                "current version %s (gen %d)",
                state.device_id,
                admission.version,
                admission.generation,
                state.version,
                state.generation,
            )
            return

        state.updated_at = time.time()
        state.total_applied += 1
        if not outcome["ok"]:
            # Failed computations reset the streak, never extend it.
            state.consecutive = 0
            state.alarm = False
            state.last_label = None
            state.last_outcome = "error"
            logger.info(
                "device %s: computation failed; streak reset to 0 (%s)",
                state.device_id,
                outcome.get("error"),
            )
            return

        label = outcome["label"]
        state.last_label = label
        if label == admission.normal_label:
            state.consecutive = 0
            state.alarm = False
            state.last_outcome = "normal"
        else:
            state.consecutive += 1
            state.last_outcome = "anomaly"
            if state.consecutive >= self._alarm_limit:
                if not state.alarm:
                    logger.warning(
                        "ALARM device %s on %s: %d consecutive result(s) != %r "
                        "(latest=%r)",
                        state.device_id,
                        admission.version,
                        state.consecutive,
                        admission.normal_label,
                        label,
                    )
                state.alarm = True
            else:
                logger.info(
                    "device %s: anomaly %d/%d (label=%r)",
                    state.device_id,
                    state.consecutive,
                    self._alarm_limit,
                    label,
                )

    # ------------------------------------------------------------------
    # Version activation
    # ------------------------------------------------------------------
    def activate_version(self, version: str, normal_label: Optional[str]) -> None:
        """Reset every device when a new model version becomes active.

        The current alarm is cleared and all in-flight tickets of the old
        version become stale: when they settle they are discarded by the
        generation/version guard instead of polluting the new version.
        """
        for device_id, state in list(self._devices.items()):
            self._reset_state(state, version, normal_label)
        logger.info(
            "version activated: %s (normal_label=%r); %d device(s) reset",
            version,
            normal_label,
            len(self._devices),
        )

    @staticmethod
    def _reset_state(
        state: DeviceState, version: Optional[str], normal_label: Optional[str]
    ) -> None:
        state.version = version
        state.normal_label = normal_label
        state.generation += 1
        state.consecutive = 0
        state.alarm = False
        # Pending old-version tickets stay queued so they can be drained and
        # counted as discarded; they no longer influence any counter.
        state.last_label = None
        state.last_outcome = None

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    def get_device(self, device_id: str) -> Optional[Dict[str, Any]]:
        state = self._devices.get(device_id)
        if state is None:
            return None
        return self._view(state)

    def list_devices(self) -> List[Dict[str, Any]]:
        return [self._view(state) for _id, state in sorted(self._devices.items())]

    def _view(self, state: DeviceState) -> Dict[str, Any]:
        return {
            "device_id": state.device_id,
            "version": state.version,
            "normal_label": state.normal_label,
            "generation": state.generation,
            "consecutive_anomalies": state.consecutive,
            "alarm": state.alarm,
            "alarm_threshold": self._alarm_limit,
            "pending": state.pending,
            "total_admitted": state.total_admitted,
            "total_applied": state.total_applied,
            "total_discarded": state.total_discarded,
            "last_label": state.last_label,
            "last_outcome": state.last_outcome,
            "updated_at": state.updated_at,
        }

    async def aclose(self) -> None:
        """Stop the per-device consumer tasks at service shutdown."""
        self._closed = True
        for state in list(self._devices.values()):
            if state._queue is not None:
                state._queue.put_nowait(None)
            if state._consumer is not None:
                state._consumer.cancel()
        for state in list(self._devices.values()):
            if state._consumer is not None:
                try:
                    await state._consumer
                except (asyncio.CancelledError, Exception):
                    pass
