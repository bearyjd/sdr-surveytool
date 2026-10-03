# capture/unknown/snippet_assembler.py
from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from capture.unknown.energy_trigger import (
    TriggerConfig,
    TriggerEvent,
    find_trigger,
    record_trigger,
)
from capture.unknown.sample_clock import SampleClock


@dataclass(frozen=True)
class CapturedSnippet:
    iq: np.ndarray  # complex64; iq[0] is stream sample start_index
    power: np.ndarray  # float32 moving-average |x|^2, aligned 1:1 with iq
    start_index: int
    start_time: datetime  # sample-derived UTC time of iq[0]
    trigger: TriggerEvent


@dataclass
class _ActiveCapture:
    trigger: TriggerEvent
    start_index: int
    iq: np.ndarray  # preallocated pre + post samples
    power: np.ndarray
    filled: int


class SnippetAssembler:
    """Turns a stream of (iq, power) chunks into complete triggered snippets.

    Pure Python/numpy (no GNU Radio import), so it is unit-tested directly
    with arbitrary chunk sizes. Stateful by necessity: it is a streaming
    buffer, and rebuilding the pre-trigger history immutably on every
    scheduler call would copy up to pre_trigger_samples items each time.

    One snippet at a time: no trigger is evaluated while a snippet is still
    being collected, whatever the cooldown. A snippet still collecting when
    the input stops is dropped, never returned half-written.
    """

    def __init__(
        self,
        clock: SampleClock,
        center_freq_hz: float,
        config: TriggerConfig,
        pre_trigger_samples: int,
        post_trigger_samples: int,
        last_trigger_at: Mapping[float, datetime],
    ) -> None:
        if post_trigger_samples < 1:
            raise ValueError(
                "post_trigger_samples must be >= 1: an empty post-trigger window "
                "never completes, so process() would never advance"
            )
        if pre_trigger_samples < 0:
            raise ValueError("pre_trigger_samples must be >= 0")
        self._clock = clock
        self._center_freq_hz = center_freq_hz
        self._config = config
        self._pre = pre_trigger_samples
        self._post = post_trigger_samples
        self._last_trigger_at = dict(last_trigger_at)
        self._history: deque[tuple[np.ndarray, np.ndarray]] = deque()
        self._history_len = 0
        self._active: _ActiveCapture | None = None

    @property
    def last_trigger_at(self) -> dict[float, datetime]:
        return dict(self._last_trigger_at)

    def process(
        self, iq: np.ndarray, power: np.ndarray, start_index: int
    ) -> list[CapturedSnippet]:
        """Consume one chunk; `iq[0]` is stream sample `start_index`. Returns
        every snippet completed within this chunk (usually none)."""
        snippets = []
        pos = 0
        while pos < len(iq):
            if self._active is not None:
                pos = self._collect(iq, power, pos)
                if self._active.filled == len(self._active.iq):
                    snippets.append(self._finish())
                continue
            trigger = find_trigger(
                power[pos:],
                start_index + pos,
                self._clock,
                self._center_freq_hz,
                self._config,
                self._last_trigger_at,
            )
            end = len(iq) if trigger is None else trigger.sample_index - start_index
            self._remember(iq[pos:end], power[pos:end])
            if trigger is None:
                break
            self._begin(trigger)
            pos = end
        return snippets

    def _remember(self, iq: np.ndarray, power: np.ndarray) -> None:
        """Keep (copies of) the most recent pre_trigger_samples. Copies are
        required: GNU Radio reuses its input buffers after work() returns."""
        if self._pre == 0 or len(iq) == 0:
            return
        self._history.append((iq[-self._pre :].copy(), power[-self._pre :].copy()))
        self._history_len += len(self._history[-1][0])
        while self._history_len - len(self._history[0][0]) >= self._pre:
            dropped, _ = self._history.popleft()
            self._history_len -= len(dropped)

    def _begin(self, trigger: TriggerEvent) -> None:
        if self._history:
            pre_iq = np.concatenate([part for part, _ in self._history])[-self._pre :]
            pre_power = np.concatenate([part for _, part in self._history])[-self._pre :]
        else:
            pre_iq = np.empty(0, dtype=np.complex64)
            pre_power = np.empty(0, dtype=np.float32)
        total = len(pre_iq) + self._post
        active = _ActiveCapture(
            trigger=trigger,
            start_index=trigger.sample_index - len(pre_iq),
            iq=np.empty(total, dtype=np.complex64),
            power=np.empty(total, dtype=np.float32),
            filled=len(pre_iq),
        )
        active.iq[: len(pre_iq)] = pre_iq
        active.power[: len(pre_iq)] = pre_power
        self._active = active
        self._last_trigger_at = record_trigger(self._last_trigger_at, trigger)

    def _collect(self, iq: np.ndarray, power: np.ndarray, pos: int) -> int:
        active = self._active
        assert active is not None, "_collect() is only called mid-capture"
        take = min(len(active.iq) - active.filled, len(iq) - pos)
        active.iq[active.filled : active.filled + take] = iq[pos : pos + take]
        active.power[active.filled : active.filled + take] = power[pos : pos + take]
        active.filled += take
        self._remember(iq[pos : pos + take], power[pos : pos + take])
        return pos + take

    def _finish(self) -> CapturedSnippet:
        active, self._active = self._active, None
        assert active is not None, "_finish() is only called mid-capture"
        return CapturedSnippet(
            iq=active.iq,
            power=active.power,
            start_index=active.start_index,
            start_time=self._clock.time_at(active.start_index),
            trigger=active.trigger,
        )
