# SPDX-License-Identifier: Apache-2.0
"""Lead-bounded talker scheduling governor.

Design reference:
- v1/v2: test/LiveServe/benchmark_design.md §14
- current: lead-bounded decode scheduling with hysteresis

Goal: cap the gap between server-side talker token cumulant (g_cum, in codec
frames) and the estimated client-played frame count (p_est_server). When the
gap (= "lead") exceeds ``L_hi``, the request enters DEFERRED state and is
withheld from ``self.running`` until ``lead < L_lo`` (hysteresis), a hold
budget expires, or a configured policy resumes it.

Key invariants:
- Requests with no observed output units are never deferred. The first
  output-unit timestamp is a playback estimate origin, not proof that a codec
  chunk or client-visible audio has been published. (R1, R2)
- Reinserting a request after a scheduling call does not reset its defer
  episode. Hysteresis and the configured resume rules maintain that state
  across subsequent observations. (R3, R4)
- ``L_lo`` defaults to ``L_hi // 2``; ``hold_max`` defaults to 100 scheduler
  observations of the request. This is not a wall-clock or lifetime defer
  limit; a resumed request may enter a later episode. (R5, R6)

The governor is a no-op when ``L_hi`` is ``None`` or ``<= 0``.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

DEFAULT_SPEAKER_RATE = 25  # frame / s, Qwen3-Omni codec rate
DEFAULT_DEFER_HOLD_MAX = 100  # per-episode scheduler observations, not codec frames
DEFAULT_MIN_OUTPUT_TOKENS_BEFORE_DEFER = 0

@dataclass
class _LeadState:
    """Per-request lead bookkeeping."""

    audio_ttfp_server: float | None = None  # monotonic time of first output token
    last_g_cum: int = 0
    # cumulative count of times this req was *kept* deferred (any step)
    deferred_steps: int = 0
    # state-machine bits
    is_deferred: bool = False
    consecutive_deferred: int = 0
    empty_backlog_steps: int = 0
    defer_started_at: float | None = None
    # cumulative count of times this req *entered* deferred state from running
    defer_episodes: int = 0
    # scheduler observations remaining before this request may re-enter
    # DEFERRED after it has just resumed.
    defer_cooldown_remaining: int = 0

@dataclass
class GovernorStats:
    """Aggregate counters returned via ``snapshot()``."""

    total_scheduler_calls: int = 0
    total_running_seen: int = 0
    total_deferred: int = 0  # step-count (sum of consecutive_deferred increments)
    requests_tracked: int = 0
    # [Step B.1 backlog-aware] number of scheduler calls in which the
    # backlog gate suppressed lead-bounded deferral (waiting queue empty).
    backlog_gate_skipped_calls: int = 0
    # number of scheduler calls in which backlog>0 *and* governor evaluated.
    backlog_gate_active_calls: int = 0
    # scheduler calls where the raw waiting queue was non-empty but not yet
    # sustained enough to open the defer gate.
    backlog_gate_warmup_calls: int = 0
    # scheduler calls where the waiting queue existed but running decode
    # concurrency was too low to justify withholding any request.
    running_gate_skipped_calls: int = 0
    # scheduler calls where backlog/running/age gates were open, used as a
    # cumulative online pressure signal before aggressive defer is enabled.
    backlog_pressure_observed_calls: int = 0
    # scheduler calls where the instantaneous gates were open but cumulative
    # pressure had not yet crossed the configured warmup threshold.
    pressure_warmup_calls: int = 0
    # scheduler calls where a burst signal bypassed the cumulative pressure
    # warmup and opened the defer gate immediately.
    pressure_bypass_calls: int = 0
    # scheduler calls where a configured burst signal used a shorter
    # cumulative warmup instead of the default pressure warmup.
    pressure_fast_warmup_calls: int = 0
    # scheduler calls where oldest-pending age was high enough to make the
    # shorter warmup eligible.
    pressure_fast_warmup_age_signal_calls: int = 0
    # scheduler calls where recent waiting arrivals were high enough to make
    # the shorter warmup eligible.
    pressure_fast_warmup_recent_arrival_signal_calls: int = 0
    # scheduler calls where the recent-arrival threshold was met, regardless
    # of whether cumulative pressure had warmed up enough to open the gate.
    pressure_fast_warmup_recent_arrival_eligible_calls: int = 0
    # scheduler calls where all configured fast-warmup signals were met when
    # the governor was running in all-signal mode.
    pressure_fast_warmup_all_signal_eligible_calls: int = 0
    max_recent_pending_arrivals: int = 0
    # number of times a pending-depth growth signal opened the fast-warmup
    # window.
    pressure_fast_warmup_growth_triggers: int = 0
    # scheduler calls where backlog count was sufficient but the oldest
    # waiting request was too young to justify a defer episode.
    backlog_age_skipped_calls: int = 0
    max_pending_age_s: float = 0.0
    # [v3] number of *episodes* (state transitions RUNNING->DEFERRED).
    total_defer_episodes: int = 0
    # eligible RUNNING->DEFERRED transitions suppressed by
    # max_new_defer_per_call.
    new_defer_limit_suppressed: int = 0
    # eligible RUNNING->DEFERRED transitions suppressed by a pressure-window
    # episode budget.
    defer_budget_suppressed: int = 0
    # eligible RUNNING->DEFERRED transitions suppressed by a per-request
    # cooldown after the request recently resumed from DEFERRED.
    defer_cooldown_suppressed: int = 0
    # number of times a pressure-window episode budget reset after the
    # backlog stayed empty long enough.
    defer_budget_resets: int = 0
    # [v3] number of forced resumes due to hold_max
    total_force_resume: int = 0
    # Request-step evaluations suppressed by the output-unit warmup threshold.
    # This threshold does not establish first codec publication or audibility.
    min_output_suppressed: int = 0
    # Number of request-step evaluations suppressed by caller
    # policy, e.g. KV pressure says this long-context request should not be
    # withheld from decode.
    policy_suppressed: int = 0
    # number of otherwise eligible RUNNING->DEFERRED entries suppressed by an
    # entry-only caller policy. Unlike ``policy_suppressed`` on an already
    # deferred request, this does not force-resume existing deferred requests.
    entry_policy_suppressed: int = 0
    # Number of already-deferred requests force-resumed because
    # the caller policy changed from allow->deny.
    policy_force_resume: int = 0
    # number of already-deferred requests resumed early because the waiting
    # backlog disappeared.
    empty_backlog_force_resume: int = 0
    time_budget_force_resume: int = 0
    # number of times a non-empty prefill-pressure observation opened the
    # bounded service window.
    prefill_priority_window_opens: int = 0
    # scheduler calls where an already-open prefill-priority service window was
    # still active.
    prefill_priority_window_active_calls: int = 0
    # already-deferred requests force-resumed because the prefill-priority
    # service window expired.
    prefill_priority_window_force_resume: int = 0
    # scheduler calls where the caller declared a global suppress condition
    # (for example waiting-short gate closed) and skipped per-request filtering
    # because no request was currently deferred.
    global_suppressed_skip_calls: int = 0
    # [v3] longest consecutive_deferred observed across all requests
    max_consecutive_deferred: int = 0

class TalkerLeadGovernor:
    """Stage-agnostic lead-bounded scheduling governor (v3 hysteresis).

    The governor is *stage-agnostic*: callers (typically a per-stage scheduler)
    decide whether to wire it in. In a multi-stage Omni deployment, only the
    talker stage's ``OmniARScheduler`` should construct one. See §14.4.

    Parameters
    ----------
    L_max : int | None
        Legacy alias for ``L_hi``. Maximum allowed lead in codec frames at
        which a running request *enters* DEFERRED. ``None`` or non-positive
        disables the governor (every call short-circuits to no-op).
    L_lo : int | None
        Lead threshold (in frames) at which a DEFERRED request returns to
        RUNNING (hysteresis lower bound). Defaults to ``L_max // 2``. Must
        satisfy ``0 <= L_lo <= L_max``.
    hold_max : int
        Maximum number of *consecutive* scheduler steps a single request may
        stay DEFERRED before being force-resumed. Anti-starvation safety.
    speaker_rate : int
        Codec frames per wall-clock second. Defaults to ``25`` (Qwen3-Omni).
    min_pending_count : int
        Minimum waiting-queue depth required for entering DEFERRED. Existing
        deferred requests still use normal hysteresis/hold/resume rules.
    min_backlog_active_steps : int
        Number of consecutive scheduler calls with ``pending_count`` above
        ``min_pending_count`` before new defer episodes are allowed. This keeps
        BP focused on sustained queue pressure instead of transient one-step
        blips that tend to hurt QPS on small closed-loop workloads.
    min_running_count : int
        Minimum running decode concurrency required before new DEFERRED
        episodes are allowed. Existing deferred requests still follow normal
        resume/hold rules. This lets large thinker stages avoid throttling
        isolated decode work when there is not enough contention to repay it.
    min_pressure_active_calls : int
        Minimum cumulative scheduler calls with backlog/running/age gates open
        before new DEFERRED episodes are allowed. Defaults to 0, which preserves
        immediate defer. This acts as a lightweight online pressure score.
    pressure_bypass_pending_count : int
        If positive, a waiting queue at least this deep bypasses
        ``min_pressure_active_calls``. This lets real open-loop bursts get the
        aggressive latency path immediately while low-pressure closed-loop
        workloads still need the warmup gate.
    pressure_bypass_pending_age_s : float
        If positive, an oldest-waiting age at least this large also bypasses
        ``min_pressure_active_calls``.
    pressure_fast_warmup_pending_count : int
        If positive, a waiting queue at least this deep can use
        ``pressure_fast_warmup_active_calls`` instead of
        ``min_pressure_active_calls``. Unlike bypass, this still requires some
        sustained observed pressure and is useful for staged bursts where an
        immediate bypass is too aggressive but the default warmup is too slow.
    pressure_fast_warmup_active_calls : int
        Shorter cumulative pressure warmup used when
        ``pressure_fast_warmup_pending_count`` or
        ``pressure_fast_warmup_pending_age_s`` is active. Values <= 0 disable
        this fast-warmup path.
    pressure_fast_warmup_pending_age_s : float
        If positive, an oldest-waiting age at least this value can use
        ``pressure_fast_warmup_active_calls`` instead of
        ``min_pressure_active_calls``. This is less aggressive than
        ``pressure_bypass_pending_age_s`` because it still requires cumulative
        observed pressure before new defer episodes open.
    pressure_fast_warmup_recent_arrivals : int
        If positive, a scheduler-provided count of recent waiting arrivals at
        least this high can use ``pressure_fast_warmup_active_calls`` instead
        of ``min_pressure_active_calls``. The scheduler owns the time window;
        the governor only applies the threshold. This targets staged bursts
        while avoiding gradually accumulated poisson queues.
    pressure_fast_warmup_pending_growth : int
        Optional burst-shape guard for fast warmup. When positive, fast warmup
        is allowed only after pending depth grows by at least this amount in a
        scheduler call. In this mode, ``pressure_fast_warmup_pending_count`` is
        used to arm the growth window, but fast warmup can continue while the
        window is latched even if the instantaneous waiting depth has already
        drained below that threshold. This avoids treating a gradually
        accumulated poisson backlog the same as a staged burst.
    pressure_fast_warmup_growth_hold_calls : int
        Number of eligible backlog-pressure scheduler calls for which a
        pending-growth trigger keeps the fast-warmup window open. The window
        is cleared if the waiting backlog drains first. Only used when
        ``pressure_fast_warmup_pending_growth`` is positive.
    pressure_fast_warmup_require_all_signals : bool
        If true, all configured fast-warmup signals must be active before the
        shorter cumulative warmup can be used. The default is false, which
        preserves the original OR semantics across pending depth/growth,
        oldest-pending age, and recent-arrival signals.
    resume_on_empty_backlog : bool
        If true, an already-deferred request resumes as soon as the waiting
        backlog has been empty for more than ``empty_backlog_grace_steps``,
        unless the caller supplies protected runnable first-chunk pressure.
        The waiting queue does not include newcomers already in running.
    empty_backlog_grace_steps : int
        Number of consecutive empty-backlog scheduler calls an already-deferred
        request may survive before early resume. Only used when
        ``resume_on_empty_backlog`` is true.
    min_output_tokens_before_defer : int
        Minimum cumulative output units required before a request may enter
        DEFERRED. This is only a warmup heuristic: the threshold does not prove
        that a codec chunk or client-visible audio has been produced.
    min_pending_age_s : float
        Optional backlog-age gate. When positive, a new DEFERRED episode is
        allowed only after the oldest waiting request has waited at least this
        long. This avoids reacting to one-tick backlog pulses while still
        opening quickly for real queues.
    max_new_defer_per_call : int
        Maximum number of new RUNNING->DEFERRED transitions admitted during a
        single scheduler call. Values <= 0 preserve the default unlimited
        behavior. Already-deferred requests are not affected.
    max_defer_episodes_per_window : int
        Maximum number of new RUNNING->DEFERRED transitions admitted during a
        backlog-pressure window. Values <= 0 preserve the default unlimited
        behavior. Already-deferred requests are not affected. The window resets
        once empty-backlog observations exceed ``defer_budget_empty_reset_steps``.
        Protected runnable first-chunk pressure keeps this budget active.
    defer_budget_empty_reset_steps : int
        Number of consecutive empty-backlog observations tolerated before the
        pressure-window budget resets on the next such observation. Zero
        resets on the first empty observation; genuine input-wait gaps count.
    defer_cooldown_steps : int
        Number of scheduler observations after a request resumes from
        DEFERRED during which that same request may not re-enter DEFERRED.
        Values <= 0 preserve the default immediate re-entry behavior.
    prefill_priority_window_steps : int
        Optional service window for prefill-priority BP. When positive, each
        non-empty pending/prefill-pressure observation opens a bounded window.
        Already-deferred decode requests may stay withheld only while actual
        pressure exists or this window remains active; once the window expires,
        they are force-resumed even if their lead is still above ``L_lo``.
    """

    def __init__(
        self,
        L_max: int | None,
        L_lo: int | None = None,
        hold_max: int = DEFAULT_DEFER_HOLD_MAX,
        speaker_rate: int = DEFAULT_SPEAKER_RATE,
        min_pending_count: int = 1,
        min_backlog_active_steps: int = 1,
        min_running_count: int = 1,
        min_pressure_active_calls: int = 0,
        pressure_bypass_pending_count: int = 0,
        pressure_bypass_pending_age_s: float = 0.0,
        pressure_fast_warmup_pending_count: int = 0,
        pressure_fast_warmup_active_calls: int = 0,
        pressure_fast_warmup_pending_age_s: float = 0.0,
        pressure_fast_warmup_recent_arrivals: int = 0,
        pressure_fast_warmup_pending_growth: int = 0,
        pressure_fast_warmup_growth_hold_calls: int = 0,
        pressure_fast_warmup_require_all_signals: bool = False,
        resume_on_empty_backlog: bool = False,
        empty_backlog_grace_steps: int = 0,
        min_output_tokens_before_defer: int = DEFAULT_MIN_OUTPUT_TOKENS_BEFORE_DEFER,
        min_pending_age_s: float = 0.0,
        max_new_defer_per_call: int = 0,
        max_defer_episodes_per_window: int = 0,
        defer_budget_empty_reset_steps: int = 0,
        defer_cooldown_steps: int = 0,
        prefill_priority_window_steps: int = 0,
    ):
        self._L_hi: int = int(L_max) if (L_max is not None and L_max > 0) else 0
        if self._L_hi == 0:
            self._L_lo: int = 0
        else:
            if L_lo is None:
                self._L_lo = max(0, self._L_hi // 2)
            else:
                lo = int(L_lo)
                # Clamp into [0, L_hi]; L_lo == L_hi degenerates to v2 behaviour.
                self._L_lo = max(0, min(lo, self._L_hi))
        self._hold_max: int = max(1, int(hold_max))
        self._speaker_rate: int = int(speaker_rate)
        self._min_pending_count = max(1, int(min_pending_count))
        self._min_backlog_active_steps = max(1, int(min_backlog_active_steps))
        self._min_running_count = max(1, int(min_running_count))
        self._min_pressure_active_calls = max(0, int(min_pressure_active_calls))
        self._pressure_bypass_pending_count = max(
            0, int(pressure_bypass_pending_count)
        )
        self._pressure_bypass_pending_age_s = max(
            0.0, float(pressure_bypass_pending_age_s)
        )
        self._pressure_fast_warmup_pending_count = max(
            0, int(pressure_fast_warmup_pending_count)
        )
        self._pressure_fast_warmup_active_calls = max(
            0, int(pressure_fast_warmup_active_calls)
        )
        self._pressure_fast_warmup_pending_age_s = max(
            0.0, float(pressure_fast_warmup_pending_age_s)
        )
        self._pressure_fast_warmup_recent_arrivals = max(
            0, int(pressure_fast_warmup_recent_arrivals)
        )
        self._pressure_fast_warmup_pending_growth = max(
            0, int(pressure_fast_warmup_pending_growth)
        )
        self._pressure_fast_warmup_growth_hold_calls = max(
            0, int(pressure_fast_warmup_growth_hold_calls)
        )
        self._pressure_fast_warmup_require_all_signals = bool(
            pressure_fast_warmup_require_all_signals
        )
        self._last_pending_count: int | None = None
        self._pressure_fast_warmup_growth_latch = 0
        self._backlog_active_steps = 0
        self._resume_on_empty_backlog = bool(resume_on_empty_backlog)
        self._empty_backlog_grace_steps = max(0, int(empty_backlog_grace_steps))
        self._min_output_tokens_before_defer = max(
            0, int(min_output_tokens_before_defer)
        )
        self._min_pending_age_s = max(0.0, float(min_pending_age_s))
        self._max_new_defer_per_call = max(0, int(max_new_defer_per_call))
        self._max_defer_episodes_per_window = max(
            0, int(max_defer_episodes_per_window)
        )
        self._defer_budget_empty_reset_steps = max(
            0, int(defer_budget_empty_reset_steps)
        )
        self._defer_cooldown_steps = max(0, int(defer_cooldown_steps))
        self._prefill_priority_window_steps = max(
            0, int(prefill_priority_window_steps)
        )
        self._prefill_priority_window_remaining = 0
        self._defer_budget_empty_steps = 0
        self._defer_budget_epoch_episodes = 0
        self._state: dict[str, _LeadState] = {}
        self._stats: GovernorStats = GovernorStats()

    @property
    def enabled(self) -> bool:
        return self._L_hi > 0

    @property
    def has_deferred(self) -> bool:
        return any(st.is_deferred for st in self._state.values())

    @property
    def L_max(self) -> int:
        # Backwards-compatible alias for L_hi.
        return self._L_hi

    @property
    def L_hi(self) -> int:
        return self._L_hi

    @property
    def L_lo(self) -> int:
        return self._L_lo

    @property
    def hold_max(self) -> int:
        return self._hold_max

    @property
    def min_pending_count(self) -> int:
        return self._min_pending_count

    @property
    def min_backlog_active_steps(self) -> int:
        return self._min_backlog_active_steps

    @property
    def min_running_count(self) -> int:
        return self._min_running_count

    @property
    def min_pressure_active_calls(self) -> int:
        return self._min_pressure_active_calls

    @property
    def pressure_bypass_pending_count(self) -> int:
        return self._pressure_bypass_pending_count

    @property
    def pressure_bypass_pending_age_s(self) -> float:
        return self._pressure_bypass_pending_age_s

    @property
    def pressure_fast_warmup_pending_count(self) -> int:
        return self._pressure_fast_warmup_pending_count

    @property
    def pressure_fast_warmup_active_calls(self) -> int:
        return self._pressure_fast_warmup_active_calls

    @property
    def pressure_fast_warmup_pending_age_s(self) -> float:
        return self._pressure_fast_warmup_pending_age_s

    @property
    def pressure_fast_warmup_recent_arrivals(self) -> int:
        return self._pressure_fast_warmup_recent_arrivals

    @property
    def pressure_fast_warmup_pending_growth(self) -> int:
        return self._pressure_fast_warmup_pending_growth

    @property
    def pressure_fast_warmup_growth_hold_calls(self) -> int:
        return self._pressure_fast_warmup_growth_hold_calls

    @property
    def pressure_fast_warmup_require_all_signals(self) -> bool:
        return self._pressure_fast_warmup_require_all_signals

    @property
    def resume_on_empty_backlog(self) -> bool:
        return self._resume_on_empty_backlog

    @property
    def empty_backlog_grace_steps(self) -> int:
        return self._empty_backlog_grace_steps

    @property
    def min_output_tokens_before_defer(self) -> int:
        return self._min_output_tokens_before_defer

    @property
    def min_pending_age_s(self) -> float:
        return self._min_pending_age_s

    @property
    def max_new_defer_per_call(self) -> int:
        return self._max_new_defer_per_call

    @property
    def max_defer_episodes_per_window(self) -> int:
        return self._max_defer_episodes_per_window

    @property
    def defer_budget_empty_reset_steps(self) -> int:
        return self._defer_budget_empty_reset_steps

    @property
    def defer_cooldown_steps(self) -> int:
        return self._defer_cooldown_steps

    def _arm_defer_cooldown(self, st: _LeadState) -> None:
        if self._defer_cooldown_steps > 0:
            st.defer_cooldown_remaining = self._defer_cooldown_steps

    def _observe_defer_budget_backlog(self, pending_count: int | None) -> None:
        if self._max_defer_episodes_per_window <= 0 or pending_count is None:
            return
        try:
            pending_count_i = max(0, int(pending_count))
        except (TypeError, ValueError):
            return
        if pending_count_i > 0:
            self._defer_budget_empty_steps = 0
            return
        self._defer_budget_empty_steps += 1
        if (
            self._defer_budget_epoch_episodes > 0
            and self._defer_budget_empty_steps
            > self._defer_budget_empty_reset_steps
        ):
            self._defer_budget_epoch_episodes = 0
            self._stats.defer_budget_resets += 1

    def observe_idle(self, pending_count: int | None = None) -> None:
        """Observe backlog state when no running requests reach filter_running.

        The scheduler does not call ``filter_running`` when no request is in
        decode. This hook lets default-off pressure-window budgets reset across
        workload gaps without perturbing the governor's normal counters.
        """
        self._observe_defer_budget_backlog(pending_count)

    def observe_global_suppressed_skip(
        self, pending_count: int | None = None
    ) -> None:
        """Observe a skipped filter when caller policy globally denies defer."""
        self._observe_defer_budget_backlog(pending_count)
        self._stats.global_suppressed_skip_calls += 1

    def filter_running(
        self,
        running: list[Any],
        now: float | None = None,
        pending_count: int | None = None,
        pending_age_s: float | None = None,
        recent_pending_arrivals: int | None = None,
        defer_allowed: Callable[[Any], bool] | None = None,
        defer_entry_allowed: Callable[[Any], bool] | None = None,
        empty_backlog_protected: bool = False,
        defer_time_limit_s: float | None = None,
    ) -> list[Any]:
        """Inspect each running request, run the v3 hysteresis state machine,
        and return the subset that should be deferred *this step*.

        Parameters
        ----------
        running : list
            Currently running requests in the talker stage.
        now : float, optional
            Monotonic timestamp; defaults to ``time.monotonic()``.
        pending_count : int, optional
            Number of requests in the waiting / backlog queue. When
            ``pending_count == 0`` the *entry* edge (RUNNING->DEFERRED) is
            blocked unless protected runnable first-chunk pressure supplies
            the required single competitor. Already-deferred requests follow the hysteresis
            exit rules (so they can return to RUNNING normally even if
            backlog disappeared).
            ``None`` (default) preserves legacy behaviour (always evaluate).
        pending_age_s : float, optional
            Age of the oldest waiting request in seconds. Only used when
            ``min_pending_age_s`` is positive.
        defer_allowed : callable, optional
            Caller-provided policy hook. Returning ``False`` suppresses
            RUNNING->DEFERRED entry for this request and force-resumes it if
            it was already deferred. This is used by KV-aware BP to avoid
            making large-KV requests linger in GPU memory under pressure.
        defer_entry_allowed : callable, optional
            Caller-provided entry-only policy hook. Returning ``False`` only
            suppresses fresh RUNNING->DEFERRED entry. Already-deferred
            requests continue through normal hysteresis/hold/resume logic.
        empty_backlog_protected : bool
            Temporarily postpone empty-backlog resume while a ready newcomer
            has not published its first codec chunk. Also supplies one known
            competitor for entry and retains the episode budget while the
            ordinary waiting queue is empty. Other entry/exit rules remain.
        defer_time_limit_s : float, optional
            Hard monotonic time budget for each defer episode, independent of
            scheduler-call count or successive newcomers. None preserves the
            default policy. Eligibility resumes on the next scheduler call.

        The caller is responsible for actually removing the deferred requests
        from ``running`` and reinserting them after ``super().schedule()``.

        Returns an empty list if the governor is disabled.
        """
        if not self.enabled:
            return []

        if now is None:
            now = time.monotonic()

        self._stats.total_scheduler_calls += 1
        self._stats.total_running_seen += len(running)
        current_pending_count: int | None = None
        if pending_count is not None:
            try:
                current_pending_count = max(0, int(pending_count))
            except (TypeError, ValueError):
                current_pending_count = None

        # Runnable first-chunk pressure must retain the episode budget across
        # re-entry even though the ordinary waiting queue is empty. Keep the
        # actual pending count for every other policy and for diagnostics.
        self._observe_defer_budget_backlog(
            1 if empty_backlog_protected and current_pending_count == 0 else pending_count
        )

        if self._prefill_priority_window_steps > 0:
            if current_pending_count is None:
                self._prefill_priority_window_remaining = (
                    self._prefill_priority_window_steps
                )
            elif current_pending_count >= self._min_pending_count:
                if self._prefill_priority_window_remaining <= 0:
                    self._stats.prefill_priority_window_opens += 1
                self._prefill_priority_window_remaining = (
                    self._prefill_priority_window_steps
                )
            elif self._prefill_priority_window_remaining > 0:
                self._prefill_priority_window_remaining -= 1
            if self._prefill_priority_window_remaining > 0:
                self._stats.prefill_priority_window_active_calls += 1

        pending_gate_open = (
            pending_count is None
            or (
                current_pending_count is not None
                and current_pending_count >= self._min_pending_count
            )
            # The scheduler supplies this only for a runnable, unpublished
            # newcomer within its original first-ready deadline. A boolean
            # proves one competitor, so it cannot satisfy a higher minimum.
            or (empty_backlog_protected and self._min_pending_count <= 1)
        )
        running_gate_open = len(running) >= self._min_running_count
        raw_backlog_open = pending_gate_open and running_gate_open
        if pending_count is None:
            self._backlog_active_steps = (
                self._min_backlog_active_steps if running_gate_open else 0
            )
        elif raw_backlog_open:
            self._backlog_active_steps += 1
        else:
            self._backlog_active_steps = 0

        pending_growth = 0
        if current_pending_count is not None:
            if self._last_pending_count is not None:
                pending_growth = max(
                    0, current_pending_count - self._last_pending_count
                )
            self._last_pending_count = current_pending_count
            if (
                self._pressure_fast_warmup_pending_growth > 0
                and pending_growth >= self._pressure_fast_warmup_pending_growth
                and (
                    self._pressure_fast_warmup_pending_count <= 0
                    or current_pending_count
                    >= self._pressure_fast_warmup_pending_count
                )
            ):
                self._pressure_fast_warmup_growth_latch = max(
                    1, self._pressure_fast_warmup_growth_hold_calls
                )
                self._stats.pressure_fast_warmup_growth_triggers += 1

        if pending_age_s is not None and pending_age_s > self._stats.max_pending_age_s:
            self._stats.max_pending_age_s = pending_age_s
        age_gate_open = (
            pending_count is None
            or self._min_pending_age_s <= 0.0
            or (
                pending_age_s is not None
                and pending_age_s >= self._min_pending_age_s
            )
        )
        instantaneous_backlog_gate_open = (
            (pending_count is None and running_gate_open)
            or (
                raw_backlog_open
                and self._backlog_active_steps >= self._min_backlog_active_steps
                and age_gate_open
            )
        )
        if instantaneous_backlog_gate_open:
            self._stats.backlog_pressure_observed_calls += 1
        pressure_required_calls = self._min_pressure_active_calls
        pressure_fast_warmup_open = False
        pressure_fast_age_signal_open = (
            self._pressure_fast_warmup_pending_age_s > 0.0
            and pending_age_s is not None
            and pending_age_s >= self._pressure_fast_warmup_pending_age_s
        )
        pressure_fast_recent_arrival_signal_open = False
        if (
            self._pressure_fast_warmup_recent_arrivals > 0
            and recent_pending_arrivals is not None
        ):
            try:
                recent_pending_arrivals_i = max(0, int(recent_pending_arrivals))
                if (
                    recent_pending_arrivals_i
                    > self._stats.max_recent_pending_arrivals
                ):
                    self._stats.max_recent_pending_arrivals = (
                        recent_pending_arrivals_i
                    )
                pressure_fast_recent_arrival_signal_open = (
                    recent_pending_arrivals_i
                    >= self._pressure_fast_warmup_recent_arrivals
                )
            except (TypeError, ValueError):
                pressure_fast_recent_arrival_signal_open = False
            if pressure_fast_recent_arrival_signal_open:
                self._stats.pressure_fast_warmup_recent_arrival_eligible_calls += 1
        if (
            instantaneous_backlog_gate_open
            and self._pressure_fast_warmup_active_calls > 0
            and current_pending_count is not None
            and self._pressure_fast_warmup_active_calls < pressure_required_calls
            and (
                self._pressure_fast_warmup_pending_count > 0
                or self._pressure_fast_warmup_pending_age_s > 0.0
                or self._pressure_fast_warmup_recent_arrivals > 0
            )
        ):
            if self._pressure_fast_warmup_pending_growth > 0:
                fast_warmup_pending_open = (
                    self._pressure_fast_warmup_growth_latch > 0
                )
            elif self._pressure_fast_warmup_pending_count > 0:
                fast_warmup_pending_open = (
                    current_pending_count >= self._pressure_fast_warmup_pending_count
                )
            else:
                fast_warmup_pending_open = False
            configured_fast_signals: list[bool] = []
            if (
                self._pressure_fast_warmup_pending_count > 0
                or self._pressure_fast_warmup_pending_growth > 0
            ):
                configured_fast_signals.append(fast_warmup_pending_open)
            if self._pressure_fast_warmup_pending_age_s > 0.0:
                configured_fast_signals.append(pressure_fast_age_signal_open)
            if self._pressure_fast_warmup_recent_arrivals > 0:
                configured_fast_signals.append(
                    pressure_fast_recent_arrival_signal_open
                )
            if self._pressure_fast_warmup_require_all_signals:
                fast_warmup_open = (
                    bool(configured_fast_signals)
                    and all(configured_fast_signals)
                )
                if fast_warmup_open:
                    self._stats.pressure_fast_warmup_all_signal_eligible_calls += 1
            else:
                fast_warmup_open = (
                    fast_warmup_pending_open
                    or pressure_fast_age_signal_open
                    or pressure_fast_recent_arrival_signal_open
                )
            if fast_warmup_open:
                pressure_required_calls = self._pressure_fast_warmup_active_calls
                pressure_fast_warmup_open = (
                    self._stats.backlog_pressure_observed_calls
                    >= pressure_required_calls
                )
        pressure_warmup_open = (
            self._stats.backlog_pressure_observed_calls
            >= pressure_required_calls
        )
        pressure_bypass_open = False
        if instantaneous_backlog_gate_open and not pressure_warmup_open:
            if (
                self._pressure_bypass_pending_count > 0
                and current_pending_count is not None
                and current_pending_count >= self._pressure_bypass_pending_count
            ):
                pressure_bypass_open = True
            if (
                self._pressure_bypass_pending_age_s > 0.0
                and pending_age_s is not None
                and pending_age_s >= self._pressure_bypass_pending_age_s
            ):
                pressure_bypass_open = True
        pressure_gate_open = pressure_warmup_open or pressure_bypass_open
        backlog_gate_open = instantaneous_backlog_gate_open and pressure_gate_open
        if backlog_gate_open:
            self._stats.backlog_gate_active_calls += 1
            if pressure_bypass_open:
                self._stats.pressure_bypass_calls += 1
            elif pressure_fast_warmup_open:
                self._stats.pressure_fast_warmup_calls += 1
                if pressure_fast_age_signal_open:
                    self._stats.pressure_fast_warmup_age_signal_calls += 1
                if pressure_fast_recent_arrival_signal_open:
                    self._stats.pressure_fast_warmup_recent_arrival_signal_calls += 1
        else:
            self._stats.backlog_gate_skipped_calls += 1
            if instantaneous_backlog_gate_open and not pressure_gate_open:
                self._stats.pressure_warmup_calls += 1
            elif not running_gate_open and pending_gate_open:
                self._stats.running_gate_skipped_calls += 1
            elif raw_backlog_open:
                if not age_gate_open:
                    self._stats.backlog_age_skipped_calls += 1
                else:
                    self._stats.backlog_gate_warmup_calls += 1

        if self._pressure_fast_warmup_growth_latch > 0:
            if current_pending_count == 0:
                self._pressure_fast_warmup_growth_latch = 0
            elif instantaneous_backlog_gate_open:
                self._pressure_fast_warmup_growth_latch -= 1

        prefill_window_expired = (
            self._prefill_priority_window_steps > 0
            and current_pending_count == 0
            and self._prefill_priority_window_remaining <= 0
        )
        deferred: list[Any] = []
        new_defer_this_call = 0
        for req in running:
            req_id = getattr(req, "request_id", None)
            if req_id is None:
                continue
            g_cum_raw = getattr(req, "omni_lead_units", None)
            if g_cum_raw is None:
                g_cum_raw = getattr(req, "num_output_tokens", 0)
            try:
                g_cum = int(g_cum_raw or 0)
            except (TypeError, ValueError):
                g_cum = int(getattr(req, "num_output_tokens", 0) or 0)

            st = self._state.get(req_id)
            if st is None:
                st = _LeadState()
                self._state[req_id] = st
                self._stats.requests_tracked += 1

            # Capture audio_ttfp the first time the talker emits any token.
            if st.audio_ttfp_server is None and g_cum > 0:
                st.audio_ttfp_server = now

            st.last_g_cum = g_cum

            # R1/R2: pre-audio-ttfp requests are exempt. Decode-only BP.
            if st.audio_ttfp_server is None:
                # If somehow flagged deferred earlier (cannot happen normally),
                # clear the bit defensively.
                if st.is_deferred:
                    st.is_deferred = False
                    st.consecutive_deferred = 0
                    st.empty_backlog_steps = 0
                continue

            elapsed = max(0.0, now - st.audio_ttfp_server)
            p_est = elapsed * self._speaker_rate
            lead = g_cum - p_est

            def _call_policy(policy: Callable[[Any], bool] | None) -> bool:
                if policy is None:
                    return True
                try:
                    return bool(policy(req))
                except Exception:
                    return True

            # === v3 hysteresis state machine ===
            if st.is_deferred:
                if (defer_time_limit_s is not None and defer_time_limit_s > 0
                        and st.defer_started_at is not None
                        and now - st.defer_started_at >= defer_time_limit_s):
                    st.is_deferred = False
                    st.consecutive_deferred = 0
                    st.empty_backlog_steps = 0
                    self._arm_defer_cooldown(st)
                    self._stats.time_budget_force_resume += 1
                    continue
                if current_pending_count == 0:
                    if prefill_window_expired:
                        st.is_deferred = False
                        st.consecutive_deferred = 0
                        st.empty_backlog_steps = 0
                        self._arm_defer_cooldown(st)
                        self._stats.prefill_priority_window_force_resume += 1
                        continue
                    st.empty_backlog_steps += 1
                    if (
                        self._resume_on_empty_backlog
                        and not empty_backlog_protected
                        and st.empty_backlog_steps > self._empty_backlog_grace_steps
                    ):
                        st.is_deferred = False
                        st.consecutive_deferred = 0
                        st.empty_backlog_steps = 0
                        self._arm_defer_cooldown(st)
                        self._stats.empty_backlog_force_resume += 1
                        continue
                else:
                    st.empty_backlog_steps = 0
                if not _call_policy(defer_allowed):
                    self._stats.policy_suppressed += 1
                    st.is_deferred = False
                    st.consecutive_deferred = 0
                    st.empty_backlog_steps = 0
                    self._arm_defer_cooldown(st)
                    self._stats.policy_force_resume += 1
                    continue
                # Already deferred. Check exit conditions in order.
                if st.consecutive_deferred >= self._hold_max:
                    # Anti-starvation: force resume after hold_max steps.
                    st.is_deferred = False
                    st.consecutive_deferred = 0
                    st.empty_backlog_steps = 0
                    self._arm_defer_cooldown(st)
                    self._stats.total_force_resume += 1
                    # Do NOT append to deferred this step.
                elif lead < self._L_lo:
                    # Lead has fallen back below the lower threshold; resume.
                    st.is_deferred = False
                    st.consecutive_deferred = 0
                    st.empty_backlog_steps = 0
                    self._arm_defer_cooldown(st)
                else:
                    # Stay deferred.
                    st.consecutive_deferred += 1
                    st.deferred_steps += 1
                    self._stats.total_deferred += 1
                    if st.consecutive_deferred > self._stats.max_consecutive_deferred:
                        self._stats.max_consecutive_deferred = st.consecutive_deferred
                    deferred.append(req)
            else:
                # Currently running. Check entry condition.
                cooldown_active = st.defer_cooldown_remaining > 0
                if cooldown_active:
                    st.defer_cooldown_remaining -= 1
                if lead >= self._L_hi and backlog_gate_open:
                    if g_cum < self._min_output_tokens_before_defer:
                        self._stats.min_output_suppressed += 1
                        continue
                    if cooldown_active:
                        self._stats.defer_cooldown_suppressed += 1
                        continue
                    if not _call_policy(defer_allowed):
                        self._stats.policy_suppressed += 1
                        continue
                    if not _call_policy(defer_entry_allowed):
                        self._stats.policy_suppressed += 1
                        self._stats.entry_policy_suppressed += 1
                        continue
                    if (
                        self._max_new_defer_per_call > 0
                        and new_defer_this_call >= self._max_new_defer_per_call
                    ):
                        self._stats.new_defer_limit_suppressed += 1
                        continue
                    if (
                        self._max_defer_episodes_per_window > 0
                        and self._defer_budget_epoch_episodes
                        >= self._max_defer_episodes_per_window
                    ):
                        self._stats.defer_budget_suppressed += 1
                        continue
                    st.is_deferred = True
                    st.defer_started_at = now
                    st.consecutive_deferred = 1
                    st.empty_backlog_steps = 0
                    st.defer_episodes += 1
                    st.deferred_steps += 1
                    new_defer_this_call += 1
                    if self._max_defer_episodes_per_window > 0:
                        self._defer_budget_epoch_episodes += 1
                    self._stats.total_deferred += 1
                    self._stats.total_defer_episodes += 1
                    if 1 > self._stats.max_consecutive_deferred:
                        self._stats.max_consecutive_deferred = 1
                    deferred.append(req)
                # else: stay running.

        return deferred

    def on_request_finished(self, req_id: str) -> dict | None:
        """Forget per-request state. Returns final per-request counters
        (for stage_metrics emission) or ``None`` if untracked.
        """
        st = self._state.pop(req_id, None)
        if st is None:
            return None
        return {
            "deferred_steps": st.deferred_steps,
            "defer_episodes": st.defer_episodes,
            "audio_ttfp_server": st.audio_ttfp_server,
            "last_g_cum": st.last_g_cum,
        }

    def snapshot(self) -> dict:
        """Return a JSON-friendly snapshot of governor counters (cumulative)."""
        return {
            "enabled": self.enabled,
            "L_max": self._L_hi,
            "L_hi": self._L_hi,
            "L_lo": self._L_lo,
            "hold_max": self._hold_max,
            "speaker_rate": self._speaker_rate,
            "min_pending_count": self._min_pending_count,
            "min_backlog_active_steps": self._min_backlog_active_steps,
            "min_running_count": self._min_running_count,
            "min_pressure_active_calls": self._min_pressure_active_calls,
            "pressure_bypass_pending_count": self._pressure_bypass_pending_count,
            "pressure_bypass_pending_age_s": self._pressure_bypass_pending_age_s,
            "backlog_active_steps": self._backlog_active_steps,
            "resume_on_empty_backlog": self._resume_on_empty_backlog,
            "empty_backlog_grace_steps": self._empty_backlog_grace_steps,
            "min_output_tokens_before_defer": self._min_output_tokens_before_defer,
            "min_pending_age_s": self._min_pending_age_s,
            "max_new_defer_per_call": self._max_new_defer_per_call,
            "max_defer_episodes_per_window": (
                self._max_defer_episodes_per_window
            ),
            "defer_budget_empty_reset_steps": (
                self._defer_budget_empty_reset_steps
            ),
            "defer_cooldown_steps": self._defer_cooldown_steps,
            "prefill_priority_window_steps": (
                self._prefill_priority_window_steps
            ),
            "prefill_priority_window_remaining": (
                self._prefill_priority_window_remaining
            ),
            "max_defer_cooldown_remaining": max(
                (st.defer_cooldown_remaining for st in self._state.values()),
                default=0,
            ),
            "defer_budget_epoch_episodes": self._defer_budget_epoch_episodes,
            "defer_budget_empty_steps": self._defer_budget_empty_steps,
            "scheduler_calls": self._stats.total_scheduler_calls,
            "running_seen": self._stats.total_running_seen,
            "deferred": self._stats.total_deferred,
            "defer_episodes": self._stats.total_defer_episodes,
            "new_defer_limit_suppressed": (
                self._stats.new_defer_limit_suppressed
            ),
            "defer_budget_suppressed": self._stats.defer_budget_suppressed,
            "defer_cooldown_suppressed": (
                self._stats.defer_cooldown_suppressed
            ),
            "defer_budget_resets": self._stats.defer_budget_resets,
            "force_resume": self._stats.total_force_resume,
            "min_output_suppressed": self._stats.min_output_suppressed,
            "policy_suppressed": self._stats.policy_suppressed,
            "entry_policy_suppressed": self._stats.entry_policy_suppressed,
            "policy_force_resume": self._stats.policy_force_resume,
            "empty_backlog_force_resume": self._stats.empty_backlog_force_resume,
            "time_budget_force_resume": self._stats.time_budget_force_resume,
            "prefill_priority_window_opens": (
                self._stats.prefill_priority_window_opens
            ),
            "prefill_priority_window_active_calls": (
                self._stats.prefill_priority_window_active_calls
            ),
            "prefill_priority_window_force_resume": (
                self._stats.prefill_priority_window_force_resume
            ),
            "global_suppressed_skip_calls": (
                self._stats.global_suppressed_skip_calls
            ),
            "max_consecutive_deferred": self._stats.max_consecutive_deferred,
            "requests_tracked": self._stats.requests_tracked,
            "deferred_step_ratio": (
                (self._stats.total_deferred / self._stats.total_running_seen)
                if self._stats.total_running_seen > 0
                else 0.0
            ),
            "avg_episode_steps": (
                (self._stats.total_deferred / self._stats.total_defer_episodes)
                if self._stats.total_defer_episodes > 0
                else 0.0
            ),
            "backlog_gate_active_calls": self._stats.backlog_gate_active_calls,
            "backlog_gate_skipped_calls": self._stats.backlog_gate_skipped_calls,
            "backlog_gate_warmup_calls": self._stats.backlog_gate_warmup_calls,
            "running_gate_skipped_calls": self._stats.running_gate_skipped_calls,
            "backlog_pressure_observed_calls": (
                self._stats.backlog_pressure_observed_calls
            ),
            "pressure_warmup_calls": self._stats.pressure_warmup_calls,
            "pressure_bypass_calls": self._stats.pressure_bypass_calls,
            "pressure_fast_warmup_pending_count": (
                self._pressure_fast_warmup_pending_count
            ),
            "pressure_fast_warmup_active_calls": (
                self._pressure_fast_warmup_active_calls
            ),
            "pressure_fast_warmup_pending_age_s": (
                self._pressure_fast_warmup_pending_age_s
            ),
            "pressure_fast_warmup_recent_arrivals": (
                self._pressure_fast_warmup_recent_arrivals
            ),
            "pressure_fast_warmup_pending_growth": (
                self._pressure_fast_warmup_pending_growth
            ),
            "pressure_fast_warmup_growth_hold_calls": (
                self._pressure_fast_warmup_growth_hold_calls
            ),
            "pressure_fast_warmup_require_all_signals": (
                self._pressure_fast_warmup_require_all_signals
            ),
            "pressure_fast_warmup_growth_latch": (
                self._pressure_fast_warmup_growth_latch
            ),
            "pressure_fast_warmup_calls": (
                self._stats.pressure_fast_warmup_calls
            ),
            "pressure_fast_warmup_age_signal_calls": (
                self._stats.pressure_fast_warmup_age_signal_calls
            ),
            "pressure_fast_warmup_recent_arrival_signal_calls": (
                self._stats.pressure_fast_warmup_recent_arrival_signal_calls
            ),
            "pressure_fast_warmup_recent_arrival_eligible_calls": (
                self._stats.pressure_fast_warmup_recent_arrival_eligible_calls
            ),
            "pressure_fast_warmup_all_signal_eligible_calls": (
                self._stats.pressure_fast_warmup_all_signal_eligible_calls
            ),
            "max_recent_pending_arrivals": (
                self._stats.max_recent_pending_arrivals
            ),
            "pressure_fast_warmup_growth_triggers": (
                self._stats.pressure_fast_warmup_growth_triggers
            ),
            "backlog_age_skipped_calls": self._stats.backlog_age_skipped_calls,
            "max_pending_age_s": self._stats.max_pending_age_s,
            "backlog_gate_active_ratio": (
                self._stats.backlog_gate_active_calls
                / max(1, self._stats.total_scheduler_calls)
            ),
        }

# ---------------------------------------------------------------------------
# Inline self-test (run with: python -m vllm_omni.core.sched.talker_lead_governor)
# ---------------------------------------------------------------------------

if __name__ == "__main__":  # pragma: no cover
    from types import SimpleNamespace

    # No-op when L_max is None / 0 / negative.
    for v in (None, 0, -1):
        g = TalkerLeadGovernor(v)
        assert not g.enabled
        assert g.filter_running([SimpleNamespace(request_id="r0", num_output_tokens=999)]) == []

    # ===== Basic admission flow (v2 compat with L_lo=L_hi) =====
    g = TalkerLeadGovernor(L_max=6, L_lo=6, speaker_rate=25)
    assert g.enabled and g.L_hi == 6 and g.L_lo == 6

    # t=0: req has produced 0 tokens, no audio_ttfp yet, must not be deferred.
    r = SimpleNamespace(request_id="r1", num_output_tokens=0)
    assert g.filter_running([r], now=0.0) == []

    # t=0.001: first token appears, audio_ttfp_server is captured.
    r.num_output_tokens = 1
    assert g.filter_running([r], now=0.001) == []
    assert g._state["r1"].audio_ttfp_server == 0.001

    # t=0.05 (50ms after ttfp -> p_est = 0.05 * 25 = 1.25 frame).
    # If g_cum = 8, lead = 8 - 1.25 = 6.75 >= 6 -> deferred.
    r.num_output_tokens = 8
    deferred = g.filter_running([r], now=0.051)
    assert deferred == [r], f"expected r1 deferred, got {deferred}"
    assert g._state["r1"].deferred_steps == 1
    assert g._state["r1"].is_deferred is True
    assert g._state["r1"].consecutive_deferred == 1

    # With L_lo == L_hi, exit only when lead < 6. lead=4-1.275=2.725 < 6 -> resume.
    r.num_output_tokens = 4
    assert g.filter_running([r], now=0.052) == []
    assert g._state["r1"].is_deferred is False
    assert g._state["r1"].deferred_steps == 1  # unchanged

    # Backlog gate: lead would trigger again, but pending_count=0 blocks entry.
    r.num_output_tokens = 12
    snap_before = g.snapshot()
    assert g.filter_running([r], now=0.053, pending_count=0) == []
    assert g._state["r1"].is_deferred is False
    snap_after = g.snapshot()
    assert snap_after["backlog_gate_skipped_calls"] == \
        snap_before["backlog_gate_skipped_calls"] + 1
    # And with backlog>0 the same lead must trigger entry.
    assert g.filter_running([r], now=0.054, pending_count=2) == [r]
    assert g._state["r1"].is_deferred is True

    # on_request_finished returns final counters and clears state.
    final = g.on_request_finished("r1")
    assert final is not None and final["deferred_steps"] >= 2
    assert g.on_request_finished("r1") is None  # idempotent

    # ===== v3 hysteresis specific =====
    g3 = TalkerLeadGovernor(L_max=12, L_lo=6, hold_max=4, speaker_rate=25)
    assert g3.L_hi == 12 and g3.L_lo == 6 and g3.hold_max == 4

    # Default L_lo derivation when not provided.
    g3d = TalkerLeadGovernor(L_max=12)
    assert g3d.L_lo == 6

    # Capture ttfp.
    r2 = SimpleNamespace(request_id="r2", num_output_tokens=1)
    g3.filter_running([r2], now=0.0, pending_count=2)
    # Push lead well above L_hi: g_cum=20, p_est=0 -> lead=20 >= 12 -> ENTER.
    r2.num_output_tokens = 20
    out = g3.filter_running([r2], now=0.001, pending_count=2)
    assert out == [r2]
    assert g3._state["r2"].is_deferred is True
    assert g3._state["r2"].consecutive_deferred == 1
    assert g3._state["r2"].defer_episodes == 1

    # lead is still high (slightly drained by p_est increment) -> STAY deferred,
    # consecutive_deferred should accumulate, NOT trigger a new episode.
    r2.num_output_tokens = 20
    out = g3.filter_running([r2], now=0.002, pending_count=2)
    assert out == [r2]
    assert g3._state["r2"].consecutive_deferred == 2
    assert g3._state["r2"].defer_episodes == 1  # no new episode

    # Drop lead below L_lo: g_cum=2, big elapsed -> lead<<6 -> RESUME.
    r2.num_output_tokens = 2
    out = g3.filter_running([r2], now=0.500, pending_count=2)
    assert out == []
    assert g3._state["r2"].is_deferred is False
    assert g3._state["r2"].consecutive_deferred == 0
    snap = g3.snapshot()
    assert snap["defer_episodes"] == 1
    # Each "stay deferred" step counts; 1 entry + 1 stay = 2 deferred ticks.
    assert snap["deferred"] == 2

    # Per-call new-defer limit caps only fresh RUNNING->DEFERRED transitions.
    # Existing deferred requests still remain deferred on later calls.
    g3l = TalkerLeadGovernor(
        L_max=12,
        L_lo=6,
        hold_max=4,
        speaker_rate=25,
        max_new_defer_per_call=1,
    )
    r2a = SimpleNamespace(request_id="r2a", num_output_tokens=1)
    r2b = SimpleNamespace(request_id="r2b", num_output_tokens=1)
    g3l.filter_running([r2a, r2b], now=0.0, pending_count=2)
    r2a.num_output_tokens = 20
    r2b.num_output_tokens = 20
    assert g3l.filter_running([r2a, r2b], now=0.001, pending_count=2) == [r2a]
    snap3l = g3l.snapshot()
    assert snap3l["defer_episodes"] == 1
    assert snap3l["new_defer_limit_suppressed"] == 1
    assert g3l.filter_running([r2a, r2b], now=0.002, pending_count=2) == [
        r2a,
        r2b,
    ]
    snap3l = g3l.snapshot()
    assert snap3l["defer_episodes"] == 2
    assert snap3l["new_defer_limit_suppressed"] == 1

    # Pressure-window defer budget caps fresh episodes across calls, then
    # resets after the backlog has stayed empty long enough.
    g3b = TalkerLeadGovernor(
        L_max=12,
        L_lo=6,
        hold_max=4,
        speaker_rate=25,
        max_defer_episodes_per_window=1,
        defer_budget_empty_reset_steps=1,
    )
    r2c = SimpleNamespace(request_id="r2c", num_output_tokens=1)
    r2d = SimpleNamespace(request_id="r2d", num_output_tokens=1)
    g3b.filter_running([r2c, r2d], now=0.0, pending_count=2)
    r2c.num_output_tokens = 20
    r2d.num_output_tokens = 20
    assert g3b.filter_running([r2c, r2d], now=0.001, pending_count=2) == [r2c]
    snap3b = g3b.snapshot()
    assert snap3b["defer_episodes"] == 1
    assert snap3b["defer_budget_epoch_episodes"] == 1
    assert snap3b["defer_budget_suppressed"] == 1
    assert g3b.filter_running([r2d], now=0.002, pending_count=2) == []
    snap3b = g3b.snapshot()
    assert snap3b["defer_budget_suppressed"] == 2
    g3b.filter_running([r2d], now=0.003, pending_count=0)
    assert g3b.snapshot()["defer_budget_resets"] == 0
    g3b.observe_idle(pending_count=0)
    snap3b = g3b.snapshot()
    assert snap3b["defer_budget_resets"] == 1
    assert snap3b["defer_budget_epoch_episodes"] == 0
    assert g3b.filter_running([r2d], now=0.004, pending_count=2) == [r2d]

    # Per-request cooldown blocks immediate re-entry after a request resumes,
    # but only for that same request and only for fresh entry episodes.
    g3c = TalkerLeadGovernor(
        L_max=12,
        L_lo=6,
        hold_max=1,
        speaker_rate=25,
        defer_cooldown_steps=2,
    )
    r2e = SimpleNamespace(request_id="r2e", num_output_tokens=1)
    g3c.filter_running([r2e], now=0.0, pending_count=2)
    r2e.num_output_tokens = 20
    assert g3c.filter_running([r2e], now=0.001, pending_count=2) == [r2e]
    assert g3c.filter_running([r2e], now=0.002, pending_count=2) == []
    assert g3c._state["r2e"].is_deferred is False
    assert g3c._state["r2e"].defer_cooldown_remaining == 2
    assert g3c.filter_running([r2e], now=0.003, pending_count=2) == []
    assert g3c.filter_running([r2e], now=0.004, pending_count=2) == []
    snap3c = g3c.snapshot()
    assert snap3c["defer_episodes"] == 1
    assert snap3c["defer_cooldown_suppressed"] == 2
    assert g3c.filter_running([r2e], now=0.005, pending_count=2) == [r2e]
    assert g3c.snapshot()["defer_episodes"] == 2

    # ===== hold_max anti-starvation =====
    g4 = TalkerLeadGovernor(L_max=12, L_lo=6, hold_max=3, speaker_rate=25)
    r3 = SimpleNamespace(request_id="r3", num_output_tokens=1)
    g4.filter_running([r3], now=0.0, pending_count=2)
    # Stay continuously above L_hi by always-large g_cum.
    r3.num_output_tokens = 999
    g4.filter_running([r3], now=0.001, pending_count=2)  # ENTER, consec=1
    g4.filter_running([r3], now=0.002, pending_count=2)  # STAY, consec=2
    g4.filter_running([r3], now=0.003, pending_count=2)  # STAY, consec=3 == hold_max
    # Next call: consecutive_deferred (=3) >= hold_max -> force resume.
    out = g4.filter_running([r3], now=0.004, pending_count=2)
    assert out == [], f"expected force-resume, got deferred={out}"
    assert g4._state["r3"].is_deferred is False
    assert g4.snapshot()["force_resume"] == 1

    # After force resume, lead still huge -> next call re-enters DEFERRED
    # with a new episode counter.
    g4.filter_running([r3], now=0.005, pending_count=2)
    assert g4._state["r3"].is_deferred is True
    assert g4._state["r3"].defer_episodes == 2

    # ===== adaptive policy gate =====
    g5 = TalkerLeadGovernor(L_max=4, L_lo=2, hold_max=4, speaker_rate=25)
    r4 = SimpleNamespace(request_id="r4", num_output_tokens=1)
    g5.filter_running([r4], now=0.0, pending_count=1)
    r4.num_output_tokens = 10
    assert g5.filter_running([r4], now=0.001, pending_count=1) == [r4]
    assert g5._state["r4"].is_deferred is True
    assert g5.filter_running([r4], now=0.002, pending_count=1, defer_allowed=lambda _r: False) == []
    assert g5._state["r4"].is_deferred is False
    assert g5.snapshot()["policy_force_resume"] == 1

    # Entry-only policy blocks new defer episodes but does not force-resume a
    # request that is already deferred.
    g5b = TalkerLeadGovernor(L_max=4, L_lo=2, hold_max=4, speaker_rate=25)
    r4b = SimpleNamespace(request_id="r4b", num_output_tokens=1)
    g5b.filter_running([r4b], now=0.0, pending_count=1)
    r4b.num_output_tokens = 10
    assert g5b.filter_running([r4b], now=0.001, pending_count=1) == [r4b]
    assert g5b.filter_running(
        [r4b],
        now=0.002,
        pending_count=1,
        defer_entry_allowed=lambda _r: False,
    ) == [r4b]
    assert g5b._state["r4b"].is_deferred is True
    r4c = SimpleNamespace(request_id="r4c", num_output_tokens=10)
    assert g5b.filter_running(
        [r4c],
        now=0.003,
        pending_count=1,
        defer_entry_allowed=lambda _r: False,
    ) == []
    snap5b = g5b.snapshot()
    assert snap5b["entry_policy_suppressed"] == 1
    assert snap5b["policy_force_resume"] == 0

    # ===== adaptive empty-backlog resume =====
    g6 = TalkerLeadGovernor(
        L_max=8, L_lo=4, hold_max=100, speaker_rate=25, resume_on_empty_backlog=True
    )
    r5 = SimpleNamespace(request_id="r5", num_output_tokens=1)
    g6.filter_running([r5], now=0.0, pending_count=1)
    r5.num_output_tokens = 20
    assert g6.filter_running([r5], now=0.001, pending_count=1) == [r5]
    assert g6._state["r5"].is_deferred is True
    assert g6.filter_running([r5], now=0.002, pending_count=0) == []
    assert g6._state["r5"].is_deferred is False
    assert g6.snapshot()["empty_backlog_force_resume"] == 1

    # Grace keeps a deferred request withheld for a bounded number of empty
    # backlog calls before early resume.
    g7 = TalkerLeadGovernor(
        L_max=8,
        L_lo=4,
        hold_max=100,
        speaker_rate=25,
        resume_on_empty_backlog=True,
        empty_backlog_grace_steps=2,
    )
    r6 = SimpleNamespace(request_id="r6", num_output_tokens=1)
    g7.filter_running([r6], now=0.0, pending_count=1)
    r6.num_output_tokens = 20
    assert g7.filter_running([r6], now=0.001, pending_count=1) == [r6]
    assert g7.filter_running([r6], now=0.002, pending_count=0) == [r6]
    assert g7.filter_running([r6], now=0.003, pending_count=0) == [r6]
    assert g7.filter_running([r6], now=0.004, pending_count=0) == []
    assert g7.snapshot()["empty_backlog_force_resume"] == 1

    # ===== sustained-backlog gate =====
    g8 = TalkerLeadGovernor(
        L_max=8,
        L_lo=4,
        hold_max=100,
        speaker_rate=25,
        min_backlog_active_steps=3,
    )
    r7 = SimpleNamespace(request_id="r7", num_output_tokens=1)
    g8.filter_running([r7], now=0.0, pending_count=1)
    r7.num_output_tokens = 20
    assert g8.filter_running([r7], now=0.001, pending_count=1) == []
    assert g8.filter_running([r7], now=0.002, pending_count=1) == [r7]
    snap8 = g8.snapshot()
    assert snap8["backlog_gate_warmup_calls"] >= 1
    assert g8._state["r7"].is_deferred is True

    # ===== running-concurrency gate =====
    g8r = TalkerLeadGovernor(
        L_max=8,
        L_lo=4,
        hold_max=100,
        speaker_rate=25,
        min_running_count=2,
    )
    r7r = SimpleNamespace(request_id="r7r", num_output_tokens=1)
    g8r.filter_running([r7r], now=0.0, pending_count=1)
    r7r.num_output_tokens = 20
    assert g8r.filter_running([r7r], now=0.001, pending_count=1) == []
    snap8r = g8r.snapshot()
    assert snap8r["running_gate_skipped_calls"] >= 1
    r7r_b = SimpleNamespace(request_id="r7r_b", num_output_tokens=20)
    out8r = g8r.filter_running([r7r, r7r_b], now=0.002, pending_count=1)
    assert r7r in out8r

    # ===== cumulative pressure warmup gate =====
    g8p = TalkerLeadGovernor(
        L_max=8,
        L_lo=4,
        hold_max=100,
        speaker_rate=25,
        min_pressure_active_calls=3,
    )
    r7p = SimpleNamespace(request_id="r7p", num_output_tokens=1)
    g8p.filter_running([r7p], now=0.0, pending_count=1)
    r7p.num_output_tokens = 20
    assert g8p.filter_running([r7p], now=0.001, pending_count=1) == []
    assert g8p.filter_running([r7p], now=0.002, pending_count=1) == [r7p]
    snap8p = g8p.snapshot()
    assert snap8p["pressure_warmup_calls"] >= 1
    assert snap8p["backlog_pressure_observed_calls"] >= 3

    # Burst bypass opens the gate before cumulative warmup completes when the
    # waiting queue is clearly deep enough.
    g8pb = TalkerLeadGovernor(
        L_max=8,
        L_lo=4,
        hold_max=100,
        speaker_rate=25,
        min_pressure_active_calls=10,
        pressure_bypass_pending_count=4,
    )
    r7pb = SimpleNamespace(request_id="r7pb", num_output_tokens=1)
    g8pb.filter_running([r7pb], now=0.0, pending_count=1)
    r7pb.num_output_tokens = 20
    assert g8pb.filter_running([r7pb], now=0.001, pending_count=1) == []
    assert g8pb.filter_running([r7pb], now=0.002, pending_count=4) == [r7pb]
    snap8pb = g8pb.snapshot()
    assert snap8pb["pressure_bypass_calls"] == 1
    assert snap8pb["pressure_warmup_calls"] >= 1

    # Fast warmup can use oldest-pending age as a softer alternative to
    # immediate age bypass: it still waits for cumulative pressure calls.
    g8fa = TalkerLeadGovernor(
        L_max=8,
        L_lo=4,
        hold_max=100,
        speaker_rate=25,
        min_pressure_active_calls=6,
        pressure_fast_warmup_active_calls=3,
        pressure_fast_warmup_pending_age_s=0.050,
    )
    r7fa = SimpleNamespace(request_id="r7fa", num_output_tokens=1)
    g8fa.filter_running([r7fa], now=0.0, pending_count=1, pending_age_s=0.010)
    r7fa.num_output_tokens = 20
    assert g8fa.filter_running(
        [r7fa], now=0.001, pending_count=1, pending_age_s=0.020
    ) == []
    assert g8fa.filter_running(
        [r7fa], now=0.002, pending_count=1, pending_age_s=0.050
    ) == [r7fa]
    snap8fa = g8fa.snapshot()
    assert snap8fa["pressure_fast_warmup_calls"] == 1
    assert snap8fa["pressure_fast_warmup_age_signal_calls"] == 1

    # Recent waiting arrivals can also use the shorter warmup without relying
    # on one-step pending growth or oldest-age threshold alone.
    g8fr = TalkerLeadGovernor(
        L_max=8,
        L_lo=4,
        hold_max=100,
        speaker_rate=25,
        min_pressure_active_calls=6,
        pressure_fast_warmup_active_calls=3,
        pressure_fast_warmup_recent_arrivals=4,
    )
    r7fr = SimpleNamespace(request_id="r7fr", num_output_tokens=1)
    g8fr.filter_running([r7fr], now=0.0, pending_count=1, recent_pending_arrivals=2)
    r7fr.num_output_tokens = 20
    assert g8fr.filter_running(
        [r7fr], now=0.001, pending_count=1, recent_pending_arrivals=2
    ) == []
    assert g8fr.filter_running(
        [r7fr], now=0.002, pending_count=1, recent_pending_arrivals=4
    ) == [r7fr]
    snap8fr = g8fr.snapshot()
    assert snap8fr["pressure_fast_warmup_calls"] == 1
    assert snap8fr["pressure_fast_warmup_recent_arrival_signal_calls"] == 1

    # All-signal mode requires every configured fast-warmup signal, so recent
    # arrivals alone and age alone are not enough to shorten the warmup.
    g8fc = TalkerLeadGovernor(
        L_max=8,
        L_lo=4,
        hold_max=100,
        speaker_rate=25,
        min_pressure_active_calls=6,
        pressure_fast_warmup_active_calls=3,
        pressure_fast_warmup_pending_age_s=0.050,
        pressure_fast_warmup_recent_arrivals=4,
        pressure_fast_warmup_require_all_signals=True,
    )
    r7fc = SimpleNamespace(request_id="r7fc", num_output_tokens=1)
    g8fc.filter_running(
        [r7fc],
        now=0.0,
        pending_count=1,
        pending_age_s=0.050,
        recent_pending_arrivals=2,
    )
    r7fc.num_output_tokens = 20
    assert g8fc.filter_running(
        [r7fc],
        now=0.001,
        pending_count=1,
        pending_age_s=0.020,
        recent_pending_arrivals=4,
    ) == []
    assert g8fc.filter_running(
        [r7fc],
        now=0.002,
        pending_count=1,
        pending_age_s=0.050,
        recent_pending_arrivals=4,
    ) == [r7fc]
    snap8fc = g8fc.snapshot()
    assert snap8fc["pressure_fast_warmup_calls"] == 1
    assert snap8fc["pressure_fast_warmup_age_signal_calls"] == 1
    assert snap8fc["pressure_fast_warmup_recent_arrival_signal_calls"] == 1
    assert snap8fc["pressure_fast_warmup_all_signal_eligible_calls"] == 1

    # Fast warmup can be guarded by a pending-depth growth signal. A merely
    # deep but flat queue must not use the shorter warmup until growth fires.
    g8fg = TalkerLeadGovernor(
        L_max=8,
        L_lo=4,
        hold_max=100,
        speaker_rate=25,
        min_pressure_active_calls=6,
        pressure_fast_warmup_pending_count=4,
        pressure_fast_warmup_active_calls=3,
        pressure_fast_warmup_pending_growth=2,
        pressure_fast_warmup_growth_hold_calls=2,
    )
    r7fg = SimpleNamespace(request_id="r7fg", num_output_tokens=1)
    g8fg.filter_running([r7fg], now=0.0, pending_count=4)
    r7fg.num_output_tokens = 20
    assert g8fg.filter_running([r7fg], now=0.001, pending_count=4) == []
    assert g8fg.filter_running([r7fg], now=0.002, pending_count=4) == []
    assert g8fg.snapshot()["pressure_fast_warmup_calls"] == 0
    assert g8fg.filter_running([r7fg], now=0.003, pending_count=6) == [r7fg]
    snap8fg = g8fg.snapshot()
    assert snap8fg["pressure_fast_warmup_calls"] == 1
    assert snap8fg["pressure_fast_warmup_growth_triggers"] == 1

    # Pending-growth latch should survive age-gate warmup ticks and be spent
    # only once the backlog is old enough to count as pressure.
    g8fga = TalkerLeadGovernor(
        L_max=8,
        L_lo=4,
        hold_max=100,
        speaker_rate=25,
        min_pressure_active_calls=6,
        pressure_fast_warmup_pending_count=4,
        pressure_fast_warmup_active_calls=2,
        pressure_fast_warmup_pending_growth=2,
        pressure_fast_warmup_growth_hold_calls=2,
        min_pending_age_s=0.050,
    )
    r7fga = SimpleNamespace(request_id="r7fga", num_output_tokens=1)
    g8fga.filter_running([r7fga], now=0.0, pending_count=4, pending_age_s=0.0)
    r7fga.num_output_tokens = 20
    assert g8fga.filter_running(
        [r7fga], now=0.001, pending_count=6, pending_age_s=0.010
    ) == []
    assert g8fga.snapshot()["pressure_fast_warmup_growth_latch"] == 2
    assert g8fga.filter_running(
        [r7fga], now=0.002, pending_count=1, pending_age_s=0.050
    ) == []
    assert g8fga.filter_running(
        [r7fga], now=0.003, pending_count=1, pending_age_s=0.060
    ) == [r7fga]

    # ===== backlog-age gate =====
    g8a = TalkerLeadGovernor(
        L_max=8,
        L_lo=4,
        hold_max=100,
        speaker_rate=25,
        min_pending_age_s=0.050,
    )
    r7a = SimpleNamespace(request_id="r7a", num_output_tokens=1)
    g8a.filter_running([r7a], now=0.0, pending_count=1, pending_age_s=0.010)
    r7a.num_output_tokens = 20
    assert g8a.filter_running([r7a], now=0.001, pending_count=1, pending_age_s=0.010) == []
    snap8a = g8a.snapshot()
    assert snap8a["backlog_age_skipped_calls"] >= 1
    assert g8a.filter_running([r7a], now=0.002, pending_count=1, pending_age_s=0.050) == [r7a]
    assert g8a._state["r7a"].is_deferred is True

    # ===== first-audible-packet guard =====
    g9 = TalkerLeadGovernor(
        L_max=8,
        L_lo=4,
        hold_max=100,
        speaker_rate=25,
        min_output_tokens_before_defer=16,
    )
    r8 = SimpleNamespace(request_id="r8", num_output_tokens=1)
    g9.filter_running([r8], now=0.0, pending_count=1)
    r8.num_output_tokens = 15
    assert g9.filter_running([r8], now=0.001, pending_count=1) == []
    assert g9.snapshot()["min_output_suppressed"] == 1
    r8.num_output_tokens = 16
    assert g9.filter_running([r8], now=0.002, pending_count=1) == [r8]

    print("[talker_lead_governor v3] self-test OK", g3.snapshot())
