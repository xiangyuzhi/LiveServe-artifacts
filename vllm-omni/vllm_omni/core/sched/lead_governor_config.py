from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class LeadGovernorParams:
    L_max: int | None
    L_lo: int | None = None
    hold_max: int = 100
    speaker_rate: int = 25
    min_pending_count: int = 1
    min_backlog_active_steps: int = 1
    min_running_count: int = 1
    min_pressure_active_calls: int = 0
    pressure_bypass_pending_count: int = 0
    pressure_bypass_pending_age_s: float = 0.0
    pressure_fast_warmup_pending_count: int = 0
    pressure_fast_warmup_active_calls: int = 0
    pressure_fast_warmup_pending_age_s: float = 0.0
    pressure_fast_warmup_recent_arrivals: int = 0
    pressure_fast_warmup_pending_growth: int = 0
    pressure_fast_warmup_growth_hold_calls: int = 0
    pressure_fast_warmup_require_all_signals: bool = False
    resume_on_empty_backlog: bool = False
    empty_backlog_grace_steps: int = 0
    min_output_tokens_before_defer: int = 0
    min_pending_age_s: float = 0.0
    max_new_defer_per_call: int = 0
    max_defer_episodes_per_window: int = 0
    defer_budget_empty_reset_steps: int = 0
    defer_cooldown_steps: int = 0
    prefill_priority_window_steps: int = 0

    def as_governor_kwargs(self) -> dict[str, Any]:
        return asdict(self)
