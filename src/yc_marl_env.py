from __future__ import annotations

"""Final V4 cooperative MARL environment for the ACK 2026 yard study.

Key design choices
------------------
* Import containers only.
* 4 blocks x 25 stacks x 4 tiers = 400 physical slots.
* One fixed YC per block; every physical move takes 2 minutes.
* One fixed stochastic operating condition: storage and retrieval requests are
  independent Poisson streams with the same mean rate ``arrival_rate_per_hour``.
* New inbound containers are storage-only in the current 480-minute episode.
* Initial-stock containers are the only same-episode retrieval targets.
* Truck ETA is noisy and becomes more accurate as actual arrival approaches.
* Storage Agent selects one of 100 stacks.
* Shared YC policy selects Mandatory, Idle (only when speculative work is the
  only alternative), or a flat proactive (Target x Destination) relocation.
* Canonical mode learns proactive target and relocation destination jointly.
* Diagnostic rule-resolved mode can restrict proactive support to one transparent
  ETA/blocker-priority target plus information-aware relocation destination.
* Capacity slack is an observation only, never a hard action mask.
* Reward is normalized to mean delay / retrieval-normalized extra YC minutes.
"""

from collections import deque
from dataclasses import dataclass
import heapq
import math
import random
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

from yard_simulator import Container, InformationAwareHeuristic, Task, YardSimulator


# ---------------------------------------------------------------------------
# Fixed V4 environment constants
# ---------------------------------------------------------------------------

N_BLOCKS = 4
STACKS_PER_BLOCK = 25
MAX_TIER = 4
TOTAL_STACKS = N_BLOCKS * STACKS_PER_BLOCK
TOTAL_SLOTS = TOTAL_STACKS * MAX_TIER
INITIAL_CONTAINERS = 268
INITIAL_PER_BLOCK = 67
OPERATING_HORIZON_MIN = 480.0
YC_MOVE_TIME_MIN = 2.0
PROACTIVE_HORIZON_MIN = 180.0
ETA_UPDATE_INTERVAL_MIN = 30.0
FUTURE_PICKUP_MEDIAN_HOURS = 72.0
FUTURE_PICKUP_LOG_SIGMA = 0.5
FUTURE_PICKUP_MIN_HOURS = 24.0
FUTURE_PICKUP_MAX_HOURS = 168.0
RELOCATION_SLACK_SLOTS_PER_BLOCK = 3

STORAGE_AGENT = "storage"
YC_AGENT_PREFIX = "yc_"

# V4 flat YC action semantics.
YC_MANDATORY = 0
YC_IDLE = 1
YC_PROACTIVE_BASE = 2
YC_TARGET_POSITIONS = STACKS_PER_BLOCK * MAX_TIER  # 100 positions / block
YC_DEST_STACKS = STACKS_PER_BLOCK                  # 25 destinations / block
YC_PAIR_COUNT = YC_TARGET_POSITIONS * YC_DEST_STACKS  # 2500
YC_ACTION_DIM = YC_PROACTIVE_BASE + YC_PAIR_COUNT      # 2502

# Legacy aliases kept only so old auxiliary scripts fail gracefully rather than
# at import time. They no longer denote independent actions in V4.
YC_RETRIEVAL = YC_MANDATORY
YC_STORAGE = YC_MANDATORY


def encode_proactive_action(target_pos: int, dest_stack: int) -> int:
    if not 0 <= target_pos < YC_TARGET_POSITIONS:
        raise ValueError(target_pos)
    if not 0 <= dest_stack < YC_DEST_STACKS:
        raise ValueError(dest_stack)
    return YC_PROACTIVE_BASE + target_pos * YC_DEST_STACKS + dest_stack


def decode_proactive_action(action: int) -> Tuple[int, int]:
    if not YC_PROACTIVE_BASE <= action < YC_ACTION_DIM:
        raise ValueError(action)
    k = action - YC_PROACTIVE_BASE
    return k // YC_DEST_STACKS, k % YC_DEST_STACKS


def is_proactive_action(action: int) -> bool:
    return YC_PROACTIVE_BASE <= action < YC_ACTION_DIM


@dataclass
class ResourceDecision:
    kind: str  # "storage" or "yc"
    cid: Optional[str] = None
    candidates: Optional[List[Tuple[int, int]]] = None
    block: Optional[int] = None


class ResourceMarlSimulator(YardSimulator):
    """Event-driven simulator that pauses at storage and YC decision epochs."""

    def __init__(
        self,
        *args,
        proactive_horizon: float = PROACTIVE_HORIZON_MIN,
        operating_horizon: float = OPERATING_HORIZON_MIN,
        enable_proactive: bool = True,
        rule_resolve_proactive_pair: bool = False,
        **kwargs,
    ):
        # Keep three physical slots per block unavailable to new storage so a
        # buried retrieval target always has at least one external relocation
        # destination. Physical yard capacity remains 400 slots; the admission
        # policy preserves 3 relocation-slack slots in each 100-slot block.
        kwargs.setdefault("relocation_buffer_slots_per_block", RELOCATION_SLACK_SLOTS_PER_BLOCK)
        super().__init__(*args, **kwargs)
        self.proactive_horizon = float(proactive_horizon)
        self.operating_horizon = float(operating_horizon)
        self.enable_proactive = bool(enable_proactive)
        self.rule_resolve_proactive_pair = bool(rule_resolve_proactive_pair)
        self.operating_closed = False

        self.pending_decision: Optional[ResourceDecision] = None
        self.pending_storage_queue: Deque[str] = deque()
        self._storage_retry_scheduled = False
        self._yc_decision_blocks: Deque[int] = deque()
        self._yc_decision_set: set[int] = set()

        self.truck_wait_area = 0.0
        self.storage_wait_area = 0.0
        self.n_storage_requests = 0
        self.n_retrieval_requests = 0
        self.arrival_rate_per_hour = math.nan

    # ------------------------------------------------------------------
    # Time-integrated costs
    # ------------------------------------------------------------------
    def _waiting_counts(self) -> Tuple[int, int]:
        truck_waiting = sum(
            1
            for c in self.containers.values()
            if c.retrievable_this_episode
            and c.truck_arrived
            and c.retrieval_complete is None
        )
        storage_waiting = sum(
            1
            for c in self.containers.values()
            if c.cohort == "new_inbound"
            and c.yard_arrival <= self.now
            and c.storage_complete is None
        )
        return truck_waiting, storage_waiting

    def _advance_clock(self, new_time: float) -> None:
        if new_time < self.now - 1e-9:
            raise RuntimeError("Event time moved backwards")
        dt = max(0.0, new_time - self.now)
        if dt > 0:
            tw, sw = self._waiting_counts()
            self.truck_wait_area += tw * dt
            self.storage_wait_area += sw * dt
        self.now = float(new_time)

    # ------------------------------------------------------------------
    # Queue mechanics
    # ------------------------------------------------------------------
    def submit_task(self, block: int, task: Task, front: bool = False) -> None:
        yc = self.ycs[block]
        if front:
            yc.queue.appendleft(task)
        else:
            yc.queue.append(task)
        yc.max_queue_len = max(yc.max_queue_len, len(yc.queue))
        self.log(
            "YC_TASK_ENQUEUED",
            task.container_id,
            block=block,
            task=task.kind,
            queue_len=len(yc.queue),
        )
        self.request_yc_decision(block)

    def _finish_task(self, block: int) -> None:
        yc = self.ycs[block]
        task = yc.current
        if task is None:
            raise RuntimeError("YC completion without active task")
        yc.busy_time += task.duration
        yc.completed_moves += 1
        self._apply_task_resource_controlled(block, task)
        self.log("YC_TASK_COMPLETE", task.container_id, block=block, task=task.kind)
        yc.busy = False
        yc.current = None
        self._schedule_storage_retry()
        self.request_yc_decision(block)

    def _apply_task_resource_controlled(self, block: int, task: Task) -> None:
        c = self.containers[task.container_id]

        if task.kind == "STORAGE":
            assert task.target_block is not None and task.target_stack is not None
            self.yard.release_reservation(task.target_block, task.target_stack)
            self.yard.push(task.target_block, task.target_stack, c.cid)
            c.block = task.target_block
            c.stack = task.target_stack
            c.stored = True
            c.storage_complete = self.now
            self.storage_moves += 1
            self.log("STORAGE_COMPLETE", c.cid, block=c.block, stack=c.stack)
            return

        if task.kind not in {"RETRIEVAL_STEP", "PREMARSHALL_STEP"}:
            raise ValueError(f"Unknown task kind: {task.kind}")

        if not c.stored or c.block is None or c.stack is None:
            c.premarshal_pending = False
            return

        source_stack = c.stack
        blockers = self.yard.blockers_above(c.cid, self.containers)
        if blockers > 0:
            moving_id = self.yard.top(block, source_stack)
            if moving_id is None or moving_id == c.cid:
                raise RuntimeError("Blocker state inconsistent")

            if task.kind == "PREMARSHALL_STEP" and task.target_stack is not None:
                dest_stack = int(task.target_stack)
                self.yard.release_reservation(block, dest_stack)
                if dest_stack not in self.yard.feasible_relocation_stacks(block, source_stack):
                    raise RuntimeError(
                        f"Selected proactive destination became infeasible: "
                        f"block={block} source={source_stack} dest={dest_stack}"
                    )
            else:
                # Reactive retrieval rehandling is intentionally outside the
                # learned proactive-relocation decision scope.
                dest_stack = self.policy.choose_relocation_destination(
                    self, moving_id, block, source_stack
                )

            popped = self.yard.pop_top(block, source_stack)
            assert popped == moving_id
            self.yard.push(block, dest_stack, moving_id)
            moving = self.containers[moving_id]
            moving.block = block
            moving.stack = dest_stack

            if task.kind == "PREMARSHALL_STEP":
                self.proactive_moves += 1
                c.premarshal_pending = False
                self.log(
                    "PROACTIVE_RELOCATION_COMPLETE",
                    moving_id,
                    block=block,
                    source_stack=source_stack,
                    dest_stack=dest_stack,
                    target=c.cid,
                )
            else:
                self.rehandling_moves += 1
                self.log(
                    "REHANDLING_COMPLETE",
                    moving_id,
                    block=block,
                    source_stack=source_stack,
                    dest_stack=dest_stack,
                    target=c.cid,
                )
                # Retrieval remains mandatory after each blocker move.
                self.submit_task(
                    block,
                    Task("RETRIEVAL_STEP", c.cid, self.move_time, created_at=self.now),
                    front=True,
                )
            return

        if task.kind == "PREMARSHALL_STEP":
            c.premarshal_pending = False
            self.log("PREMARSHALL_READY", c.cid, block=block, stack=c.stack)
            return

        popped = self.yard.pop_top(block, source_stack)
        assert popped == c.cid
        c.stored = False
        c.retrieval_complete = self.now
        self.retrieval_moves += 1
        self.log("RETRIEVAL_COMPLETE", c.cid, block=block, stack=source_stack)

    # ------------------------------------------------------------------
    # Storage decisions
    # ------------------------------------------------------------------
    def _on_container_arrival(self, cid: str) -> None:
        self.log("CONTAINER_ARRIVAL", cid)
        candidates = self.yard.feasible_storage_slots()
        if not candidates:
            self.pending_storage_queue.append(cid)
            self.log("STORAGE_DEFERRED", cid, reason="yard_full")
            return
        self.pending_decision = ResourceDecision("storage", cid=cid, candidates=list(candidates))

    def _schedule_storage_retry(self) -> None:
        if self.pending_storage_queue and not self._storage_retry_scheduled:
            if self.yard.feasible_storage_slots():
                self._storage_retry_scheduled = True
                self.schedule(self.now, "STORAGE_RETRY")

    def _on_storage_retry(self) -> None:
        self._storage_retry_scheduled = False
        if self.pending_decision is not None or not self.pending_storage_queue:
            return
        candidates = self.yard.feasible_storage_slots()
        if not candidates:
            return
        cid = self.pending_storage_queue.popleft()
        self.log("STORAGE_RETRY", cid, waiting_queue=len(self.pending_storage_queue))
        self.pending_decision = ResourceDecision("storage", cid=cid, candidates=list(candidates))

    def storage_action_mask(self) -> np.ndarray:
        n = self.yard.n_blocks * self.yard.stacks_per_block
        mask = np.zeros(n, dtype=np.bool_)
        d = self.pending_decision
        if d is None or d.kind != "storage":
            return mask
        for b, s in d.candidates or []:
            mask[b * self.yard.stacks_per_block + s] = True
        return mask

    def apply_storage_action(self, action: int) -> None:
        d = self.pending_decision
        if d is None or d.kind != "storage" or d.cid is None:
            raise RuntimeError("No storage decision is pending")
        mask = self.storage_action_mask()
        if action < 0 or action >= len(mask) or not bool(mask[action]):
            raise ValueError(f"Invalid storage action {action}")
        b = action // self.yard.stacks_per_block
        s = action % self.yard.stacks_per_block
        cid = d.cid
        self.pending_decision = None
        self.yard.reserve(b, s)
        self.log("STORAGE_DECISION", cid, selected_block=b, selected_stack=s)
        self.submit_task(
            b,
            Task(
                "STORAGE",
                cid,
                self.move_time,
                target_block=b,
                target_stack=s,
                created_at=self.now,
            ),
        )

    # ------------------------------------------------------------------
    # ETA and proactive candidates
    # ------------------------------------------------------------------
    def _on_eta_update(self, cid: str, new_eta: float) -> None:
        if self.operating_closed:
            return
        c = self.containers[cid]
        c.prev_eta = c.eta
        c.eta = float(new_eta)
        c.eta_delta = c.eta - c.prev_eta
        c.last_eta_update = self.now
        self.log(
            "TRUCK_ETA_UPDATE",
            cid,
            old_eta=c.prev_eta,
            new_eta=c.eta,
            eta_delta=c.eta_delta,
        )
        if c.block is not None:
            self.request_yc_decision(c.block)

    def _proactive_eligible(self, cid: str) -> bool:
        if not self.enable_proactive or self.operating_closed:
            return False
        c = self.containers[cid]
        if (
            not c.retrievable_this_episode
            or not c.stored
            or c.block is None
            or c.stack is None
            or c.retrieval_complete is not None
            or c.retrieval_requested
            or c.truck_arrived
            or c.premarshal_pending
        ):
            return False
        lead = c.eta - self.now
        if not (0.0 < lead <= self.proactive_horizon):
            return False
        return self.yard.blockers_above(cid, self.containers) > 0

    def proactive_candidates(self, block: int) -> List[str]:
        if not self.enable_proactive or self.operating_closed:
            return []
        candidates = [
            cid
            for cid, c in self.containers.items()
            if c.block == block and self._proactive_eligible(cid)
        ]
        candidates.sort(key=self._proactive_priority)
        return candidates

    def _proactive_priority(self, cid: str) -> Tuple[float, int, float, str]:
        c = self.containers[cid]
        lead = max(0.0, c.eta - self.now)
        blockers = self.yard.blockers_above(cid, self.containers)
        advance = max(0.0, -c.eta_delta)
        return (lead, -blockers, -advance, cid)

    # ------------------------------------------------------------------
    # Mandatory / proactive YC decisions
    # ------------------------------------------------------------------
    def _valid_queue_tasks(self, block: int, kind: str) -> List[Task]:
        out: List[Task] = []
        for task in self.ycs[block].queue:
            if task.kind != kind:
                continue
            c = self.containers[task.container_id]
            if kind == "RETRIEVAL_STEP" and not c.stored:
                continue
            if kind == "STORAGE" and c.storage_complete is not None:
                continue
            out.append(task)
        return out

    def _mandatory_kind(self, block: int) -> Optional[str]:
        retrieval = self._valid_queue_tasks(block, "RETRIEVAL_STEP")
        storage = self._valid_queue_tasks(block, "STORAGE")
        if retrieval and not storage:
            return "RETRIEVAL_STEP"
        if storage and not retrieval:
            return "STORAGE"
        if not retrieval and not storage:
            return None
        oldest_r = max(
            max(0.0, self.now - self.containers[t.container_id].truck_actual)
            for t in retrieval
        )
        oldest_s = max(
            max(0.0, self.now - self.containers[t.container_id].yard_arrival)
            for t in storage
        )
        return "RETRIEVAL_STEP" if oldest_r >= oldest_s else "STORAGE"

    def target_capacity_slack(self, cid: str) -> float:
        c = self.containers[cid]
        if c.block is None:
            return 0.0
        block = c.block
        lead = max(0.0, c.eta - self.now)
        mandatory_n = (
            len(self._valid_queue_tasks(block, "RETRIEVAL_STEP"))
            + len(self._valid_queue_tasks(block, "STORAGE"))
        )
        return lead - mandatory_n * self.move_time

    def block_capacity_slack(self, block: int) -> float:
        proactive = self.proactive_candidates(block)
        if not proactive:
            return 0.0
        return min(self.target_capacity_slack(cid) for cid in proactive)

    def _target_position(self, cid: str) -> int:
        c = self.containers[cid]
        if c.block is None or c.stack is None:
            raise RuntimeError(f"Container {cid} has no yard position")
        stack = self.yard.stacks[c.block][c.stack]
        try:
            tier = stack.index(cid)
        except ValueError as exc:
            raise RuntimeError(f"Container {cid} missing from yard stack") from exc
        return c.stack * self.yard.max_tier + tier

    def _cid_at_target_position(self, block: int, target_pos: int) -> Optional[str]:
        source_stack = target_pos // self.yard.max_tier
        tier = target_pos % self.yard.max_tier
        if source_stack < 0 or source_stack >= self.yard.stacks_per_block:
            return None
        stack = self.yard.stacks[block][source_stack]
        if tier >= len(stack):
            return None
        return stack[tier]

    def proactive_pair_actions(self, block: int) -> List[int]:
        """All physically feasible (target, destination) proactive moves.

        Capacity slack and queue pressure are deliberately *not* masks in V4.
        """
        actions: List[int] = []
        for cid in self.proactive_candidates(block):
            c = self.containers[cid]
            assert c.stack is not None
            target_pos = self._target_position(cid)
            for dest in self.yard.feasible_relocation_stacks(block, c.stack):
                actions.append(encode_proactive_action(target_pos, dest))
        return actions

    def rule_resolved_proactive_action(self, block: int) -> Optional[int]:
        """Resolve one transparent proactive move for operation-only control.

        Target priority is the existing ETA/blocker/ETA-advance ordering used by
        proactive_candidates().  The moved object is the current top blocker of
        that target.  Destination is selected by the existing information-aware
        relocation heuristic.  This helper changes action support only; it does
        not alter task duration, reward, or simulator transitions.
        """
        candidates = self.proactive_candidates(block)
        if not candidates:
            return None
        cid = candidates[0]
        c = self.containers[cid]
        if c.stack is None:
            return None
        source_stack = c.stack
        moving_id = self.yard.top(block, source_stack)
        if moving_id is None or moving_id == cid:
            return None
        dest_stack = self.policy.choose_relocation_destination(
            self, moving_id, block, source_stack
        )
        return encode_proactive_action(self._target_position(cid), dest_stack)

    def yc_action_mask(self, block: int) -> np.ndarray:
        mask = np.zeros(YC_ACTION_DIM, dtype=np.bool_)
        if self.ycs[block].busy:
            return mask

        mandatory = self._mandatory_kind(block)
        if mandatory is not None:
            mask[YC_MANDATORY] = True

        if self.rule_resolve_proactive_pair:
            resolved = self.rule_resolved_proactive_action(block)
            proactive_actions = [] if resolved is None else [resolved]
        else:
            proactive_actions = self.proactive_pair_actions(block)
        for a in proactive_actions:
            mask[a] = True

        # Idle is exposed only when speculative work exists but no compulsory
        # work is waiting. This prevents meaningless repeated idle decisions
        # during mandatory drain while preserving the option to skip proactive.
        mask[YC_IDLE] = bool(proactive_actions) and mandatory is None
        return mask

    def request_yc_decision(self, block: int) -> None:
        if block < 0 or block >= len(self.ycs) or self.ycs[block].busy:
            return
        if not bool(self.yc_action_mask(block).any()):
            return
        if block not in self._yc_decision_set:
            self._yc_decision_set.add(block)
            self._yc_decision_blocks.append(block)

    def _pop_next_yc_decision(self) -> Optional[ResourceDecision]:
        while self._yc_decision_blocks:
            block = self._yc_decision_blocks.popleft()
            self._yc_decision_set.discard(block)
            if self.ycs[block].busy:
                continue
            if bool(self.yc_action_mask(block).any()):
                return ResourceDecision("yc", block=block)
        return None

    def _pop_queue_task(self, block: int, kind: str) -> Task:
        yc = self.ycs[block]
        candidates = self._valid_queue_tasks(block, kind)
        if not candidates:
            raise RuntimeError(f"No {kind} task available in block {block}")
        if kind == "RETRIEVAL_STEP":
            chosen = min(
                candidates,
                key=lambda t: (
                    self.containers[t.container_id].truck_actual,
                    t.created_at,
                    t.container_id,
                ),
            )
        else:
            chosen = min(candidates, key=lambda t: (t.created_at, t.container_id))
        yc.queue.remove(chosen)
        return chosen

    def _start_selected_task(self, block: int, task: Task) -> None:
        yc = self.ycs[block]
        if yc.busy:
            raise RuntimeError("YC is already busy")
        yc.busy = True
        yc.current = task
        self.log("YC_TASK_START", task.container_id, block=block, task=task.kind)
        self.schedule(self.now + task.duration, "YC_TASK_COMPLETE", block=block)

    def apply_yc_action(self, action: int) -> None:
        d = self.pending_decision
        if d is None or d.kind != "yc" or d.block is None:
            raise RuntimeError("No YC decision is pending")
        block = d.block
        mask = self.yc_action_mask(block)
        if action < 0 or action >= YC_ACTION_DIM or not bool(mask[action]):
            raise ValueError(f"Invalid YC action {action} for block {block}")
        self.pending_decision = None
        self.log("YC_SCHEDULING_DECISION", block=block, action=int(action))

        if action == YC_IDLE:
            return

        if action == YC_MANDATORY:
            kind = self._mandatory_kind(block)
            if kind is None:
                raise RuntimeError("Mandatory action selected without mandatory work")
            task = self._pop_queue_task(block, kind)
            self._start_selected_task(block, task)
            return

        target_pos, dest_stack = decode_proactive_action(action)
        cid = self._cid_at_target_position(block, target_pos)
        if cid is None or not self._proactive_eligible(cid):
            raise RuntimeError("Selected proactive target is no longer eligible")
        c = self.containers[cid]
        if c.stack is None or dest_stack not in self.yard.feasible_relocation_stacks(block, c.stack):
            raise RuntimeError("Selected proactive destination is infeasible")

        self.yard.reserve(block, dest_stack)
        c.premarshal_pending = True
        self.log(
            "PREMARSHALL_REQUEST",
            cid,
            block=block,
            target_pos=target_pos,
            dest_stack=dest_stack,
        )
        self._start_selected_task(
            block,
            Task(
                "PREMARSHALL_STEP",
                cid,
                self.move_time,
                target_stack=dest_stack,
                created_at=self.now,
            ),
        )

    # ------------------------------------------------------------------
    # Truck arrivals / retrieval
    # ------------------------------------------------------------------
    def _on_truck_arrival(self, cid: str) -> None:
        c = self.containers[cid]
        if not c.retrievable_this_episode:
            raise RuntimeError("Same-episode retrieval assigned to ineligible container")
        c.truck_arrived = True
        self.log("TRUCK_ARRIVAL", cid)
        if c.stored and not c.retrieval_requested:
            self._request_retrieval(cid)

    def _request_retrieval(self, cid: str) -> None:
        c = self.containers[cid]
        if not c.stored or c.block is None:
            return
        c.retrieval_requested = True
        c.premarshal_pending = False
        self.log("RETRIEVAL_REQUEST", cid, block=c.block)
        self.submit_task(
            c.block,
            Task("RETRIEVAL_STEP", cid, self.move_time, created_at=self.now),
        )

    # ------------------------------------------------------------------
    # Event progression and mandatory drain
    # ------------------------------------------------------------------
    def advance_until_decision_or_done(self) -> Optional[ResourceDecision]:
        if self.pending_decision is not None:
            return self.pending_decision

        d = self._pop_next_yc_decision()
        if d is not None:
            self.pending_decision = d
            return d

        while self.events:
            ev = heapq.heappop(self.events)
            self._advance_clock(ev.time)

            if ev.kind == "CONTAINER_ARRIVAL":
                self._on_container_arrival(ev.payload["cid"])
            elif ev.kind == "TRUCK_ETA_UPDATE":
                self._on_eta_update(ev.payload["cid"], ev.payload["eta"])
            elif ev.kind == "TRUCK_ARRIVAL":
                self._on_truck_arrival(ev.payload["cid"])
            elif ev.kind == "STORAGE_RETRY":
                self._on_storage_retry()
            elif ev.kind == "YC_TASK_COMPLETE":
                self._finish_task(ev.payload["block"])
            elif ev.kind == "OPERATING_WINDOW_END":
                self.operating_closed = True
                self.log("OPERATING_WINDOW_END")
                for b in range(self.yard.n_blocks):
                    self.request_yc_decision(b)
            else:
                raise ValueError(f"Unknown event: {ev.kind}")

            if self.pending_decision is not None:
                return self.pending_decision
            d = self._pop_next_yc_decision()
            if d is not None:
                self.pending_decision = d
                return d

        # Drain any mandatory work that remains after exogenous events stop.
        self._schedule_storage_retry()
        if self.pending_decision is not None:
            return self.pending_decision
        for b in range(self.yard.n_blocks):
            self.request_yc_decision(b)
        d = self._pop_next_yc_decision()
        if d is not None:
            self.pending_decision = d
            return d

        if self.pending_storage_queue:
            if self.yard.feasible_storage_slots():
                self._on_storage_retry()
                return self.pending_decision
            raise RuntimeError(
                "Mandatory drain deadlock: storage jobs remain but no future event can free capacity"
            )
        return None

    # ------------------------------------------------------------------
    # Diagnostics and KPIs
    # ------------------------------------------------------------------
    def structural_risk_potential(self) -> float:
        virtual = [[list(stack) for stack in block] for block in self.yard.stacks]
        for b, yc in enumerate(self.ycs):
            tasks: List[Task] = []
            if yc.current is not None and yc.current.kind == "STORAGE":
                tasks.append(yc.current)
            tasks.extend(t for t in yc.queue if t.kind == "STORAGE")
            for task in tasks:
                if task.target_stack is not None:
                    virtual[b][task.target_stack].append(task.container_id)
        inversions = 0
        for block in virtual:
            for stack in block:
                for upper_pos in range(1, len(stack)):
                    upper = self.containers[stack[upper_pos]]
                    for lower_pos in range(upper_pos):
                        lower = self.containers[stack[lower_pos]]
                        if lower.eta < upper.eta:
                            inversions += 1
        return -float(inversions) / max(len(self.containers), 1)

    def yc_queue_pressure_potential(self, block: int) -> float:
        n_t = max(self.n_retrieval_requests, 1)
        n_s = max(self.n_storage_requests, 1)
        retrieval_pressure = sum(
            1.0 + min(max(0.0, self.now - self.containers[t.container_id].truck_actual) / 60.0, 4.0)
            for t in self._valid_queue_tasks(block, "RETRIEVAL_STEP")
        ) / n_t
        storage_pressure = sum(
            1.0 + min(max(0.0, self.now - self.containers[t.container_id].yard_arrival) / 60.0, 4.0)
            for t in self._valid_queue_tasks(block, "STORAGE")
        ) / n_s
        proactive_pressure = 0.25 * len(self.proactive_candidates(block)) / n_t
        return -(retrieval_pressure + storage_pressure + proactive_pressure)

    def cost_snapshot(self) -> Tuple[float, float, int, float]:
        return (
            self.truck_wait_area,
            self.storage_wait_area,
            self.rehandling_moves + self.proactive_moves,
            self.structural_risk_potential(),
        )

    def kpis(self) -> Dict[str, float]:
        storage_waits = [
            c.storage_complete - c.yard_arrival
            for c in self.containers.values()
            if c.cohort == "new_inbound" and c.storage_complete is not None
        ]
        truck_waits = [
            c.retrieval_complete - c.truck_actual
            for c in self.containers.values()
            if c.retrievable_this_episode and c.retrieval_complete is not None
        ]
        total_moves = sum(yc.completed_moves for yc in self.ycs)
        mean_yc_utilization = (
            float(np.mean([yc.busy_time / self.now for yc in self.ycs])) if self.now > 0 else 0.0
        )
        mean_storage = float(np.mean(storage_waits)) if storage_waits else math.nan
        mean_truck = float(np.mean(truck_waits)) if truck_waits else math.nan
        n_t = max(self.n_retrieval_requests, 1)
        extra_minutes = (self.rehandling_moves + self.proactive_moves) * self.move_time
        return {
            "mean_storage_completion_delay": mean_storage,
            "mean_truck_completion_delay": mean_truck,
            "mean_storage_wait": mean_storage,
            "mean_truck_wait": mean_truck,
            "rehandling_moves": float(self.rehandling_moves),
            "proactive_moves": float(self.proactive_moves),
            "extra_yc_minutes": float(extra_minutes),
            "extra_yc_minutes_per_retrieval": float(extra_minutes / n_t),
            "total_yc_moves": float(total_moves),
            "mean_yc_utilization": mean_yc_utilization,
            "max_yc_queue": float(max((yc.max_queue_len for yc in self.ycs), default=0)),
            "yard_occupancy_end": self.yard.occupancy(),
            "retrieved_containers": float(sum(c.retrieval_complete is not None for c in self.containers.values() if c.retrievable_this_episode)),
            "retrieval_requests": float(self.n_retrieval_requests),
            "storage_requests": float(self.n_storage_requests),
            "unretrieved_containers": float(sum(c.retrievable_this_episode and c.retrieval_complete is None for c in self.containers.values())),
            "unstored_inbound_containers": float(sum(c.cohort == "new_inbound" and c.storage_complete is None for c in self.containers.values())),
            "episode_end_time": float(self.now),
            "drain_duration": float(max(0.0, self.now - self.operating_horizon)),
            "arrival_rate_per_hour": float(self.arrival_rate_per_hour),
        }


# ---------------------------------------------------------------------------
# Scenario generation
# ---------------------------------------------------------------------------

def _poisson_times(rng: random.Random, rate_per_hour: float, horizon_min: float) -> List[float]:
    if rate_per_hour <= 0:
        return []
    rate_per_min = rate_per_hour / 60.0
    t = 0.0
    out: List[float] = []
    while True:
        t += rng.expovariate(rate_per_min)
        if t >= horizon_min:
            break
        out.append(t)
    return out




def _sample_future_pickup_delay_min(rng: random.Random) -> float:
    """Synthetic future pickup delay for containers not retrieved this episode.

    The distribution is deliberately right-skewed and on a multi-day timescale:
    lognormal with 72 h median and log-SD 0.5, clipped to 24--168 h.
    It is a synthetic priority proxy, not a fitted terminal-specific dwell-time model.
    """
    hours = rng.lognormvariate(math.log(FUTURE_PICKUP_MEDIAN_HOURS), FUTURE_PICKUP_LOG_SIGMA)
    hours = min(FUTURE_PICKUP_MAX_HOURS, max(FUTURE_PICKUP_MIN_HOURS, hours))
    return hours * 60.0

def _eta_update_times(actual: float) -> List[float]:
    times: List[float] = []
    lead = PROACTIVE_HORIZON_MIN
    while lead >= ETA_UPDATE_INTERVAL_MIN:
        t = actual - lead
        if 0.0 < t < actual:
            times.append(t)
        lead -= ETA_UPDATE_INTERVAL_MIN
    return sorted(set(times))


def build_resource_marl_scenario(
    seed: int,
    arrival_rate_per_hour: float,
    n_blocks: int = N_BLOCKS,
    stacks_per_block: int = STACKS_PER_BLOCK,
    max_tier: int = MAX_TIER,
    initial_containers: int = INITIAL_CONTAINERS,
    operating_horizon: float = OPERATING_HORIZON_MIN,
    enable_proactive: bool = True,
    move_time: float = YC_MOVE_TIME_MIN,
    rule_resolve_proactive_pair: bool = False,
) -> ResourceMarlSimulator:
    if n_blocks != N_BLOCKS or stacks_per_block != STACKS_PER_BLOCK or max_tier != MAX_TIER:
        raise ValueError("V4 final environment uses fixed 4 blocks x 25 stacks x 4 tiers")
    if initial_containers != INITIAL_CONTAINERS:
        raise ValueError("V4 final environment uses exactly 268 initial containers")

    sim = ResourceMarlSimulator(
        n_blocks=n_blocks,
        stacks_per_block=stacks_per_block,
        max_tier=max_tier,
        seed=seed,
        enable_proactive=enable_proactive,
        move_time=move_time,
        operating_horizon=operating_horizon,
        rule_resolve_proactive_pair=rule_resolve_proactive_pair,
    )
    sim.arrival_rate_per_hour = float(arrival_rate_per_hour)
    rng = random.Random(seed)

    # Exactly 67 initial containers per block; retrieval order is generated
    # independently afterward so the initial layout is not pre-sorted.
    initial_ids: List[str] = []
    per_block = initial_containers // n_blocks
    if per_block * n_blocks != initial_containers:
        raise ValueError("Initial containers must divide evenly across blocks")
    for b in range(n_blocks):
        for j in range(per_block):
            feasible = [s for s in range(stacks_per_block) if len(sim.yard.stacks[b][s]) < max_tier]
            if not feasible:
                raise RuntimeError("Unable to construct initial feasible layout")
            s = rng.choice(feasible)
            cid = f"I{b}_{j:03d}"
            # Non-selected initial containers receive a future pickup proxy so
            # they remain meaningful blockers but cannot be retrieved this episode.
            future_actual = _sample_future_pickup_delay_min(rng)
            initial_eta = max(0.0, future_actual + rng.gauss(0.0, 60.0))
            sim.add_initial_container(cid, b, s, future_actual, initial_eta)
            c = sim.containers[cid]
            c.cohort = "initial"
            c.retrievable_this_episode = False
            initial_ids.append(cid)

    storage_times = _poisson_times(rng, arrival_rate_per_hour, operating_horizon)
    retrieval_times = _poisson_times(rng, arrival_rate_per_hour, operating_horizon)
    retrieval_times = retrieval_times[: len(initial_ids)]
    sim.n_storage_requests = len(storage_times)
    sim.n_retrieval_requests = len(retrieval_times)

    # Retrieval target mapping is generated once at initialization and remains
    # identical across policies for a fixed scenario seed.
    target_ids = rng.sample(initial_ids, k=len(retrieval_times))
    for cid, actual in zip(target_ids, retrieval_times):
        c = sim.containers[cid]
        c.retrievable_this_episode = True
        c.truck_actual = float(actual)
        c.eta = max(0.0, actual + rng.gauss(0.0, 60.0))
        c.prev_eta = c.eta
        c.eta_delta = 0.0
        sim.schedule(actual, "TRUCK_ARRIVAL", cid=cid)
        for t in _eta_update_times(actual):
            lead = max(actual - t, 0.0)
            sigma = min(30.0, max(5.0, 0.2 * lead))
            eta = max(t, actual + rng.gauss(0.0, sigma))
            sim.schedule(t, "TRUCK_ETA_UPDATE", cid=cid, eta=eta)

    # New inbound containers have future pickup proxies but no same-episode
    # truck event by construction.
    for i, yard_arrival in enumerate(storage_times):
        cid = f"N{i:04d}"
        future_actual = yard_arrival + _sample_future_pickup_delay_min(rng)
        eta = max(yard_arrival, future_actual + rng.gauss(0.0, 60.0))
        c = Container(
            cid=cid,
            yard_arrival=float(yard_arrival),
            truck_actual=float(future_actual),
            eta=float(eta),
            prev_eta=float(eta),
            cohort="new_inbound",
            retrievable_this_episode=False,
        )
        sim.containers[cid] = c
        sim.schedule(yard_arrival, "CONTAINER_ARRIVAL", cid=cid)

    sim.schedule(operating_horizon, "OPERATING_WINDOW_END")
    return sim


# ---------------------------------------------------------------------------
# Gym-like asynchronous environment wrapper
# ---------------------------------------------------------------------------

class ResourceMARLYardEnv:
    GLOBAL_CONTEXT_DIM = 8
    BLOCK_FEAT_DIM = 8
    SLOT_FEAT_DIM = 4
    STORAGE_CONTEXT_DIM = 4
    YC_CONTEXT_DIM = 18
    YC_PAIR_FEAT_DIM = 9
    YC_PAIR_COUNT = YC_PAIR_COUNT
    YC_OBS_DIM = YC_CONTEXT_DIM + YC_PAIR_COUNT * YC_PAIR_FEAT_DIM

    def __init__(
        self,
        seed: int = 1,
        arrival_rate_per_hour: float = 20.0,
        include_resource_state: bool = True,
        truck_wait_weight: float = 1.0,
        storage_wait_weight: float = 1.0,
        extra_move_weight: float = 0.10,
        risk_shaping_weight: float = 0.0,
        yc_queue_shaping_weight: float = 0.0,
        enable_proactive: bool = True,
        yc_move_time: float = YC_MOVE_TIME_MIN,
        rule_resolve_proactive_pair: bool = False,
    ):
        self.seed = int(seed)
        self.arrival_rate_per_hour = float(arrival_rate_per_hour)
        self.include_resource_state = bool(include_resource_state)
        self.truck_wait_weight = float(truck_wait_weight)
        self.storage_wait_weight = float(storage_wait_weight)
        self.extra_move_weight = float(extra_move_weight)
        self.risk_shaping_weight = float(risk_shaping_weight)
        self.yc_queue_shaping_weight = float(yc_queue_shaping_weight)
        self.enable_proactive = bool(enable_proactive)
        self.yc_move_time = float(yc_move_time)
        self.rule_resolve_proactive_pair = bool(rule_resolve_proactive_pair)

        self.n_blocks = N_BLOCKS
        self.stacks_per_block = STACKS_PER_BLOCK
        self.max_tier = MAX_TIER
        self.n_slots = TOTAL_STACKS
        self.storage_action_dim = self.n_slots
        self.yc_action_dim = YC_ACTION_DIM

        self.global_obs_dim = (
            self.GLOBAL_CONTEXT_DIM
            + self.n_blocks * self.BLOCK_FEAT_DIM
            + self.n_slots * self.SLOT_FEAT_DIM
        )
        self.storage_obs_dim = (
            self.STORAGE_CONTEXT_DIM
            + self.n_blocks * self.BLOCK_FEAT_DIM
            + self.n_slots * self.SLOT_FEAT_DIM
        )
        self.yc_obs_dim = self.YC_OBS_DIM
        self.central_yc_obs_dim = self.global_obs_dim + self.yc_obs_dim

        self.sim: Optional[ResourceMarlSimulator] = None
        self.last_cost_snapshot = (0.0, 0.0, 0, 0.0)
        self._episode_return = 0.0
        self._decision_count = 0
        self._last_reward_components = {
            "team": 0.0,
            "truck": 0.0,
            "storage": 0.0,
            "extra_yc": 0.0,
            "storage_local": 0.0,
            "yc_local": 0.0,
            "total": 0.0,
        }

    def reset(self, seed: Optional[int] = None, arrival_rate_per_hour: Optional[float] = None):
        if seed is not None:
            self.seed = int(seed)
        if arrival_rate_per_hour is not None:
            self.arrival_rate_per_hour = float(arrival_rate_per_hour)
        self.sim = build_resource_marl_scenario(
            seed=self.seed,
            arrival_rate_per_hour=self.arrival_rate_per_hour,
            enable_proactive=self.enable_proactive,
            move_time=self.yc_move_time,
            rule_resolve_proactive_pair=self.rule_resolve_proactive_pair,
        )
        self._episode_return = 0.0
        self._decision_count = 0
        self._last_reward_components = {k: 0.0 for k in self._last_reward_components}
        self.sim.advance_until_decision_or_done()
        self.last_cost_snapshot = self.sim.cost_snapshot()
        obs = self.actor_observation()
        return obs, self._info(done=self.sim.pending_decision is None and not self.sim.events)

    def active_agent(self) -> str:
        if self.sim is None or self.sim.pending_decision is None:
            return "done"
        d = self.sim.pending_decision
        if d.kind == "storage":
            return STORAGE_AGENT
        if d.kind == "yc" and d.block is not None:
            return f"{YC_AGENT_PREFIX}{d.block}"
        raise RuntimeError(f"Unknown decision {d}")

    def active_block(self) -> Optional[int]:
        if self.sim is None or self.sim.pending_decision is None:
            return None
        return self.sim.pending_decision.block

    def action_mask(self) -> np.ndarray:
        if self.sim is None or self.sim.pending_decision is None:
            return np.zeros(1, dtype=np.bool_)
        d = self.sim.pending_decision
        if d.kind == "storage":
            return self.sim.storage_action_mask()
        if d.kind == "yc" and d.block is not None:
            return self.sim.yc_action_mask(d.block)
        return np.zeros(1, dtype=np.bool_)

    def step(self, action: int):
        if self.sim is None or self.sim.pending_decision is None:
            raise RuntimeError("Environment must be reset or episode is complete")
        before = self.sim.cost_snapshot()
        d = self.sim.pending_decision
        active_block = d.block

        if d.kind == "storage":
            phi_before = self.sim.structural_risk_potential()
            self.sim.apply_storage_action(int(action))
            phi_after = self.sim.structural_risk_potential()
            storage_local = self.risk_shaping_weight * (phi_after - phi_before)
            yc_local = 0.0
        elif d.kind == "yc":
            if active_block is None:
                raise RuntimeError("YC decision missing block")
            phi_before = self.sim.yc_queue_pressure_potential(active_block)
            self.sim.apply_yc_action(int(action))
            phi_after = self.sim.yc_queue_pressure_potential(active_block)
            yc_local = self.yc_queue_shaping_weight * (phi_after - phi_before)
            storage_local = 0.0
        else:
            raise RuntimeError(f"Unknown decision kind {d.kind}")

        self.sim.advance_until_decision_or_done()
        after = self.sim.cost_snapshot()
        n_t = max(self.sim.n_retrieval_requests, 1)
        n_s = max(self.sim.n_storage_requests, 1)

        delta_truck = (after[0] - before[0]) / n_t
        delta_storage = (after[1] - before[1]) / n_s
        delta_extra_yc = (after[2] - before[2]) * self.sim.move_time / n_t
        truck_term = -self.truck_wait_weight * delta_truck
        storage_term = -self.storage_wait_weight * delta_storage
        extra_term = -self.extra_move_weight * delta_extra_yc
        team_reward = truck_term + storage_term + extra_term
        reward = team_reward + storage_local + yc_local

        self._last_reward_components = {
            "team": float(team_reward),
            "truck": float(truck_term),
            "storage": float(storage_term),
            "extra_yc": float(extra_term),
            "storage_local": float(storage_local),
            "yc_local": float(yc_local),
            "total": float(reward),
        }
        self.last_cost_snapshot = after
        self._episode_return += reward
        self._decision_count += 1

        done = self.sim.pending_decision is None and not self.sim.events
        obs = self.actor_observation() if not done else np.zeros(1, dtype=np.float32)
        return obs, float(reward), done, False, self._info(done)

    def _info(self, done: bool) -> dict:
        info = {
            "active_agent": self.active_agent() if not done else "done",
            "active_block": self.active_block() if not done else None,
            "action_mask": self.action_mask() if not done else np.zeros(1, dtype=np.bool_),
            "decision_count": self._decision_count,
            "episode_return": self._episode_return,
            "arrival_rate_per_hour": self.arrival_rate_per_hour,
            "reward_components": dict(self._last_reward_components),
        }
        if done and self.sim is not None:
            info["kpis"] = self.sim.kpis()
        return info

    # ------------------------------------------------------------------
    # Observation helpers
    # ------------------------------------------------------------------
    def future_inbound_pressure(self, window: float = 30.0) -> float:
        """Observable recent-arrival / pending-work pressure, not future oracle."""
        if self.sim is None or window <= 0:
            return 0.0
        recent = sum(
            1
            for row in self.sim.event_log
            if row.get("event_type") == "CONTAINER_ARRIVAL"
            and self.sim.now - window <= float(row.get("timestamp", -1e9)) <= self.sim.now
        )
        expected = max(self.arrival_rate_per_hour * window / 60.0, 1.0)
        pending = len(self.sim.pending_storage_queue)
        return float(np.clip(recent / expected + pending / 10.0, 0.0, 3.0))

    def _block_features(self, b: int, resource_state: bool) -> List[float]:
        sim = self.sim
        assert sim is not None
        used = sum(len(st) for st in sim.yard.stacks[b])
        capacity = sim.yard.stacks_per_block * sim.yard.max_tier
        occ = used / capacity
        retrieval_tasks = sim._valid_queue_tasks(b, "RETRIEVAL_STEP")
        storage_tasks = sim._valid_queue_tasks(b, "STORAGE")
        proactive = sim.proactive_candidates(b)
        rq = min(len(retrieval_tasks) / 10.0, 3.0) if resource_state else 0.0
        sq = min(len(storage_tasks) / 10.0, 3.0) if resource_state else 0.0
        pq = min(len(proactive) / 10.0, 3.0) if resource_state else 0.0
        busy = (1.0 if sim.ycs[b].busy else 0.0) if resource_state else 0.0
        stored_ids = [cid for st in sim.yard.stacks[b] for cid in st]
        imminent = sum(
            1
            for cid in stored_ids
            if sim.containers[cid].retrievable_this_episode
            and 0 <= sim.containers[cid].eta - sim.now <= PROACTIVE_HORIZON_MIN
        )
        blockers = sum(sim.yard.blockers_above(cid, sim.containers) for cid in stored_ids)
        imminent_ratio = imminent / max(len(stored_ids), 1)
        mean_blockers = blockers / max(len(stored_ids) * sim.yard.max_tier, 1)
        return [occ, rq, sq, pq, busy, imminent_ratio, mean_blockers, 0.0]

    def global_observation(self, resource_state: bool = True) -> np.ndarray:
        if self.sim is None or self.sim.pending_decision is None:
            return np.zeros(self.global_obs_dim, dtype=np.float32)
        sim = self.sim
        d = sim.pending_decision
        event_storage = 1.0 if d.kind == "storage" else 0.0
        event_yc = 1.0 - event_storage
        time_norm = min(sim.now / OPERATING_HORIZON_MIN, 2.0)
        yard_occ = sim.yard.occupancy()
        inbound = self.future_inbound_pressure() if resource_state else 0.0
        target_lead = 0.0
        target_eta_delta = 0.0
        if d.kind == "storage" and d.cid is not None:
            c = sim.containers[d.cid]
            target_lead = float(np.clip((c.eta - sim.now) / 1440.0, 0.0, 2.0))
            target_eta_delta = float(np.clip(c.eta_delta / 60.0, -2.0, 2.0))
        mandatory_total = sum(
            len(sim._valid_queue_tasks(b, "RETRIEVAL_STEP"))
            + len(sim._valid_queue_tasks(b, "STORAGE"))
            for b in range(self.n_blocks)
        )
        values: List[float] = [
            event_storage,
            event_yc,
            time_norm,
            yard_occ,
            inbound,
            target_lead,
            target_eta_delta,
            min(mandatory_total / 20.0, 3.0) if resource_state else 0.0,
        ]
        for b in range(self.n_blocks):
            feats = self._block_features(b, resource_state)
            feats[-1] = 1.0 if d.kind == "yc" and d.block == b else 0.0
            values.extend(feats)

        storage_mask = sim.storage_action_mask() if d.kind == "storage" else np.zeros(self.n_slots, dtype=np.bool_)
        target_eta = sim.containers[d.cid].eta if d.kind == "storage" and d.cid is not None else sim.now + 720.0
        for idx in range(self.n_slots):
            b = idx // self.stacks_per_block
            s = idx % self.stacks_per_block
            stack = sim.yard.stacks[b][s]
            reserved = sim.yard.reserved[b][s]
            height = (len(stack) + reserved) / sim.yard.max_tier
            inversion = sum(1 for lower_id in stack if sim.containers[lower_id].eta < target_eta) / sim.yard.max_tier
            queue_work = (
                len(sim._valid_queue_tasks(b, "RETRIEVAL_STEP"))
                + len(sim._valid_queue_tasks(b, "STORAGE"))
            ) / 10.0 if resource_state else 0.0
            feasible = float(storage_mask[idx]) if d.kind == "storage" else float(len(stack) + reserved < sim.yard.max_tier)
            values.extend([height, inversion, min(queue_work, 3.0), feasible])
        arr = np.asarray(values, dtype=np.float32)
        if arr.shape != (self.global_obs_dim,):
            raise RuntimeError(f"Global observation shape mismatch {arr.shape} != {(self.global_obs_dim,)}")
        return arr

    def storage_observation(self) -> np.ndarray:
        full = self.global_observation(resource_state=self.include_resource_state)
        if self.sim is None or self.sim.pending_decision is None:
            return np.zeros(self.storage_obs_dim, dtype=np.float32)
        local = np.concatenate([full[[2, 3, 4, 5]], full[self.GLOBAL_CONTEXT_DIM:]]).astype(np.float32, copy=False)
        if local.shape != (self.storage_obs_dim,):
            raise RuntimeError(f"Storage observation shape mismatch {local.shape} != {(self.storage_obs_dim,)}")
        return local

    def _yc_context_observation(self, block: int, resource_state: Optional[bool] = None) -> np.ndarray:
        sim = self.sim
        if sim is None:
            return np.zeros(self.YC_CONTEXT_DIM, dtype=np.float32)
        resource = self.include_resource_state if resource_state is None else bool(resource_state)
        local = self._block_features(block, resource)[:7]
        retrieval = sim._valid_queue_tasks(block, "RETRIEVAL_STEP")
        storage = sim._valid_queue_tasks(block, "STORAGE")
        proactive = sim.proactive_candidates(block)
        oldest_retrieval = max((sim.now - t.created_at for t in retrieval), default=0.0)
        oldest_storage = max((sim.now - t.created_at for t in storage), default=0.0)
        urgent_lead = min((max(0.0, sim.containers[cid].eta - sim.now) for cid in proactive), default=PROACTIVE_HORIZON_MIN)
        max_blockers = max((sim.yard.blockers_above(cid, sim.containers) for cid in proactive), default=0)
        capacity_slack = sim.block_capacity_slack(block) if proactive else 0.0
        mandatory_counts = [
            len(sim._valid_queue_tasks(b, "RETRIEVAL_STEP"))
            + len(sim._valid_queue_tasks(b, "STORAGE"))
            for b in range(self.n_blocks)
        ]
        proactive_counts = [len(sim.proactive_candidates(b)) for b in range(self.n_blocks)]
        obs = np.asarray(
            [
                min(sim.now / OPERATING_HORIZON_MIN, 2.0),
                sim.yard.occupancy(),
                self.future_inbound_pressure() if resource else 0.0,
                *local,
                min(oldest_retrieval / 120.0, 3.0) if resource else 0.0,
                min(oldest_storage / 120.0, 3.0) if resource else 0.0,
                min(urgent_lead / PROACTIVE_HORIZON_MIN, 2.0),
                float(np.clip(capacity_slack / PROACTIVE_HORIZON_MIN, -2.0, 2.0)) if resource else 0.0,
                min(max_blockers / sim.yard.max_tier, 1.0),
                min(float(np.mean(mandatory_counts)) / 10.0, 3.0) if resource else 0.0,
                min(float(np.max(mandatory_counts)) / 10.0, 3.0) if resource else 0.0,
                min(float(np.mean(proactive_counts)) / 10.0, 3.0) if resource else 0.0,
            ],
            dtype=np.float32,
        )
        if obs.shape != (self.YC_CONTEXT_DIM,):
            raise RuntimeError(f"YC context shape mismatch {obs.shape}")
        return obs

    def _proactive_pair_features(self, block: int, resource_state: Optional[bool] = None) -> np.ndarray:
        sim = self.sim
        feats = np.zeros((YC_PAIR_COUNT, self.YC_PAIR_FEAT_DIM), dtype=np.float32)
        if sim is None:
            return feats
        resource = self.include_resource_state if resource_state is None else bool(resource_state)
        eligible = set(sim.proactive_candidates(block))
        feasible_by_source: Dict[int, set[int]] = {}
        for target_pos in range(YC_TARGET_POSITIONS):
            cid = sim._cid_at_target_position(block, target_pos)
            if cid is None or cid not in eligible:
                continue
            c = sim.containers[cid]
            if c.stack is None:
                continue
            source_stack = c.stack
            blockers = sim.yard.blockers_above(cid, sim.containers)
            moving_id = sim.yard.top(block, source_stack)
            if moving_id is None or moving_id == cid:
                continue
            moving = sim.containers[moving_id]
            lead = max(0.0, c.eta - sim.now)
            advance = max(0.0, -c.eta_delta)
            moving_lead = moving.eta - sim.now
            slack = sim.target_capacity_slack(cid)
            feasible_set = feasible_by_source.setdefault(
                source_stack, set(sim.yard.feasible_relocation_stacks(block, source_stack))
            )
            for dest in range(YC_DEST_STACKS):
                idx = target_pos * YC_DEST_STACKS + dest
                dest_stack = sim.yard.stacks[block][dest]
                inversion = sum(1 for lower_id in dest_stack if sim.containers[lower_id].eta < moving.eta)
                feasible = dest in feasible_set
                nearest_eta = min(
                    (max(0.0, sim.containers[x].eta - sim.now) for x in dest_stack),
                    default=1440.0,
                )
                feats[idx] = np.asarray(
                    [
                        1.0 if feasible else 0.0,
                        min(lead / PROACTIVE_HORIZON_MIN, 2.0),
                        min(advance / 60.0, 2.0),
                        min(blockers / sim.yard.max_tier, 1.0),
                        float(np.clip(slack / PROACTIVE_HORIZON_MIN, -2.0, 2.0)) if resource else 0.0,
                        float(np.clip(moving_lead / 1440.0, -1.0, 2.0)),
                        min(len(dest_stack) / sim.yard.max_tier, 1.0),
                        min(inversion / sim.yard.max_tier, 1.0),
                        min(nearest_eta / 1440.0, 2.0),
                    ],
                    dtype=np.float32,
                )
        return feats

    def yc_observation(self, block: int, resource_state: Optional[bool] = None) -> np.ndarray:
        context = self._yc_context_observation(block, resource_state=resource_state)
        pair = self._proactive_pair_features(block, resource_state=resource_state).reshape(-1)
        obs = np.concatenate([context, pair]).astype(np.float32, copy=False)
        if obs.shape != (self.yc_obs_dim,):
            raise RuntimeError(f"YC observation shape mismatch {obs.shape} != {(self.yc_obs_dim,)}")
        return obs

    def actor_observation(self) -> np.ndarray:
        agent = self.active_agent()
        if agent == STORAGE_AGENT:
            return self.storage_observation()
        if agent.startswith(YC_AGENT_PREFIX):
            block = int(agent.split("_")[1])
            return self.yc_observation(block)
        return np.zeros(1, dtype=np.float32)

    def critic_observation(self) -> np.ndarray:
        return self.global_observation(resource_state=True)

    def centralized_actor_observation(self) -> np.ndarray:
        """Stronger-information actor input for the centralized PPO baseline."""
        agent = self.active_agent()
        if agent == STORAGE_AGENT:
            return self.global_observation(resource_state=True)
        if agent.startswith(YC_AGENT_PREFIX):
            block = int(agent.split("_")[1])
            return np.concatenate(
                [self.global_observation(resource_state=True), self.yc_observation(block, resource_state=True)]
            ).astype(np.float32, copy=False)
        return np.zeros(1, dtype=np.float32)


if __name__ == "__main__":
    env = ResourceMARLYardEnv(seed=11, arrival_rate_per_hour=20.0)
    _, info = env.reset()
    rng = np.random.default_rng(0)
    done = False
    for _ in range(100000):
        valid = np.flatnonzero(info["action_mask"])
        if len(valid) == 0:
            raise RuntimeError("No valid action")
        # Avoid random pathological idling when mandatory work exists.
        action = int(rng.choice(valid))
        _, _, done, _, info = env.step(action)
        if done:
            break
    print(info.get("kpis", {}))
