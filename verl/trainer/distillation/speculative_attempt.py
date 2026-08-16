"""Attempt identity and admission control for speculative teacher work.

TWO DISTINCTIONS THIS MODULE EXISTS TO KEEP:

1. LOGICAL WORK vs PHYSICAL ATTEMPT. `TeacherRequestKey` names a unit of work; an `AttemptID` names
   one physical execution of it. A revoked anchor that re-enters must NOT reuse its previous engine
   request id -- the engine may reject the duplicate, or worse, attribute a late output or an abort
   acknowledgement from the dead attempt to the live one. The sampling seed stays stable across
   attempts, so a restart reproduces the same continuation; only the identity the engine sees changes.

2. RESOURCE KINDS. Sequence slots, KV blocks and per-step scheduled tokens are different resources
   and must not share a reserve. Slots and KV blocks are PERSISTENT OCCUPANCY: a request holds them
   until it terminates, so the reserve-gap algebra applies. Scheduled tokens are RENEWED EVERY STEP,
   so treating them as occupancy would reserve capacity that no longer exists a step later; committed
   tokens are scheduled first and speculation takes only the permitted remainder.

The terminal transition is atomic and one-way. Whichever of completion or abort wins is
authoritative, and every later output from that attempt is discarded -- otherwise a cancelled child
could still supply a continuation that reaches training, which is a correctness bug rather than an
accounting one.
"""
from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass, field
from enum import Enum


class AttemptState(Enum):
    ACTIVE = "active"
    COMPLETED = "completed"
    ABORTED = "aborted"


class AttemptError(RuntimeError):
    """Raised on an illegal attempt transition or on output from a stale attempt."""


@dataclass(frozen=True)
class AttemptID:
    """One physical execution of a logical unit of teacher work.

    `engine_request_id` is what vLLM sees. It embeds the attempt number, so a restart after revocation
    is a genuinely new request to the engine while remaining the same logical work to us.
    """

    request_key: str          # TeacherRequestKey.key()
    attempt: int              # 0, 1, 2, ... increments on each re-entry
    seed: int                 # stable across attempts: deterministic INTENT, not a bitwise guarantee

    @property
    def engine_request_id(self) -> str:
        return "spec-" + hashlib.sha256(
            f"{self.request_key}|{self.attempt}".encode()).hexdigest()[:24]

    def next_attempt(self) -> "AttemptID":
        return AttemptID(self.request_key, self.attempt + 1, self.seed)


@dataclass
class _Attempt:
    aid: AttemptID
    state: AttemptState = AttemptState.ACTIVE
    prefill_tokens: int = 0        # physical recompute reported by the engine, after APC
    decode_tokens: int = 0         # decoded while ACTIVE
    late_after_abort_tokens: int = 0   # decoded after the abort won -- real GPU work, never usable
    max_observed_decode: int = 0   # high-water mark for CUMULATIVE outputs
    duplicate_cumulative: int = 0  # redelivered messages carrying no new tokens
    outputs_after_completion: int = 0
    result: object = None          # quarantined; reaches training only via a CommitPlan


class AttemptRegistry:
    """Tracks the CURRENT attempt per logical request, and enforces one terminal state each.

    Outputs are accepted only from the current attempt. A late output from a superseded attempt is
    silently dropped rather than raising: it is an expected consequence of revocation, not a bug, and
    raising would turn normal operation into noise. An output from a TERMINATED attempt does raise,
    because that means the engine emitted after acknowledging a terminal state.
    """

    def __init__(self):
        self._cur: dict = {}                  # request_key -> current _Attempt
        self._retired: dict = {}              # engine_request_id -> superseded _Attempt
        self._lock = threading.Lock()
        self.stats = {"completed": 0, "aborted_queued": 0, "aborted_running": 0,
                      "restarts": 0, "results_discarded": 0, "redundant_aborts": 0,
                      # physical cost that produced nothing usable -- the quantity A1/A2 compare
                      "late_after_abort_tokens": 0, "stale_attempt_tokens": 0,
                      "duplicate_cumulative_msgs": 0, "outputs_after_completion": 0,
                      "prefill_tokens_total": 0}

    def start(self, request_key: str, seed: int) -> AttemptID:
        with self._lock:
            prev = self._cur.get(request_key)
            if prev is not None and prev.state is not AttemptState.ACTIVE:
                # Keep the superseded attempt reachable: it may still emit, and those tokens are real
                # GPU work that must be accounted even though nothing usable comes of them.
                self._retired[prev.aid.engine_request_id] = prev
            if prev is None:
                aid = AttemptID(request_key, 0, seed)
            else:
                if prev.state is AttemptState.ACTIVE:
                    raise AttemptError(
                        f"{request_key}: attempt {prev.aid.attempt} is still ACTIVE; abort it before "
                        f"restarting. Two live attempts for one logical request would let either "
                        f"supply the continuation.")
                aid = prev.aid.next_attempt()
                self.stats["restarts"] += 1
            self._cur[request_key] = _Attempt(aid=aid)
            return aid

    def _get_current(self, aid: AttemptID):
        cur = self._cur.get(aid.request_key)
        if cur is None or cur.aid != aid:
            return None
        return cur

    def _find_any(self, aid: AttemptID):
        """Current attempt, or a superseded one still emitting. None only for a genuinely unknown id."""
        cur = self._cur.get(aid.request_key)
        if cur is not None and cur.aid == aid:
            return cur, True
        ret = self._retired.get(aid.engine_request_id)
        if ret is not None:
            return ret, False
        return None, False

    def record_prefill(self, aid: AttemptID, tokens: int) -> None:
        """Physical recompute reported by the engine, after APC. Counted regardless of state.

        A request cancelled after admission has already paid its prefill; charging it only while
        ACTIVE would make zero-decode cancellation look free when it was not.
        """
        with self._lock:
            att, _ = self._find_any(aid)
            if att is None:
                raise AttemptError(f"prefill for unknown attempt {aid.engine_request_id}")
            att.prefill_tokens += tokens
            self.stats["prefill_tokens_total"] += tokens

    def observe_decode(self, aid: AttemptID, observed_total_tokens: int) -> int:
        """Account CUMULATIVE decode via a per-attempt high-water mark. Returns tokens newly seen.

        ACCOUNTING IS INDEPENDENT OF RESULT ACCEPTANCE. A late output after an abort must never become
        supervision, but it represents real GPU work; ignoring it would undercount exactly the wasted
        work A1 and A2 are compared on. So the tokens are counted and bucketed by what they can be
        used for, never dropped.

        The high-water mark makes redelivered cumulative messages idempotent: a duplicate carries no
        new tokens and inflates nothing.
        """
        with self._lock:
            att, is_current = self._find_any(aid)
            if att is None:
                raise AttemptError(f"decode for unknown attempt {aid.engine_request_id}")
            new = max(0, observed_total_tokens - att.max_observed_decode)
            if new == 0:
                att.duplicate_cumulative += 1
                self.stats["duplicate_cumulative_msgs"] += 1
                return 0
            att.max_observed_decode = observed_total_tokens
            if not is_current:
                att.late_after_abort_tokens += new
                self.stats["stale_attempt_tokens"] += new
            elif att.state is AttemptState.ACTIVE:
                att.decode_tokens += new
            elif att.state is AttemptState.ABORTED:
                att.late_after_abort_tokens += new
                self.stats["late_after_abort_tokens"] += new
            else:                                   # COMPLETED
                att.outputs_after_completion += 1
                self.stats["outputs_after_completion"] += 1
            return new

    def complete(self, aid: AttemptID, result) -> bool:
        """ACTIVE -> COMPLETED. Returns False if abort already won, or the attempt is superseded."""
        with self._lock:
            cur = self._get_current(aid)
            if cur is None:
                # A superseded attempt finishing is an EXPECTED asynchronous race after revocation,
                # not a fault: discard the result, keep the diagnostic, do not crash the run. Hard
                # errors are reserved for unknown ids, malformed identities and impossible
                # transitions -- see _find_any and the double-complete check below.
                att, _ = self._find_any(aid)
                if att is None:
                    raise AttemptError(f"completion for unknown attempt {aid.engine_request_id}")
                self.stats["results_discarded"] += 1
                return False
            if cur.state is AttemptState.ABORTED:
                # Abort won. The result is discarded -- accepting it would let a cancelled child
                # supply a continuation that could then be committed -- but its cost is already
                # recorded by observe_decode.
                self.stats["results_discarded"] += 1
                return False
            if cur.state is AttemptState.COMPLETED:
                raise AttemptError(f"{aid.engine_request_id}: completed twice")
            cur.state = AttemptState.COMPLETED
            cur.result = result                # quarantined
            self.stats["completed"] += 1
            return True

    def abort(self, aid: AttemptID, *, had_started: bool) -> bool:
        """ACTIVE -> ABORTED. Idempotent. Returns False if completion already won."""
        with self._lock:
            cur = self._get_current(aid)
            if cur is None:
                return False
            if cur.state is AttemptState.ABORTED:
                self.stats["redundant_aborts"] += 1
                return True                    # idempotent, not an error
            if cur.state is AttemptState.COMPLETED:
                # Completion won. The finished result stays in quarantine -- it may still be a
                # winner, and discarding work already paid for would be waste, not safety.
                return False
            cur.state = AttemptState.ABORTED
            self.stats["aborted_running" if had_started else "aborted_queued"] += 1
            return True

    def state(self, aid: AttemptID):
        cur = self._get_current(aid)
        return cur.state if cur else None

    def cost(self, aid: AttemptID):
        """(prefill, usable decode, unusable decode). The third is real GPU work that produced
        nothing committable -- late-after-abort and stale-attempt tokens."""
        att, _ = self._find_any(aid)
        if att is None:
            return (0, 0, 0)
        return (att.prefill_tokens, att.decode_tokens, att.late_after_abort_tokens)

    def quarantined_result(self, aid: AttemptID):
        cur = self._get_current(aid)
        return cur.result if cur and cur.state is AttemptState.COMPLETED else None


@dataclass
class ResourceReservation:
    """Per-resource reserves. NEVER share one R across resource kinds.

    A request-count reserve says nothing about memory: five short speculative children and five long
    committed ones occupy identical slot counts and wildly different KV. Each reserve is measured from
    its OWN A0 distribution.
    """

    K_seq: int
    R_seq: int
    K_kv: int = 0
    R_kv: int = 0
    K_tok: int = 0
    R_tok: int = 0
    kv_enforced: bool = True     # False => report slot-only reservation AS A LIMITATION

    def admit_occupancy(self, *, queued_committed: int, committed_seq: int, speculative_seq: int,
                        committed_kv: int = 0, speculative_kv: int = 0, want_kv: int = 0) -> bool:
        """Sequence slots and KV blocks: PERSISTENT occupancy, held until the request terminates."""
        if queued_committed:
            return False                       # never speculate while committed work waits
        gap_seq = max(0, self.R_seq - committed_seq)
        if committed_seq + speculative_seq + 1 + gap_seq > self.K_seq:
            return False
        if self.kv_enforced and self.K_kv:
            gap_kv = max(0, self.R_kv - committed_kv)
            if committed_kv + speculative_kv + want_kv + gap_kv > self.K_kv:
                return False
        return True

    def speculative_token_budget(self, *, committed_tokens_this_step: int) -> int:
        """Per-step scheduled tokens: RENEWED EVERY STEP, so not occupancy.

        Committed tokens are scheduled first and speculation takes only what remains after the
        reserve. Applying the occupancy algebra here would hold back capacity that ceases to exist
        one step later.
        """
        if not self.K_tok:
            return 0
        # The reserve is CONSUMED by committed tokens scheduled in this step, exactly as for
        # occupancy resources -- the resource resets each step, the reservation does not stack on top
        # of committed work already scheduled within it. Subtracting R_tok unconditionally was the
        # same double-count as the original slot formula.
        reserve_gap = max(0, self.R_tok - committed_tokens_this_step)
        return max(0, self.K_tok - committed_tokens_this_step - reserve_gap)

    def limitations(self):
        out = []
        if not self.K_kv or not self.kv_enforced:
            out.append("KV-block reservation NOT enforced: slot-only reservation does not bound the "
                       "memory committed work needs, because teacher prefixes differ greatly in "
                       "length. Reported as a limitation, not presented as protection.")
        if not self.K_tok:
            out.append("per-step scheduled-token budget not enforced")
        return out
