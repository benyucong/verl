"""Attempt identity and admission control for speculative teacher work.

TWO DISTINCTIONS THIS MODULE EXISTS TO KEEP:

1. LOGICAL WORK vs PHYSICAL ATTEMPT. `TeacherRequestKey` names a unit of work; an `AttemptID` names
   one physical execution of it. A revoked anchor that re-enters must NOT reuse its previous engine
   request id -- the engine may reject the duplicate, or worse, attribute a late output or an abort
   acknowledgement from the dead attempt to the live one. The sampling seed stays stable across
   attempts -- deterministic INTENT -- while the identity the engine sees changes.

2. RESOURCE KINDS. Sequence slots, KV blocks and per-step scheduled tokens are different resources
   and must not share a reserve. Slots and KV blocks are PERSISTENT OCCUPANCY: a request holds them
   until it terminates, so the reserve-gap algebra applies. Scheduled tokens are RENEWED EVERY STEP,
   so treating them as occupancy would reserve capacity that no longer exists a step later. Committed
   tokens are scheduled first and speculation takes the remainder -- but the reserve is still CONSUMED
   within the step, so it is max(0, K_tok - max(R_tok, committed)), not K_tok - committed - R_tok.

The terminal transition is atomic and one-way. Whichever of completion or abort wins is
authoritative, and every later RESULT from that attempt is discarded -- otherwise a cancelled child
could still supply a continuation that reaches training, which is a correctness bug rather than an
accounting one.

RESULT ACCEPTANCE AND COST ACCOUNTING ARE INDEPENDENT PATHS. A late output after an abort must never
become supervision, but it consumed real GPU time; dropping it from the ACCOUNTING as well would
undercount precisely the wasted work A1 and A2 are compared on. Every output is therefore costed and
bucketed by usability, and only the result is discarded. Cumulative outputs are accounted through a
per-attempt high-water mark, so redelivered messages inflate nothing.

A late output from a superseded or aborted attempt is an EXPECTED asynchronous race: discarded,
accounted, recorded as a diagnostic -- never a crash. Hard errors are reserved for unknown request
ids, malformed identities and impossible transitions such as double completion.

ON SEEDS: a stable seed expresses deterministic INTENT. It is NOT a claim that a restarted attempt is
bitwise reproducible under a different continuous-batching schedule -- batch composition changes
kernel reduction order. Algorithmically a restart is a fresh valid teacher sample; bitwise
reproducibility is a separate question and must be tested separately if it is ever relied upon.
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
    """Raised ONLY on unknown request ids, malformed identities, or impossible transitions.

    Deliberately NOT raised for a late output from a superseded or aborted attempt: that is an
    expected asynchronous race after revocation, and crashing a run on it would turn normal operation
    into a fault.
    """


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
    """An IMMUTABLE PHYSICAL-WORK LEDGER. It records what happened, never whether it was useful.

    Usability cannot be known when tokens arrive: it depends on whether the canonical post-EOS
    selector later commits this anchor. Banking pre-abort decode as "usable" on arrival would
    undercount waste for every anchor that is proposed, decoded, and then loses -- which is the
    dominant waste mode the A1/A2 comparison exists to measure.
    """

    aid: AttemptID
    state: AttemptState = AttemptState.ACTIVE
    admitted: bool = False             # the engine began work: prefill started
    logical_prefix_tokens: int = 0     # what was SUBMITTED
    physical_prefill_tokens: int = 0   # what the engine RECOMPUTED after APC reuse
    prefill_recorded: bool = False     # idempotency guard: engines may report more than once
    # decode, bucketed by WHEN it happened -- not by whether it turned out to be useful
    decode_while_active: int = 0
    decode_after_abort: int = 0
    decode_while_stale: int = 0        # emitted by an attempt already superseded
    max_observed_decode: int = 0       # high-water mark for CUMULATIVE outputs
    duplicate_cumulative: int = 0
    outputs_after_completion: int = 0
    result: object = None              # quarantined; reaches training only via a CommitPlan
    # ABORT IS NOT INSTANTANEOUS. A GPU step may already be in flight when the abort is requested, so
    # "cancelled" cannot promise "no further engine work" at that instant. The gap between these is a
    # measured quantity, not an assumption.
    abort_requested_seq: int = -1      # monotonic tick when cancellation was requested
    abort_acknowledged_seq: int = -1   # when the engine confirmed it
    last_physical_work_seq: int = -1   # the last tick at which this attempt actually consumed GPU

    @property
    def total_decode(self) -> int:
        return self.decode_while_active + self.decode_after_abort + self.decode_while_stale


class AttemptRegistry:
    """Tracks the CURRENT attempt per logical request, and enforces one terminal state each.

    RESULTS are accepted only from the current, still-ACTIVE attempt. COSTS are accounted from every
    attempt, current or superseded, terminated or not -- a superseded attempt that keeps emitting is
    consuming real GPU time, and forgetting it would understate speculative waste.

    Superseded attempts are retained rather than deleted for exactly that reason.
    """

    def __init__(self):
        self._cur: dict = {}                  # request_key -> current _Attempt
        self._retired: dict = {}              # engine_request_id -> superseded _Attempt
        # THE SINGLE ACCEPTED RESULT per logical request. Two attempts can both complete -- attempt 0
        # finishing as revocation lands, attempt 1 finishing after re-entry -- and only one of them
        # may ever supply supervision.
        self._accepted: dict = {}             # request_key -> engine_request_id of the accepted one
        self._seq = 0                         # monotonic tick for the abort timeline
        self._lock = threading.Lock()
        self.stats = {"completed": 0, "aborted_queued": 0, "aborted_running": 0,
                      "restarts": 0, "results_discarded": 0, "redundant_aborts": 0,
                      # physical cost that produced nothing usable -- the quantity A1/A2 compare
                      "decode_after_abort_tokens": 0, "decode_while_stale_tokens": 0,
                      "duplicate_cumulative_msgs": 0, "outputs_after_completion": 0,
                      "duplicate_prefill_reports": 0, "duplicate_completions_rejected": 0,
                      "physical_prefill_tokens_total": 0, "logical_prefix_tokens_total": 0,
                      "aborted_never_admitted": 0}

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

    def mark_admitted(self, aid: AttemptID, *, logical_prefix_tokens: int = 0) -> None:
        """The engine began work on this attempt. Until this is called it is QUEUED, and a
        cancellation costs nothing at all -- not even prefill."""
        with self._lock:
            att, _ = self._find_any(aid)
            if att is None:
                raise AttemptError(f"admit for unknown attempt {aid.engine_request_id}")
            att.admitted = True
            if logical_prefix_tokens:
                att.logical_prefix_tokens = logical_prefix_tokens

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

    def record_prefill(self, aid: AttemptID, *, physical_tokens: int,
                       logical_tokens: int = 0) -> None:
        """IDEMPOTENT. Engines may report prefill more than once for one request.

        Two quantities, reported separately and never conflated: `logical_tokens` is the prefix
        submitted, `physical_tokens` is what the engine actually recomputed after APC reuse. A nested
        speculative prefix can be almost entirely cached, so the logical figure would wildly
        overstate the GPU cost; the physical figure is what was paid.

        Recording prefill implies admission -- an engine does not prefill a request it never took.
        """
        with self._lock:
            att, _ = self._find_any(aid)
            if att is None:
                raise AttemptError(f"prefill for unknown attempt {aid.engine_request_id}")
            if att.prefill_recorded:
                self.stats["duplicate_prefill_reports"] += 1
                return
            att.prefill_recorded = True
            att.admitted = True
            self._seq += 1
            att.last_physical_work_seq = self._seq
            att.physical_prefill_tokens = physical_tokens
            if logical_tokens:
                att.logical_prefix_tokens = logical_tokens
            self.stats["physical_prefill_tokens_total"] += physical_tokens
            self.stats["logical_prefix_tokens_total"] += att.logical_prefix_tokens

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
            self._seq += 1
            att.last_physical_work_seq = self._seq
            if not is_current:
                att.decode_while_stale += new
                self.stats["decode_while_stale_tokens"] += new
            elif att.state is AttemptState.ACTIVE:
                att.decode_while_active += new      # WHEN it happened; not a usability claim
            elif att.state is AttemptState.ABORTED:
                att.decode_after_abort += new
                self.stats["decode_after_abort_tokens"] += new
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
            # SINGLE ACCEPTED RESULT. If another attempt for this logical request already supplied
            # one, this completion is real physical work but must never supervise: two attempts
            # finishing in a race would otherwise both be counted reusable and the same chunk would
            # be trained on twice.
            if self._accepted.get(aid.request_key) not in (None, aid.engine_request_id):
                cur.state = AttemptState.COMPLETED
                self.stats["duplicate_completions_rejected"] += 1
                self.stats["results_discarded"] += 1
                return False
            cur.state = AttemptState.COMPLETED
            cur.result = result                # quarantined
            self._accepted[aid.request_key] = aid.engine_request_id
            self.stats["completed"] += 1
            return True

    def abort(self, aid: AttemptID) -> bool:
        """ACTIVE -> ABORTED. Idempotent. Returns False if completion already won.

        Whether this was a queued or a running cancellation is derived from the LEDGER -- whether the
        engine ever admitted the attempt -- rather than taken on the caller's word. A caller that
        merely believes a request was still queued is exactly how a prefill gets lost from the
        accounting.
        """
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
            self._seq += 1
            cur.abort_requested_seq = self._seq
            cur.state = AttemptState.ABORTED
            if not cur.admitted:
                # ZERO TEACHER-MODEL WORK -- not "free". No prefill and no decode reached the model,
                # but the request still cost queue occupancy, an RPC round trip and control-plane
                # bookkeeping. Those are small, and they are not nothing.
                self.stats["aborted_never_admitted"] += 1
                self.stats["aborted_queued"] += 1
            elif cur.decode_while_active == 0:
                self.stats["aborted_queued"] += 1             # admitted and prefilled, never decoded
            else:
                self.stats["aborted_running"] += 1
            return True

    def acknowledge_abort(self, aid: AttemptID) -> None:
        """The engine confirmed the cancellation. Work may still have landed between request and this."""
        with self._lock:
            att, _ = self._find_any(aid)
            if att is None:
                raise AttemptError(f"abort ack for unknown attempt {aid.engine_request_id}")
            self._seq += 1
            att.abort_acknowledged_seq = self._seq

    def abort_timeline(self, aid: AttemptID):
        """(requested, acknowledged, last_physical_work) ticks. Work after `requested` is real."""
        att, _ = self._find_any(aid)
        if att is None:
            return (-1, -1, -1)
        return (att.abort_requested_seq, att.abort_acknowledged_seq, att.last_physical_work_seq)

    def accepted_attempt(self, request_key: str):
        return self._accepted.get(request_key)

    def state(self, aid: AttemptID):
        cur = self._get_current(aid)
        return cur.state if cur else None

    def physical_cost(self, aid: AttemptID):
        """(physical prefill, decode-while-active, decode-after-terminal). PHYSICAL WORK ONLY.

        Deliberately NOT (prefill, usable, unusable): usability depends on the canonical commit set,
        which does not exist yet when these tokens arrive. Use `classify()` once it does.
        """
        att, _ = self._find_any(aid)
        if att is None:
            return (0, 0, 0)
        return (att.physical_prefill_tokens, att.decode_while_active,
                att.decode_after_abort + att.decode_while_stale)

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


def classify_after_commit(registry: "AttemptRegistry", committed_request_keys,
                          audit_ok=None) -> dict:
    """Split the physical ledger into REUSABLE and WASTED. PURE and REPEATABLE.

    Reads the ledger and mutates nothing. Running it twice, or again with a revised commit set,
    returns a consistent view of the same immutable costs -- the ledger records physical work, this
    only interprets it.

    USABILITY REQUIRES ALL FOUR:
      1. the anchor survives the canonical commit;
      2. the attempt COMPLETED;
      3. its output passes the AuditKey checks (`audit_ok`, default: all pass);
      4. it is THE SINGLE ACCEPTED attempt for that logical request.

    Condition 4 is what makes retries honest. If attempt 0 decodes 200 tokens and is aborted, and
    attempt 1 completes after re-entry, only attempt 1 is usable -- and attempt 0's ENTIRE prefill and
    decode is wasted, not merely its post-abort tail. Its clean pre-abort decode bought nothing,
    because the continuation that will actually be used came from a different attempt.

    Post-terminal tokens -- after an abort, or from a superseded attempt -- are always wasted,
    whatever the commit decides.
    """
    committed = set(committed_request_keys)
    ok = (lambda key: True) if audit_ok is None else audit_ok
    out = {"reusable_decode": 0, "wasted_decode": 0,
           "wasted_committed_unfinished": 0, "wasted_lost_anchor": 0,
           "wasted_superseded_attempt": 0, "wasted_audit_failed": 0,
           "wasted_post_terminal": 0,
           "physical_prefill_reusable": 0, "physical_prefill_wasted": 0,
           "logical_prefix_total": 0, "n_reusable": 0, "n_wasted": 0}
    with registry._lock:                                    # noqa: SLF001 -- same module
        attempts = list(registry._cur.values()) + list(registry._retired.values())  # noqa: SLF001
        accepted = dict(registry._accepted)                 # noqa: SLF001
    for att in attempts:
        key = att.aid.request_key
        post_terminal = att.decode_after_abort + att.decode_while_stale
        out["wasted_post_terminal"] += post_terminal
        out["logical_prefix_total"] += att.logical_prefix_tokens

        is_accepted = accepted.get(key) == att.aid.engine_request_id
        usable = (key in committed
                  and att.state is AttemptState.COMPLETED
                  and is_accepted
                  and ok(key))
        if usable:
            out["reusable_decode"] += att.decode_while_active
            out["physical_prefill_reusable"] += att.physical_prefill_tokens
            out["n_reusable"] += 1
        else:
            # Order matters. "Another attempt won" is checked BEFORE "this one did not complete":
            # for a retry, both are true, and the informative fact is that a sibling supplied the
            # continuation -- so this attempt's whole cost bought nothing, however far it got.
            someone_else_accepted = (accepted.get(key) is not None
                                     and accepted.get(key) != att.aid.engine_request_id)
            if key not in committed:
                bucket = "wasted_lost_anchor"
            elif someone_else_accepted:
                bucket = "wasted_superseded_attempt"
            elif att.state is not AttemptState.COMPLETED:
                bucket = "wasted_committed_unfinished"
            else:
                bucket = "wasted_audit_failed"
            out[bucket] += att.decode_while_active
            out["wasted_decode"] += att.decode_while_active
            out["physical_prefill_wasted"] += att.physical_prefill_tokens
            out["n_wasted"] += 1
    out["wasted_decode"] += out["wasted_post_terminal"]
    return out
