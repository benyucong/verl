"""Explicit selection policy for OPDFlow. Mechanism stays in the runtime; policy lives here.

WHY THIS EXISTS. `score(chunk)` is too ambiguous to be a contract. Offline analysis ranked candidate
chunks by the CHUNK MEAN of a per-token signal while the generation pipeline ranked by the ANCHOR
TOKEN, and nothing detected the mismatch because both were just "the score". The two rankings are
near-disjoint -- measured Jaccard 0.013-0.027, median 0.000 -- so an entire validation compared a
chunk-mean proxy against an anchor-ranked generation workload and the conclusion drawn from it was
invalid. A selection policy is at least seven independent decisions, and every one of them changes
which chunks the teacher is asked to generate:

    signal source        what per-token quantity is being ranked
    signal producer      WHO computes it, which fixes WHEN it becomes available
    aggregation          how per-token values become one score per candidate  <-- the bug lived here
    anchor/window        the exact offset and span the score refers to
    candidates           which start positions are eligible at all
    constraint           how overlapping candidates are resolved
    budgets              M chunks, N continuations each

OPDFlow is selector-AGNOSTIC. Entropy and teacher-gap are both heuristic policy plugins; neither is
part of the system design, and the runtime must not embed either. This module is the boundary.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field

# Which component computes a signal -- this FIXES when it is available in the pipeline, which is a
# scheduling fact, not a taste. student_forward signals are ready before any teacher round-trip;
# teacher_scoring signals require one; teacher_generation signals require the expensive path the
# selector is supposed to be economising, so a selector using one is necessarily two-stage.
SIGNAL_PRODUCERS = {
    "student_entropy":     "student_forward",
    "student_logprob":     "student_forward",
    "uniform":             "none",
    "random":              "none",
    "teacher_gap_abs":     "teacher_scoring",
    "teacher_gap_signed":  "teacher_scoring",
    "teacher_gap_pos":     "teacher_scoring",
    "teacher_reverse_kl":  "teacher_scoring",
    "verifier_score":      "external",
}

AGGREGATIONS = ("anchor", "mean", "sum", "max")
CANDIDATE_POLICIES = ("all_starts", "stride", "disjoint_grid")
CONSTRAINTS = ("greedy_resample_nonoverlap", "dp_optimal_nonoverlap", "none")
RANKING_STATES = ("finalized", "provisional")

# Fields that determine WHICH chunks get selected. Two specs agreeing on all of these select the same
# chunks from the same signal; differing on any one selects a different set. M is excluded on purpose
# -- comparing M=8 against M=10 under one policy is a budget question, which is legitimate and is not
# a policy mismatch. N is included: it changes the supervision each selected chunk carries, so
# comparing across N is comparing different measurements.
POLICY_FIELDS = ("signal_source", "aggregation", "anchor_offset", "chunk_tokens",
                 "candidate_policy", "candidate_stride", "constraint", "N")


class SelectorMismatch(RuntimeError):
    """Raised when analysis would compare scores it computed against chunks generated another way."""


@dataclass(frozen=True)
class SelectorSpec:
    """A complete, executable description of one selection policy.

    OmniOPD as published is
        SelectorSpec("student_entropy", aggregation="anchor", anchor_offset=-1, M=10)
    and GapSelect as specified is
        SelectorSpec("teacher_gap_abs", aggregation="mean", M=8)
    which differ in BOTH the signal and the aggregation. Before this dataclass existed both were
    written as "the score" and the difference was invisible.
    """

    signal_source: str
    aggregation: str = "anchor"
    # Offset from the chunk start to the token whose signal is read, for aggregation="anchor".
    # -1 is OmniOPD's rule: response position t is scored by H[prompt_len + t - 1], the logits that
    # PRODUCED y_t rather than those it conditions. Ignored for window aggregations, and asserted so.
    anchor_offset: int = -1
    chunk_tokens: int = 50                      # C, the audit window
    candidate_policy: str = "all_starts"
    candidate_stride: int = 1
    constraint: str = "greedy_resample_nonoverlap"
    M: int = 10                                 # chunks audited per response
    N: int = 10                                 # teacher continuations per chunk
    ranking_state: str = "finalized"            # provisional => may be revised as tokens stream in
    notes: str = ""                             # free text; NEVER part of the hash or the match

    def __post_init__(self):
        if self.signal_source not in SIGNAL_PRODUCERS:
            raise ValueError(f"unknown signal_source {self.signal_source!r}; "
                             f"register it in SIGNAL_PRODUCERS with its producer so the scheduler "
                             f"knows when the signal becomes available")
        for name, allowed in (("aggregation", AGGREGATIONS),
                              ("candidate_policy", CANDIDATE_POLICIES),
                              ("constraint", CONSTRAINTS),
                              ("ranking_state", RANKING_STATES)):
            if getattr(self, name) not in allowed:
                raise ValueError(f"{name}={getattr(self, name)!r} not in {allowed}")
        if self.chunk_tokens < 1 or self.M < 1 or self.N < 1:
            raise ValueError("chunk_tokens, M and N must all be >= 1")
        if self.candidate_stride < 1:
            raise ValueError("candidate_stride must be >= 1")
        if self.aggregation != "anchor" and self.anchor_offset != -1:
            # A non-default offset that the aggregation ignores is a silent lie in the manifest.
            raise ValueError(f"anchor_offset={self.anchor_offset} is meaningless for "
                             f"aggregation={self.aggregation!r}; leave it at the default")

    @property
    def signal_producer(self) -> str:
        """student_forward | teacher_scoring | teacher_generation | external | none."""
        return SIGNAL_PRODUCERS[self.signal_source]

    @property
    def needs_teacher_before_selection(self) -> bool:
        """True when the signal only exists after a teacher round-trip.

        The scheduler needs this: such a policy cannot select during streaming without either a
        teacher scoring pass on partial output or a deferred decision. It is a property of the
        POLICY, which is exactly why it belongs in the spec rather than in the runtime.
        """
        return self.signal_producer in ("teacher_scoring", "teacher_generation")

    def policy_key(self) -> dict:
        return {k: getattr(self, k) for k in POLICY_FIELDS}

    def version(self) -> str:
        """Stable hash over the policy-defining fields only.

        Excludes M (budget comparisons are legitimate within one policy) and `notes`, so re-wording a
        comment never invalidates a stored manifest.
        """
        blob = json.dumps(self.policy_key(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    def to_manifest(self) -> dict:
        """Everything a manifest must record, with the derived facts made explicit rather than
        recoverable-in-principle. A reader must not have to re-run this module to know what was run."""
        d = asdict(self)
        d.update(signal_producer=self.signal_producer,
                 needs_teacher_before_selection=self.needs_teacher_before_selection,
                 selector_version=self.version(),
                 policy_fields=list(POLICY_FIELDS))
        return d

    @classmethod
    def from_manifest(cls, d: dict) -> "SelectorSpec":
        return cls(**{k: v for k, v in d.items()
                      if k in cls.__dataclass_fields__})

    def describe(self) -> str:
        agg = (f"anchor@{self.anchor_offset}" if self.aggregation == "anchor" else self.aggregation)
        return (f"{self.signal_source}[{self.signal_producer}] {agg} C={self.chunk_tokens} "
                f"{self.constraint} M={self.M} N={self.N} v{self.version()}")


def assert_spec_matches(analysis: SelectorSpec, manifest_spec: SelectorSpec, *,
                        allow_cross_policy: bool = False, context: str = "") -> None:
    """FATAL comparison between the spec analysis used and the spec that generated the chunks.

    This is the guard for the exact failure that occurred: analysis re-derived a selection with
    aggregation="mean" and then looked up teacher continuations that had been generated for an
    aggregation="anchor" selection. Both ran without error and produced a plausible table.

    Cross-policy comparison is a legitimate experiment -- "how much does aggregation change what gets
    selected?" is a real question -- but it must be REQUESTED, never defaulted into.
    """
    diffs = {k: (getattr(analysis, k), getattr(manifest_spec, k))
             for k in POLICY_FIELDS
             if getattr(analysis, k) != getattr(manifest_spec, k)}
    if not diffs:
        return
    if allow_cross_policy:
        return
    lines = [f"    {k}: analysis={a!r} but manifest={m!r}" for k, (a, m) in diffs.items()]
    raise SelectorMismatch(
        f"selector policy mismatch{' in ' + context if context else ''} -- refusing to compare "
        f"scores computed under one policy against chunks generated under another.\n"
        + "\n".join(lines)
        + f"\n  analysis: {analysis.describe()}"
        + f"\n  manifest: {manifest_spec.describe()}\n"
        f"  If this IS the experiment (e.g. measuring how aggregation changes selection), pass "
        f"allow_cross_policy=True explicitly and say so in the write-up.")


# ---------------------------------------------------------------------------------------------
# Reference specs. Named so that write-ups cite a spec instead of prose, and so the near-disjoint
# pair that caused the bug is visible side by side.
# ---------------------------------------------------------------------------------------------

OMNIOPD_PUBLISHED = SelectorSpec(
    signal_source="student_entropy", aggregation="anchor", anchor_offset=-1,
    chunk_tokens=50, M=10, N=10,
    notes="OmniOPD 3.2.3 Eq 6-7: peak-entropy ANCHOR. The primary workload.")

GAPSELECT_M10 = SelectorSpec(
    signal_source="teacher_gap_abs", aggregation="mean", chunk_tokens=50, M=10, N=10,
    notes="Teacher-side plugin. Kept as a scheduler stress workload because its signal only exists "
          "after teacher scoring; NOT claimed to improve the algorithm.")

GAPSELECT_M8_CLOSED = SelectorSpec(
    signal_source="teacher_gap_abs", aggregation="mean", chunk_tokens=50, M=8, N=10,
    notes="CLOSED. Failed both pre-registered gates against OMNIOPD_PUBLISHED "
          "(docs/genopdflow_phase2_gapselect.md). Retained only so the closed claim is citable.")


# ---------------------------------------------------------------------------------------------
# Reference executor. One function, driven entirely by the spec, so that aggregation is a PARAMETER
# rather than something welded into a call site. `select_chunks` in omniopd_chunks.py is the
# in-trainer twin and is verified against this for the anchor case by the regression test -- if the
# two ever diverge, that test fails rather than a paper table being wrong.
# ---------------------------------------------------------------------------------------------

def candidate_scores(signal, prompt_len, resp_len, spec):
    """Per-candidate score, one entry per eligible chunk start t.

    `signal` is indexed over the WHOLE sequence: response position t is signal[prompt_len + t + off]
    for aggregation="anchor" with off = spec.anchor_offset, and window aggregations read the response
    span [t, t+C). Returns [(t, score)] with candidates that would run past the response dropped --
    a truncated audit window is not the same measurement as a full one.
    """
    C, out = spec.chunk_tokens, []
    for t in range(0, resp_len - C + 1, spec.candidate_stride):
        if spec.aggregation == "anchor":
            i = prompt_len + t + spec.anchor_offset
            if not (0 <= i < len(signal)):
                continue
            out.append((t, signal[i]))
            continue
        lo = prompt_len + t
        w = signal[lo:lo + C]
        if len(w) < C:
            continue
        if spec.aggregation == "mean":
            out.append((t, sum(w) / C))
        elif spec.aggregation == "sum":
            out.append((t, sum(w)))
        elif spec.aggregation == "max":
            out.append((t, max(w)))
    return out


def apply_constraint(scored, spec):
    """Resolve overlapping candidates into at most M non-overlapping chunk starts.

    Tie-break is toward the LARGER t, matching the in-trainer operator: sorting (score, t)
    descending puts the later position first among equals. It is recorded here rather than left to
    the sort's incidental behaviour because it decides real chunks whenever a signal has plateaus.
    """
    C = spec.chunk_tokens
    if spec.constraint == "none":
        return sorted(t for t, _ in sorted(scored, key=lambda st: (-st[1], -st[0]))[:spec.M])
    if spec.constraint == "greedy_resample_nonoverlap":
        chosen = []
        for t, _ in sorted(scored, key=lambda st: (-st[1], -st[0])):
            if any(abs(t - u) < C for u in chosen):
                continue
            chosen.append(t)
            if len(chosen) == spec.M:
                break
        return sorted(chosen)
    if spec.constraint == "dp_optimal_nonoverlap":
        w = {t: s for t, s in scored}
        ts = sorted(w)
        n = len(ts)
        nxt = []                      # first index whose start is >= ts[i] + C
        j = 0
        for i in range(n):
            j = max(j, i + 1)
            while j < n and ts[j] - ts[i] < C:
                j += 1
            nxt.append(j)
        best = [[0.0] * (spec.M + 1) for _ in range(n + 1)]
        take = [[False] * (spec.M + 1) for _ in range(n + 1)]
        for i in range(n - 1, -1, -1):
            for k in range(1, spec.M + 1):
                skip = best[i + 1][k]
                got = w[ts[i]] + best[nxt[i]][k - 1]
                if got > skip:
                    best[i][k], take[i][k] = got, True
                else:
                    best[i][k] = skip
        sel, i, k = [], 0, spec.M
        while i < n and k > 0:
            if take[i][k]:
                sel.append(ts[i]); i, k = nxt[i], k - 1
            else:
                i += 1
        return sorted(sel)
    raise ValueError(f"unhandled constraint {spec.constraint!r}")


def select(signal, prompt_len, resp_len, spec):
    """Execute a SelectorSpec. The ONLY supported way to turn a per-token signal into chunk starts."""
    return apply_constraint(candidate_scores(signal, prompt_len, resp_len, spec), spec)


def manifest_block(specs_by_arm, *, extra=None):
    """The selector half of a generation manifest: one full spec per arm, plus a cross-arm summary.

    Recorded per arm so a later reader can reconstruct exactly what was selected without rerunning
    anything, and so `assert_spec_matches` has something to compare against.
    """
    block = {
        "schema_version": 1,
        "arms": {name: spec.to_manifest() for name, spec in specs_by_arm.items()},
        "selector_versions": {name: spec.version() for name, spec in specs_by_arm.items()},
        "distinct_policies": sorted({spec.version() for spec in specs_by_arm.values()}),
        "signal_producers": sorted({spec.signal_producer for spec in specs_by_arm.values()}),
        "any_needs_teacher_before_selection": any(
            s.needs_teacher_before_selection for s in specs_by_arm.values()),
    }
    if extra:
        block.update(extra)
    return block


def spec_from_manifest_arm(manifest, arm):
    """Recover one arm's spec from a manifest, raising rather than defaulting if it is absent.

    A manifest written before this schema existed has no recoverable policy, and guessing one is how
    the original mismatch survived. Refuse instead.
    """
    sel = (manifest or {}).get("selectors")
    if not sel or "arms" not in sel:
        raise SelectorMismatch(
            "manifest has no `selectors` block -- its selection policy is unrecoverable, so no "
            "analysis may claim to reproduce it. Regenerate the plan with manifest_block().")
    if arm not in sel["arms"]:
        raise SelectorMismatch(f"arm {arm!r} not in manifest; have {sorted(sel['arms'])}")
    return SelectorSpec.from_manifest(sel["arms"][arm])
