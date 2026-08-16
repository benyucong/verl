"""Explicit selection policy for OPDFlow. Mechanism stays in the runtime; policy lives here.

WHY THIS EXISTS. `score(chunk)` is too ambiguous to be a contract. Offline analysis ranked candidate
chunks by the CHUNK MEAN of a per-token signal while the generation pipeline ranked by the ANCHOR
TOKEN, and nothing detected the mismatch because both were just "the score". The two rankings are
near-disjoint -- measured Jaccard 0.013-0.027, median 0.000 -- so an entire validation compared a
chunk-mean proxy against an anchor-ranked generation workload, and its conclusion was invalid.

THREE THINGS THIS MODULE INSISTS ON, each because the obvious shortcut is unsafe:

1. THREE HASHES, NOT ONE. A single hash forces a bad choice: include the budget and legitimate
   M=8-vs-M=10 sweeps trip the guard; exclude it and data generated at M=8 can be analysed as M=10
   without tripping anything -- the same class of bug as the aggregation mismatch. So:
       ranker_hash             the ranking RULE: signal identity, aggregation, anchor, candidates,
                               constraint, tie-break. Excludes budget, so controlled sweeps can
                               assert "same ranker, different budget" and mean it.
       selection_instance_hash ranker + M + C. THIS is what generation-vs-analysis must match:
                               M and C together are the audit budget, and a set selected under one
                               budget is not a set selected under another.
       workload_hash           selection instance + N, teacher sampling params, models, seeds,
                               continuation cap -- everything that changes what the numbers MEAN.

2. THE SIGNAL PROVIDER IS DECLARED, NOT DERIVED FROM THE NAME. "student entropy" can come from an
   exact GPU-side reduction in the rollout sampler, an exact post-EOS actor forward, a top-k
   approximation, or a separate scoring worker. Those differ in availability, fidelity and systems
   cost, and only the first can score a chunk while the trajectory is still generating -- which is
   the whole basis of same-trajectory overlap. A name cannot carry that, so it is four fields.

3. AN UNAVAILABLE SIGNAL IS `DEFER`, NEVER A FABRICATED ONE. The pipeline currently contains a
   fallback that substitutes uniform ones when a signal is missing: selection becomes arbitrary while
   retained-value metrics keep reporting as though it were real. That pattern produces plausible
   numbers from a selector that is not selecting. Here the only legal outcomes are a real ranking or
   DEFER, and inventing values raises.

OPDFlow is selector-AGNOSTIC. Entropy and teacher-gap are both heuristic policy plugins; neither is
part of the system design. This module is the boundary.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field

# --------------------------------------------------------------------------------------------
# Signal identity: semantics x provider x availability x fidelity.
# --------------------------------------------------------------------------------------------

SIGNAL_SEMANTICS = (
    "student_entropy",        # H_t = -sum_v p_v log p_v over the student's next-token distribution
    "student_logprob",        # log p_S(y_t) at the realised token
    "teacher_gap_abs",        # |log p_S(y_t) - log p_T(y_t)|; a RANKING SCORE, never a divergence
    "teacher_gap_signed",
    "teacher_gap_pos",
    "teacher_reverse_kl",
    "verifier_score",
    "uniform",
    "random",
)

# When a provider's value can first be read, relative to the trajectory it scores. This is the field
# that decides whether same-trajectory teacher overlap is possible at all.
AVAILABILITY = (
    "per_token_online",       # during decode, token by token -- enables progressive overlap
    "per_chunk_online",       # when a streaming chunk closes
    "post_trajectory",        # only after EOS (a second forward, or a trainer-side quantity)
    "post_teacher_scoring",   # only after a teacher round-trip
    "static",                 # needs nothing
)

FIDELITY = (
    "exact",                  # full-vocabulary, no truncation
    "approx_topk",            # truncated support plus a tail bucket
    "exact_on_realised_token",  # exact, but only at the token actually sampled
    "none",
)

# Registry of PROVIDERS -- the concrete mechanism that computes a signal. Declaring a provider that
# is not registered raises, because an unregistered provider has no known availability, and the
# scheduler would have to guess exactly where guessing has already cost this project a result.
PROVIDERS = {
    # student_entropy, four genuinely different mechanisms:
    "rollout_exact_scalar":    dict(availability="per_token_online",     fidelity="exact"),
    "rollout_topk_approx":     dict(availability="per_token_online",     fidelity="approx_topk"),
    "post_eos_actor_forward":  dict(availability="post_trajectory",      fidelity="exact"),
    "student_scoring_worker":  dict(availability="post_trajectory",      fidelity="exact"),
    "trainer_forward":         dict(availability="post_teacher_scoring", fidelity="exact"),
    # teacher-side:
    "teacher_scoring_k0":      dict(availability="post_teacher_scoring", fidelity="exact_on_realised_token"),
    "teacher_scoring_topk":    dict(availability="post_teacher_scoring", fidelity="approx_topk"),
    # signal-free:
    "none":                    dict(availability="static",               fidelity="none"),
    "external":                dict(availability="post_trajectory",      fidelity="exact"),
}

AGGREGATIONS = ("anchor", "mean", "sum", "max")
CANDIDATE_POLICIES = ("all_starts", "stride", "disjoint_grid")
CONSTRAINTS = ("greedy_resample_nonoverlap", "dp_optimal_nonoverlap", "none")
TIE_BREAKS = ("larger_t", "smaller_t")
RANKING_STATES = ("finalized", "provisional")

# The ranking RULE. Budget (M, C) is deliberately absent -- that is the point of a separate hash.
RANKER_FIELDS = ("signal_semantics", "signal_provider", "availability", "fidelity",
                 "aggregation", "anchor_offset", "candidate_policy", "candidate_stride",
                 "constraint", "tie_break")
# What generation and analysis must agree on. M and C are the audit budget and change the SET.
SELECTION_INSTANCE_FIELDS = RANKER_FIELDS + ("M", "chunk_tokens")


class SelectorMismatch(RuntimeError):
    """Raised when analysis would compare scores it computed against chunks generated another way."""


class SignalNotReady(RuntimeError):
    """Raised when a selector is asked to rank before its signal exists.

    The only correct responses to a missing signal are to wait (DEFER) or to fail. Substituting
    uniform values yields a ranking that looks real and is not.
    """


class _Defer:
    """Sentinel: the selector cannot decide yet and the caller must retry when the signal lands."""
    __slots__ = ()

    def __repr__(self):
        return "DEFER"

    def __bool__(self):
        # Never let `if selection:` quietly treat a deferral as "nothing selected".
        raise SignalNotReady("DEFER is not a selection; handle it explicitly before testing truth")


DEFER = _Defer()


@dataclass(frozen=True)
class SelectorSpec:
    """A complete, executable description of one selection policy.

    OmniOPD as published:
        SelectorSpec("student_entropy", "post_eos_actor_forward", aggregation="anchor", M=10)
    the same algorithm on the online path OPDFlow wants:
        SelectorSpec("student_entropy", "rollout_exact_scalar", aggregation="anchor", M=10)
    These share a `signal_semantics` and differ in everything that matters to the scheduler.
    """

    signal_semantics: str
    signal_provider: str
    # Declared explicitly and CHECKED against PROVIDERS rather than derived: a spec that lies about
    # when its signal arrives would mis-plan the pipeline, so the lie must be caught at construction.
    availability: str = ""
    fidelity: str = ""

    aggregation: str = "anchor"
    # Offset from the chunk start to the token whose signal is read, for aggregation="anchor".
    # -1 is OmniOPD's rule: response position t is scored by H[prompt_len + t - 1], the logits that
    # PRODUCED y_t rather than those it conditions.
    anchor_offset: int = -1
    candidate_policy: str = "all_starts"
    candidate_stride: int = 1
    constraint: str = "greedy_resample_nonoverlap"
    # Real policy, not an incidental property of the sort. The two offline rankers disagreed on this
    # and it changed 1 of 144 selections before it was caught.
    tie_break: str = "larger_t"

    chunk_tokens: int = 50                      # C: the audit window
    M: int = 10                                 # chunks audited per response
    N: int = 10                                 # teacher continuations per chunk (workload-level)
    ranking_state: str = "finalized"            # provisional => may be revised as tokens stream in
    notes: str = ""                             # free text; never part of any hash

    def __post_init__(self):
        if self.signal_semantics not in SIGNAL_SEMANTICS:
            raise ValueError(f"unknown signal_semantics {self.signal_semantics!r}")
        if self.signal_provider not in PROVIDERS:
            raise ValueError(f"unknown signal_provider {self.signal_provider!r}; register it in "
                             f"PROVIDERS with its availability and fidelity -- an unregistered "
                             f"provider has no known availability and the scheduler would guess")
        known = PROVIDERS[self.signal_provider]
        for f in ("availability", "fidelity"):
            if not getattr(self, f):
                object.__setattr__(self, f, known[f])
            elif getattr(self, f) != known[f]:
                raise ValueError(
                    f"{f}={getattr(self, f)!r} contradicts provider {self.signal_provider!r}, which "
                    f"is registered as {known[f]!r}. Fix the spec or register a new provider; do not "
                    f"paper over the difference.")
        for name, allowed in (("availability", AVAILABILITY), ("fidelity", FIDELITY),
                              ("aggregation", AGGREGATIONS), ("tie_break", TIE_BREAKS),
                              ("candidate_policy", CANDIDATE_POLICIES),
                              ("constraint", CONSTRAINTS), ("ranking_state", RANKING_STATES)):
            if getattr(self, name) not in allowed:
                raise ValueError(f"{name}={getattr(self, name)!r} not in {allowed}")
        if self.chunk_tokens < 1 or self.M < 1 or self.N < 1 or self.candidate_stride < 1:
            raise ValueError("chunk_tokens, M, N and candidate_stride must all be >= 1")
        if self.aggregation != "anchor" and self.anchor_offset != -1:
            raise ValueError(f"anchor_offset={self.anchor_offset} is meaningless for "
                             f"aggregation={self.aggregation!r}; leave it at the default")

    # -- the three hashes ----------------------------------------------------------------------

    @staticmethod
    def _h(d):
        return hashlib.sha256(
            json.dumps(d, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]

    def ranker_hash(self) -> str:
        """The ranking rule alone. Two specs sharing this rank candidates identically."""
        return self._h({k: getattr(self, k) for k in RANKER_FIELDS})

    def selection_instance_hash(self) -> str:
        """Ranker + audit budget (M, C). **This is the generation-vs-analysis check.**"""
        return self._h({k: getattr(self, k) for k in SELECTION_INSTANCE_FIELDS})

    def workload_hash(self, *, models=None, seeds=None, teacher_sampling=None,
                      continuation_tokens=None) -> str:
        """Selection instance + everything that changes what the resulting numbers MEAN.

        Continuation length is separate from `chunk_tokens` on purpose. The config currently passes
        one value as both (`omniopd.C` -> `generation_tokens`), and OmniOPD's Appendix B is
        self-contradictory about whether they should be equal (deviation D3, unresolved). Conflating
        them in a hash would make an unresolved ambiguity invisible.
        """
        return self._h({
            "selection_instance": self.selection_instance_hash(),
            "N": self.N,
            "models": models or {},
            "seeds": seeds or {},
            "teacher_sampling": teacher_sampling or {},
            "continuation_tokens": (self.chunk_tokens if continuation_tokens is None
                                    else continuation_tokens),
        })

    # -- scheduling facts ----------------------------------------------------------------------

    @property
    def enables_same_trajectory_overlap(self) -> bool:
        """True only if a chunk can be scored while its own trajectory is still generating.

        This is the property the whole progressive-execution design rests on. `post_trajectory`
        reproduces OmniOPD faithfully but forecloses same-trajectory overlap: the score for chunk 1
        is unknown until EOS. Cross-trajectory pipelining survives either way.
        """
        return self.availability in ("per_token_online", "per_chunk_online")

    @property
    def needs_teacher_before_selection(self) -> bool:
        return self.availability == "post_teacher_scoring"

    def to_manifest(self) -> dict:
        d = asdict(self)
        d.update(ranker_hash=self.ranker_hash(),
                 selection_instance_hash=self.selection_instance_hash(),
                 enables_same_trajectory_overlap=self.enables_same_trajectory_overlap,
                 needs_teacher_before_selection=self.needs_teacher_before_selection,
                 ranker_fields=list(RANKER_FIELDS),
                 selection_instance_fields=list(SELECTION_INSTANCE_FIELDS))
        return d

    @classmethod
    def from_manifest(cls, d: dict) -> "SelectorSpec":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def describe(self) -> str:
        agg = f"anchor@{self.anchor_offset}" if self.aggregation == "anchor" else self.aggregation
        return (f"{self.signal_semantics} via {self.signal_provider} "
                f"[{self.availability}/{self.fidelity}] {agg} C={self.chunk_tokens} M={self.M} "
                f"N={self.N} {self.constraint}/{self.tie_break} "
                f"ranker={self.ranker_hash()} inst={self.selection_instance_hash()}")


def assert_spec_matches(analysis: SelectorSpec, manifest_spec: SelectorSpec, *,
                        allow_cross_policy: bool = False, context: str = "") -> None:
    """FATAL comparison on the SELECTION INSTANCE hash (ranker + M + C).

    Guards the failure that occurred -- analysis re-deriving a selection with aggregation="mean" and
    looking up continuations generated for aggregation="anchor" -- and equally the budget failure:
    chunks generated at M=8 must not be analysed as M=10, which an earlier ranker-only key allowed.

    Cross-policy comparison is a legitimate experiment and stays possible, but must be REQUESTED.
    """
    if analysis.selection_instance_hash() == manifest_spec.selection_instance_hash():
        return
    if allow_cross_policy:
        return
    diffs = {k: (getattr(analysis, k), getattr(manifest_spec, k))
             for k in SELECTION_INSTANCE_FIELDS
             if getattr(analysis, k) != getattr(manifest_spec, k)}
    same_ranker = analysis.ranker_hash() == manifest_spec.ranker_hash()
    hint = ("  The RANKER matches; only the budget differs. That is a legitimate sweep, but it is "
            "not a valid lookup: chunks selected under one budget are a different SET. Use "
            "assert_same_ranker() for the sweep and read each budget from its own manifest.\n"
            if same_ranker else "")
    raise SelectorMismatch(
        f"selector selection-instance mismatch{' in ' + context if context else ''} -- refusing to "
        f"compare scores computed under one policy against chunks generated under another.\n"
        + "\n".join(f"    {k}: analysis={a!r} but manifest={m!r}" for k, (a, m) in diffs.items())
        + f"\n{hint}  analysis: {analysis.describe()}\n  manifest: {manifest_spec.describe()}\n"
        f"  If this IS the experiment, pass allow_cross_policy=True explicitly and say so in the "
        f"write-up.")


def assert_same_ranker(a: SelectorSpec, b: SelectorSpec, *, context: str = "") -> None:
    """For controlled budget sweeps: the ranking rule must be identical while M or C differ."""
    if a.ranker_hash() == b.ranker_hash():
        return
    diffs = {k: (getattr(a, k), getattr(b, k)) for k in RANKER_FIELDS
             if getattr(a, k) != getattr(b, k)}
    raise SelectorMismatch(
        f"ranker mismatch{' in ' + context if context else ''} -- a budget sweep must hold the "
        f"ranking rule fixed, otherwise it measures two things at once.\n"
        + "\n".join(f"    {k}: {x!r} vs {y!r}" for k, (x, y) in diffs.items()))


def require_signal(values, spec: SelectorSpec, *, context: str = ""):
    """Return `values`, or raise. NEVER substitutes a placeholder.

    The pipeline today contains a fallback that supplies uniform ones when a signal is missing, after
    which selection is arbitrary but retained-value metrics still report as though it were real.
    A selector with no signal has exactly two honest outcomes: DEFER, or fail.
    """
    if values is None:
        raise SignalNotReady(
            f"{spec.signal_semantics} from {spec.signal_provider} is unavailable"
            f"{' in ' + context if context else ''} (availability={spec.availability}). Return DEFER "
            f"and retry when the signal lands. Do not substitute uniform or placeholder values: the "
            f"ranking would be arbitrary while every downstream metric still reported it as real.")
    return values


# ---------------------------------------------------------------------------------------------
# Reference executor. Driven entirely by the spec, so aggregation and tie-break are PARAMETERS
# rather than properties of a call site. `select_chunks` in omniopd_chunks.py is the in-trainer twin
# and is pinned to this for the anchor case by scripts/test_selector_spec.py.
# ---------------------------------------------------------------------------------------------

def candidate_scores(signal, prompt_len, resp_len, spec):
    """Per-candidate score, one entry per eligible chunk start t.

    `signal` is indexed over the WHOLE sequence. Candidates whose window would run past the response
    are dropped: a truncated audit window is a different measurement, not a smaller one.
    """
    signal = require_signal(signal, spec, context="candidate_scores")
    C, out = spec.chunk_tokens, []
    for t in range(0, resp_len - C + 1, spec.candidate_stride):
        if spec.aggregation == "anchor":
            i = prompt_len + t + spec.anchor_offset
            if 0 <= i < len(signal):
                out.append((t, signal[i]))
            continue
        w = signal[prompt_len + t: prompt_len + t + C]
        if len(w) < C:
            continue
        out.append((t, sum(w) / C if spec.aggregation == "mean"
                    else sum(w) if spec.aggregation == "sum" else max(w)))
    return out


def apply_constraint(scored, spec):
    """Resolve overlapping candidates into at most M non-overlapping starts, per spec.tie_break."""
    C = spec.chunk_tokens
    sign = -1 if spec.tie_break == "larger_t" else 1
    order = sorted(scored, key=lambda st: (-st[1], sign * st[0]))
    if spec.constraint == "none":
        return sorted(t for t, _ in order[:spec.M])
    if spec.constraint == "greedy_resample_nonoverlap":
        chosen = []
        for t, _ in order:
            if any(abs(t - u) < C for u in chosen):
                continue
            chosen.append(t)
            if len(chosen) == spec.M:
                break
        return sorted(chosen)
    if spec.constraint == "dp_optimal_nonoverlap":
        w = dict(scored)
        ts = sorted(w)
        n = len(ts)
        nxt, j = [], 0
        for i in range(n):
            j = max(j, i + 1)
            while j < n and ts[j] - ts[i] < C:
                j += 1
            nxt.append(j)
        best = [[0.0] * (spec.M + 1) for _ in range(n + 1)]
        take = [[False] * (spec.M + 1) for _ in range(n + 1)]
        for i in range(n - 1, -1, -1):
            for k in range(1, spec.M + 1):
                got = w[ts[i]] + best[nxt[i]][k - 1]
                if got > best[i + 1][k]:
                    best[i][k], take[i][k] = got, True
                else:
                    best[i][k] = best[i + 1][k]
        sel, i, k = [], 0, spec.M
        while i < n and k > 0:
            if take[i][k]:
                sel.append(ts[i]); i, k = nxt[i], k - 1
            else:
                i += 1
        return sorted(sel)
    raise ValueError(f"unhandled constraint {spec.constraint!r}")


def select(signal, prompt_len, resp_len, spec):
    """Execute a SelectorSpec. The only supported way to turn a per-token signal into chunk starts.

    Returns DEFER when the signal is not yet available AND the policy permits waiting -- i.e. the
    signal is expected later rather than absent. Anything else raises.
    """
    if signal is None:
        if spec.availability == "static":
            raise SignalNotReady("a static-availability signal cannot be missing")
        return DEFER
    return apply_constraint(candidate_scores(signal, prompt_len, resp_len, spec), spec)


def manifest_block(specs_by_arm, *, models=None, seeds=None, teacher_sampling=None,
                   continuation_tokens=None, extra=None):
    """The selector half of a generation manifest: one full spec per arm plus all three hashes."""
    wl = {name: spec.workload_hash(models=models, seeds=seeds,
                                   teacher_sampling=teacher_sampling,
                                   continuation_tokens=continuation_tokens)
          for name, spec in specs_by_arm.items()}
    block = {
        "schema_version": 2,
        "arms": {name: spec.to_manifest() for name, spec in specs_by_arm.items()},
        "ranker_hashes": {n: s.ranker_hash() for n, s in specs_by_arm.items()},
        "selection_instance_hashes": {n: s.selection_instance_hash() for n, s in specs_by_arm.items()},
        "workload_hashes": wl,
        "models": models or {}, "seeds": seeds or {},
        "teacher_sampling": teacher_sampling or {},
        "continuation_tokens": continuation_tokens,
        "distinct_rankers": sorted({s.ranker_hash() for s in specs_by_arm.values()}),
        "distinct_selection_instances": sorted({s.selection_instance_hash()
                                                for s in specs_by_arm.values()}),
        "any_enables_same_trajectory_overlap": any(
            s.enables_same_trajectory_overlap for s in specs_by_arm.values()),
        "any_needs_teacher_before_selection": any(
            s.needs_teacher_before_selection for s in specs_by_arm.values()),
    }
    if extra:
        block.update(extra)
    return block


def spec_from_manifest_arm(manifest, arm):
    """Recover one arm's spec, raising rather than defaulting if absent.

    A manifest written before this schema has no recoverable policy, and guessing one is how the
    original mismatch survived. Refuse instead.
    """
    sel = (manifest or {}).get("selectors")
    if not sel or "arms" not in sel:
        raise SelectorMismatch(
            "manifest has no `selectors` block -- its selection policy is unrecoverable, so no "
            "analysis may claim to reproduce it. Regenerate the plan with manifest_block().")
    if arm not in sel["arms"]:
        raise SelectorMismatch(f"arm {arm!r} not in manifest; have {sorted(sel['arms'])}")
    return SelectorSpec.from_manifest(sel["arms"][arm])


# ---------------------------------------------------------------------------------------------
# Reference specs. The first two share `signal_semantics` and differ only in provider -- which is
# precisely the distinction a name-derived producer could not express, and the one that decides
# whether same-trajectory overlap is possible.
# ---------------------------------------------------------------------------------------------

OMNIOPD_PUBLISHED = SelectorSpec(
    signal_semantics="student_entropy", signal_provider="post_eos_actor_forward",
    aggregation="anchor", anchor_offset=-1, chunk_tokens=50, M=10, N=10,
    notes="OmniOPD 3.2.3 Eq 6-7, faithful reference. Exact, but post-EOS: reproduces the algorithm "
          "and forecloses same-trajectory overlap. The Phase-0 baseline.")

OMNIOPD_ONLINE_EXACT = SelectorSpec(
    signal_semantics="student_entropy", signal_provider="rollout_exact_scalar",
    aggregation="anchor", anchor_offset=-1, chunk_tokens=50, M=10, N=10,
    notes="Same algorithm, same fidelity, scored per token during decode. The desired OPDFlow path: "
          "identical selection to OMNIOPD_PUBLISHED, available early enough to overlap. Feasibility "
          "depends on a GPU-side scalar reduction in the vLLM sampler -- UNVERIFIED.")

OMNIOPD_ONLINE_TOPK = SelectorSpec(
    signal_semantics="student_entropy", signal_provider="rollout_topk_approx",
    aggregation="anchor", anchor_offset=-1, chunk_tokens=50, M=10, N=10,
    notes="Optional fallback and an explicit DEVIATION: omniopd_chunks.py:120 currently refuses "
          "entropy_topk != 0 in favour of exact full-vocab (FID-1). Only on measured evidence that "
          "selection is unchanged.")

GAPSELECT_M10 = SelectorSpec(
    signal_semantics="teacher_gap_abs", signal_provider="teacher_scoring_k0",
    aggregation="mean", chunk_tokens=50, M=10, N=10,
    notes="Teacher-side plugin and scheduler stress workload: the one policy whose signal does not "
          "exist until after teacher scoring, so it exercises the deferred-selection path. NOT an "
          "algorithmic improvement claim.")

GAPSELECT_M8_CLOSED = SelectorSpec(
    signal_semantics="teacher_gap_abs", signal_provider="teacher_scoring_k0",
    aggregation="mean", chunk_tokens=50, M=8, N=10,
    notes="CLOSED. Failed both pre-registered gates against OMNIOPD_PUBLISHED "
          "(docs/genopdflow_phase2_gapselect.md). Retained so the closed claim stays citable.")
