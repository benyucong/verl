"""Explicit selection policy for OPDFlow. Mechanism stays in the runtime; policy lives here.

WHY THIS EXISTS. `score(chunk)` is too ambiguous to be a contract. Offline analysis ranked candidate
chunks by the CHUNK MEAN of a per-token signal while the generation pipeline ranked by the ANCHOR
TOKEN, and nothing detected the mismatch because both were just "the score". The two rankings are
near-disjoint -- measured Jaccard 0.013-0.027, median 0.000 -- so an entire validation compared a
chunk-mean proxy against an anchor-ranked generation workload, and its conclusion was invalid.

THREE THINGS THIS MODULE INSISTS ON, each because the obvious shortcut is unsafe:

1. FOUR LEVELS OF IDENTITY, SEPARATING MATHEMATICS FROM IMPLEMENTATION. Two exact implementations of
   the same selector are the SAME SELECTOR; where the numbers came from is provenance, not identity.
       score_rule_hash      the mathematics: signal semantics, WHICH DISTRIBUTION, aggregation,
                            anchor offset, tie-break.
       selector_hash        score rule + candidate construction + C + constraint + M.
                            THIS is the generation-vs-analysis check.
       workload_hash        selector + N + continuation length + models + sampling + seeds --
                            everything that changes what the numbers MEAN.
       provider_fingerprint post-EOS forward vs online vLLM, dtype, engine version, code hash.
                            RECORDED, never part of selector identity.

   A single hash forces a bad choice: include the budget and legitimate M=8-vs-M=10 sweeps trip the
   guard; exclude it and data generated at M=8 can be analysed as M=10 without tripping anything.
   Hence score_rule (for sweeps) and selector (for lookups) as separate levels.

   Because the provider is NOT in the selector hash, two providers claiming exactness must be held to
   it: `assert_provider_equivalence` demands EXACT final-set equality. A failure means the online
   implementation is NOT YET FAITHFUL -- it does not silently become a different intended algorithm.

2. WHICH DISTRIBUTION, STATED. Entropy over the raw model distribution and over the behavior
   distribution (after temperature / top-k / top-p) COINCIDE at temperature=1, top_p=1, top_k
   disabled -- the current setup -- and diverge everywhere else. Leaving it implicit means a later
   sampling change silently redefines the selector while every name stays the same.

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

# NOTE THE PRECISE CLAIM. "exact_full_support" means the reduction runs over the FULL, unpadded
# vocabulary with no truncation. It does NOT mean exact arithmetic, and it does NOT mean bit-identical
# across implementations: vLLM reduces bf16 logits in fp32 after a TP all-gather, while the offline
# oracle applies an fp32 lm_head to hidden states. Both are full-support; neither is the other's
# ground truth to the last bit. That is exactly why provider equivalence is MEASURED on final
# selections rather than asserted from the fidelity label.
FIDELITY = (
    "exact_full_support",       # full vocabulary, no truncation -- see the note above
    "approx_topk",              # truncated support plus a tail bucket
    "exact_on_realised_token",  # full precision, but only at the token actually sampled
    "none",
)

# Registry of PROVIDERS -- the concrete mechanism that computes a signal. Declaring a provider that
# is not registered raises, because an unregistered provider has no known availability, and the
# scheduler would have to guess exactly where guessing has already cost this project a result.
PROVIDERS = {
    # student_entropy, four genuinely different mechanisms:
    "rollout_exact_scalar":    dict(availability="per_token_online",     fidelity="exact_full_support"),
    "rollout_topk_approx":     dict(availability="per_token_online",     fidelity="approx_topk"),
    "post_eos_actor_forward":  dict(availability="post_trajectory",      fidelity="exact_full_support"),
    "student_scoring_worker":  dict(availability="post_trajectory",      fidelity="exact_full_support"),
    "trainer_forward":         dict(availability="post_teacher_scoring", fidelity="exact_full_support"),
    # teacher-side:
    "teacher_scoring_k0":      dict(availability="post_teacher_scoring", fidelity="exact_on_realised_token"),
    "teacher_scoring_topk":    dict(availability="post_teacher_scoring", fidelity="approx_topk"),
    # signal-free:
    "none":                    dict(availability="static",               fidelity="none"),
    "external":                dict(availability="post_trajectory",      fidelity="exact_full_support"),
}

# Which distribution the signal is taken over. These coincide only at temperature=1 with top-p and
# top-k disabled; vLLM applies temperature at sampler.py:177 and truncation at
# topk_topp_sampler.py:104, so a "behavior" signal read after those points is a different quantity
# from the raw one under any other sampling config.
DISTRIBUTIONS = ("raw_model", "behavior")

AGGREGATIONS = ("anchor", "mean", "sum", "max")
CANDIDATE_POLICIES = ("all_starts", "stride", "disjoint_grid")
CONSTRAINTS = ("greedy_resample_nonoverlap", "dp_optimal_nonoverlap", "none")
TIE_BREAKS = ("larger_t", "smaller_t")
RANKING_STATES = ("finalized", "provisional")

# THE MATHEMATICS. No provider, no budget: two exact implementations of this rule are the same rule.
SCORE_RULE_FIELDS = ("signal_semantics", "distribution", "aggregation", "anchor_offset", "tie_break")
# THE SELECTOR. Adds candidate construction and the budget, which together fix WHICH chunks come out.
SELECTOR_FIELDS = SCORE_RULE_FIELDS + ("candidate_policy", "candidate_stride", "constraint",
                                       "chunk_tokens", "M")


class SelectorMismatch(RuntimeError):
    """Raised when analysis would compare scores it computed against chunks generated another way."""


class ProviderNotCertified(SelectorMismatch):
    """Raised when two artifacts share a selector but were produced by different, uncertified providers.

    Sharing `selector_hash` means "these claim to be the same logical selector" -- a CLAIM, not a
    finding. Without this, approximate top-k entropy would pass the selector-hash guard and be
    silently interchanged with exact entropy, which is the precise failure the four-level split was
    introduced to expose rather than hide.
    """


@dataclass(frozen=True)
class EquivalenceScope:
    """WHAT a certificate was earned on. Exact equality on one dataset is not equality forever.

    Two providers can agree on 48 short thinking-OFF responses and diverge on longer ones, on a
    different checkpoint, on a different dtype, or after an engine upgrade. A certificate that did not
    name its conditions would silently outlive them -- so every field here participates in matching,
    and new trajectories require a new certificate.
    """

    model_snapshot: str = ""      # the exact student checkpoint the test ran against
    trajectory_hash: str = ""     # hash of the token ids tested; NEW TRAJECTORIES => NEW CERTIFICATE
    engine_version: str = ""      # 0.15.1 is not 0.20.1
    dtype: str = ""               # bf16 logits / fp32 reduction is not fp32 throughout
    config_hash: str = ""         # sampling + engine config that could change the distribution

    def key(self):
        return (self.model_snapshot, self.trajectory_hash, self.engine_version,
                self.dtype, self.config_hash)

    def describe(self) -> str:
        parts = [f"{k}={v}" for k, v in
                 (("model", self.model_snapshot), ("traj", self.trajectory_hash),
                  ("engine", self.engine_version), ("dtype", self.dtype),
                  ("cfg", self.config_hash)) if v]
        return " ".join(parts) or "<unscoped>"


@dataclass(frozen=True)
class ProviderEquivalenceCertificate:
    """Evidence that two providers were TESTED and produced identical final selections.

    Issued only by `assert_provider_equivalence` on exact set equality. Scoped three ways, because
    equivalence is a measurement and measurements have conditions:
      - to one `selector_hash` -- an approximation can agree at M=10 and diverge at M=3;
      - to one ordered pair of fingerprints;
      - to one `EquivalenceScope` -- model, trajectories, engine version, dtype, config.
    """

    selector_hash: str
    provider_a: str
    provider_b: str
    n_responses: int
    n_chunks: int
    scope: EquivalenceScope = field(default_factory=EquivalenceScope)
    evidence: str = ""            # run id, artifact path, or commit -- how to re-check this

    def covers(self, selector_hash: str, fp_a: str, fp_b: str, scope: EquivalenceScope) -> bool:
        return (self.selector_hash == selector_hash
                and {self.provider_a, self.provider_b} == {fp_a, fp_b}
                and self.scope.key() == scope.key())

    def describe(self) -> str:
        return (f"certificate[{self.selector_hash}] {self.provider_a} == {self.provider_b} "
                f"({self.n_responses} responses, {self.n_chunks} chunks; {self.scope.describe()}"
                f"{'; ' + self.evidence if self.evidence else ''})")


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
class ProviderFingerprint:
    """WHERE the numbers came from. Provenance, never identity.

    Two providers claiming `fidelity="exact"` are asserting they compute the SAME selector. That
    assertion is checked by `assert_provider_equivalence` (exact final-set equality), not by being
    folded into the selector hash -- folding it in would declare them different algorithms, which is
    the opposite of what is meant, and would make faithfulness unfalsifiable.
    """

    signal_provider: str
    availability: str = ""
    fidelity: str = ""
    dtype: str = ""              # e.g. "bf16 logits / fp32 reduction"
    engine: str = ""             # e.g. "vllm"
    engine_version: str = ""     # PIN IT. 0.15.1 on acc is not 0.20.1 on LUMI until tested.
    code_hash: str = ""          # patch or source hash of the computing code

    def __post_init__(self):
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
                    f"{f}={getattr(self, f)!r} contradicts provider {self.signal_provider!r}, "
                    f"registered as {known[f]!r}. Fix the spec or register a new provider.")

    @property
    def enables_same_trajectory_overlap(self) -> bool:
        """True only if a chunk can be scored while its own trajectory is still generating.

        The entire progressive-execution design rests on this. `post_trajectory` reproduces OmniOPD
        faithfully but forecloses same-trajectory overlap; cross-trajectory pipelining survives.
        """
        return self.availability in ("per_token_online", "per_chunk_online")

    @property
    def needs_teacher_before_selection(self) -> bool:
        return self.availability == "post_teacher_scoring"

    def fingerprint(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True,
                                         separators=(",", ":")).encode()).hexdigest()[:16]

    def describe(self) -> str:
        v = f"{self.engine}{'@' + self.engine_version if self.engine_version else ''}"
        return (f"{self.signal_provider}[{self.availability}/{self.fidelity}]"
                f"{' ' + v if v else ''}{' ' + self.dtype if self.dtype else ''}")


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
    # RAW MODEL or BEHAVIOR distribution. They coincide only at temperature=1 / top_p=1 / top_k off.
    # Stating it prevents a later sampling change from silently redefining the selector.
    distribution: str = "raw_model"
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
    # Provenance. Optional so a spec can name the MATHEMATICS alone; required before any run.
    provider: "ProviderFingerprint | None" = None
    notes: str = ""                             # free text; never part of any hash

    def __post_init__(self):
        if self.signal_semantics not in SIGNAL_SEMANTICS:
            raise ValueError(f"unknown signal_semantics {self.signal_semantics!r}")
        for name, allowed in (("distribution", DISTRIBUTIONS),
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

    def score_rule_hash(self) -> str:
        """The MATHEMATICS: signal semantics, distribution, aggregation, anchor, tie-break.

        Deliberately provider-free and budget-free. Two exact implementations of one rule share this,
        which is what makes "is the online provider faithful?" a falsifiable question rather than a
        definitional one.
        """
        return self._h({k: getattr(self, k) for k in SCORE_RULE_FIELDS})

    def selector_hash(self) -> str:
        """Score rule + candidate construction + C + constraint + M.

        **The generation-vs-analysis check.** M and C are the audit budget: a set selected under one
        budget is not a set selected under another, so reading M=8 data as M=10 must fail here.
        """
        return self._h({k: getattr(self, k) for k in SELECTOR_FIELDS})

    def provider_fingerprint(self) -> str:
        return self.provider.fingerprint() if self.provider else ""

    def workload_hash(self, *, models=None, seeds=None, teacher_sampling=None,
                      continuation_tokens=None) -> str:
        """Selection instance + everything that changes what the resulting numbers MEAN.

        Continuation length is separate from `chunk_tokens` on purpose. The config currently passes
        one value as both (`omniopd.C` -> `generation_tokens`), and OmniOPD's Appendix B is
        self-contradictory about whether they should be equal (deviation D3, unresolved). Conflating
        them in a hash would make an unresolved ambiguity invisible.
        """
        return self._h({
            "selector": self.selector_hash(),
            "N": self.N,
            "models": models or {},
            "seeds": seeds or {},
            "teacher_sampling": teacher_sampling or {},
            "continuation_tokens": (self.chunk_tokens if continuation_tokens is None
                                    else continuation_tokens),
        })

    # -- scheduling facts ----------------------------------------------------------------------

    # Scheduling facts are properties of the PROVIDER, not of the mathematics.
    @property
    def enables_same_trajectory_overlap(self) -> bool:
        return bool(self.provider) and self.provider.enables_same_trajectory_overlap

    @property
    def needs_teacher_before_selection(self) -> bool:
        return bool(self.provider) and self.provider.needs_teacher_before_selection

    def with_provider(self, provider: "ProviderFingerprint") -> "SelectorSpec":
        """Same mathematics, different implementation. score_rule_hash and selector_hash are
        unchanged by construction -- which is the whole point of the split."""
        return SelectorSpec(**{**asdict(self), "provider": provider,
                               "notes": self.notes})

    def to_manifest(self) -> dict:
        d = asdict(self)
        d.update(score_rule_hash=self.score_rule_hash(),
                 selector_hash=self.selector_hash(),
                 provider_fingerprint=self.provider_fingerprint(),
                 enables_same_trajectory_overlap=self.enables_same_trajectory_overlap,
                 needs_teacher_before_selection=self.needs_teacher_before_selection,
                 score_rule_fields=list(SCORE_RULE_FIELDS),
                 selector_fields=list(SELECTOR_FIELDS))
        return d

    @classmethod
    def from_manifest(cls, d: dict) -> "SelectorSpec":
        kw = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        prov = kw.get("provider")
        if isinstance(prov, dict):
            kw["provider"] = ProviderFingerprint(
                **{k: v for k, v in prov.items() if k in ProviderFingerprint.__dataclass_fields__})
        return cls(**kw)

    def describe(self) -> str:
        agg = f"anchor@{self.anchor_offset}" if self.aggregation == "anchor" else self.aggregation
        prov = f" via {self.provider.describe()}" if self.provider else " (no provider)"
        return (f"{self.signal_semantics}/{self.distribution}{prov} {agg} C={self.chunk_tokens} "
                f"M={self.M} N={self.N} {self.constraint}/{self.tie_break} "
                f"rule={self.score_rule_hash()} sel={self.selector_hash()}")


def assert_spec_matches(analysis: SelectorSpec, manifest_spec: SelectorSpec, *,
                        certificates=(), scope: "EquivalenceScope | None" = None,
                        allow_cross_policy: bool = False, context: str = "") -> None:
    """FATAL comparison on the SELECTOR hash (score rule + candidates + C + constraint + M).

    Guards the failure that occurred -- analysis re-deriving a selection with aggregation="mean" and
    looking up continuations generated for aggregation="anchor" -- and equally the budget failure:
    chunks generated at M=8 must not be analysed as M=10, which an earlier ranker-only key allowed.

    Cross-policy comparison is a legitimate experiment and stays possible, but must be REQUESTED.
    """
    if analysis.selector_hash() == manifest_spec.selector_hash():
        # Same LOGICAL selector. Artifacts are interchangeable only if they came from the same
        # provider, or if a certificate records that the two providers were tested equal. A shared
        # selector_hash is a claim of equivalence, never a demonstration of it.
        fa, fb = analysis.provider_fingerprint(), manifest_spec.provider_fingerprint()
        if fa == fb:
            return
        sh = analysis.selector_hash()
        if certificates and scope is None:
            raise ProviderNotCertified(
                f"certificates were supplied but no EquivalenceScope was given"
                f"{' in ' + context if context else ''}. A certificate is valid only for the model, "
                f"trajectories, engine version, dtype and config it was earned on; without the "
                f"current scope it cannot be checked, and accepting it unchecked is how a "
                f"certificate silently outlives its conditions.")
        if any(c.covers(sh, fa, fb, scope) for c in certificates):
            return
        near = [c for c in certificates if c.covers(sh, fa, fb, c.scope)]
        stale = ("\n  A certificate exists for this selector and provider pair but for a DIFFERENT "
                 f"scope:\n    have: {near[0].scope.describe()}\n    need: {scope.describe()}\n"
                 "  Re-run the equivalence test on these artifacts.\n" if (near and scope) else "")
        raise ProviderNotCertified(
            f"same logical selector, DIFFERENT providers, no equivalence certificate"
            f"{' in ' + context if context else ''}.\n"
            f"  analysis: {analysis.provider.describe() if analysis.provider else '<none>'}\n"
            f"  manifest: {manifest_spec.provider.describe() if manifest_spec.provider else '<none>'}\n"
            f"  selector: {sh}\n"
            f"  A shared selector_hash means these CLAIM to compute the same selector; it is not "
            f"evidence that they do. Until an equivalence test passes, the second is a CANDIDATE "
            f"IMPLEMENTATION of the same logical selector, not a faithful one -- and an approximate "
            f"provider (e.g. top-k entropy) would otherwise pass this guard and be mistaken for "
            f"exact.\n"
            f"{stale}"
            f"  Run assert_provider_equivalence() on real selections and pass the certificate it "
            f"returns, or use the same provider for both artifacts.")
    if allow_cross_policy:
        return
    diffs = {k: (getattr(analysis, k), getattr(manifest_spec, k))
             for k in SELECTOR_FIELDS
             if getattr(analysis, k) != getattr(manifest_spec, k)}
    same_rule = analysis.score_rule_hash() == manifest_spec.score_rule_hash()
    hint = ("  The SCORE RULE matches; only candidate construction or the budget differs. That is a "
            "legitimate sweep, but it is not a valid lookup: chunks selected under one budget are a "
            "different SET. Use assert_same_score_rule() for the sweep and read each budget from its "
            "own manifest.\n" if same_rule else "")
    raise SelectorMismatch(
        f"selector mismatch{' in ' + context if context else ''} -- refusing to compare scores "
        f"computed under one policy against chunks generated under another.\n"
        + "\n".join(f"    {k}: analysis={a!r} but manifest={m!r}" for k, (a, m) in diffs.items())
        + f"\n{hint}  analysis: {analysis.describe()}\n  manifest: {manifest_spec.describe()}\n"
        f"  If this IS the experiment, pass allow_cross_policy=True explicitly and say so in the "
        f"write-up.")


def assert_same_score_rule(a: SelectorSpec, b: SelectorSpec, *, context: str = "") -> None:
    """For controlled budget sweeps: the mathematics must be identical while M or C differ."""
    if a.score_rule_hash() == b.score_rule_hash():
        return
    diffs = {k: (getattr(a, k), getattr(b, k)) for k in SCORE_RULE_FIELDS
             if getattr(a, k) != getattr(b, k)}
    raise SelectorMismatch(
        f"score-rule mismatch{' in ' + context if context else ''} -- a budget sweep must hold the "
        f"mathematics fixed, otherwise it measures two things at once.\n"
        + "\n".join(f"    {k}: {x!r} vs {y!r}" for k, (x, y) in diffs.items()))


def assert_provider_equivalence(sel_a, sel_b, spec_a: SelectorSpec, spec_b: SelectorSpec, *,
                                scope: "EquivalenceScope | None" = None,
                                context: str = "") -> "ProviderEquivalenceCertificate":
    """Two providers claiming exactness must produce the SAME FINAL SET. Exact equality, not Jaccard.

    This is the check the four-level split buys. Because the provider is NOT in `selector_hash`,
    `post_eos_actor_forward` and `rollout_exact_scalar` are the same selector by construction -- so
    the claim "the online path is faithful" becomes falsifiable, and this is where it is falsified.

    A failure means THE ONLINE IMPLEMENTATION IS NOT YET FAITHFUL. It does not mean the online path
    is a different intended algorithm, and it must never be written up that way: near-identical is
    not identical, and this project has already been burned treating a 0.97 Jaccard as agreement.

    `sel_a`/`sel_b` are per-response selections: {response_id: set(chunk_starts)}.
    Returns a `ProviderEquivalenceCertificate` on success, which `assert_spec_matches` accepts as
    licence to interchange artifacts from the two providers.
    """
    if spec_a.selector_hash() != spec_b.selector_hash():
        raise SelectorMismatch(
            f"provider equivalence is only meaningful between the SAME selector; got "
            f"{spec_a.selector_hash()} vs {spec_b.selector_hash()}")
    keys = set(sel_a) | set(sel_b)
    bad = {k: (sorted(sel_a.get(k, ())), sorted(sel_b.get(k, ()))) for k in sorted(keys)
           if set(sel_a.get(k, ())) != set(sel_b.get(k, ()))}
    if not bad:
        return ProviderEquivalenceCertificate(
            selector_hash=spec_a.selector_hash(),
            provider_a=spec_a.provider_fingerprint(), provider_b=spec_b.provider_fingerprint(),
            n_responses=len(keys), n_chunks=sum(len(sel_a.get(k, ())) for k in keys),
            scope=scope or EquivalenceScope(), evidence=context)
    inter = sum(len(set(sel_a.get(k, ())) & set(sel_b.get(k, ()))) for k in keys)
    union = sum(len(set(sel_a.get(k, ())) | set(sel_b.get(k, ()))) for k in keys)
    shown = "\n".join(f"    {k}: {a} vs {b}" for k, (a, b) in list(bad.items())[:5])
    raise SelectorMismatch(
        f"provider equivalence FAILED{' in ' + context if context else ''}: "
        f"{len(bad)}/{len(keys)} responses select different chunk sets.\n{shown}\n"
        f"    (Jaccard {inter / max(union, 1):.4f} -- reported for context only; the bar is EXACT "
        f"set equality, and a high Jaccard with frequent set inequality is precisely the failure "
        f"this check exists to catch.)\n"
        f"  a: {spec_a.provider.describe() if spec_a.provider else '?'}\n"
        f"  b: {spec_b.provider.describe() if spec_b.provider else '?'}\n"
        f"  This means the online implementation is NOT YET FAITHFUL. It is not a different "
        f"intended algorithm and must not be described as one.")


def require_signal(values, spec: SelectorSpec, *, context: str = ""):
    """Return `values`, or raise. NEVER substitutes a placeholder.

    The pipeline today contains a fallback that supplies uniform ones when a signal is missing, after
    which selection is arbitrary but retained-value metrics still report as though it were real.
    A selector with no signal has exactly two honest outcomes: DEFER, or fail.
    """
    if values is None:
        prov = spec.provider.signal_provider if spec.provider else "<no provider declared>"
        avail = spec.provider.availability if spec.provider else "unknown"
        raise SignalNotReady(
            f"{spec.signal_semantics} from {prov} is unavailable"
            f"{' in ' + context if context else ''} (availability={avail}). Return DEFER "
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
        if spec.provider and spec.provider.availability == "static":
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
        "schema_version": 3,
        "arms": {name: spec.to_manifest() for name, spec in specs_by_arm.items()},
        "score_rule_hashes": {n: s.score_rule_hash() for n, s in specs_by_arm.items()},
        "selector_hashes": {n: s.selector_hash() for n, s in specs_by_arm.items()},
        "provider_fingerprints": {n: s.provider_fingerprint() for n, s in specs_by_arm.items()},
        "workload_hashes": wl,
        "models": models or {}, "seeds": seeds or {},
        "teacher_sampling": teacher_sampling or {},
        "continuation_tokens": continuation_tokens,
        "distinct_score_rules": sorted({s.score_rule_hash() for s in specs_by_arm.values()}),
        "distinct_selectors": sorted({s.selector_hash() for s in specs_by_arm.values()}),
        "distinct_providers": sorted({s.provider_fingerprint() for s in specs_by_arm.values()}),
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

# The two exact providers share score_rule_hash AND selector_hash by construction. That identity is
# the point: it makes "is the online path faithful?" falsifiable via assert_provider_equivalence,
# rather than true by definition.

_OMNIOPD_RULE = dict(signal_semantics="student_entropy", distribution="raw_model",
                     aggregation="anchor", anchor_offset=-1, chunk_tokens=50, M=10, N=10)

POST_EOS_EXACT = ProviderFingerprint(
    signal_provider="post_eos_actor_forward", dtype="fp32 lm_head / fp32 reduction")

# engine_version is PINNED. 0.15.1 on acc is the first target; 0.20.1 on LUMI is structurally
# similar but UNTESTED, and must not be claimed until that source and runtime are exercised.
ROLLOUT_EXACT_VLLM_0151 = ProviderFingerprint(
    signal_provider="rollout_exact_scalar", dtype="bf16 logits / fp32 reduction",
    engine="vllm", engine_version="0.15.1", code_hash="")

OMNIOPD_PUBLISHED = SelectorSpec(
    **_OMNIOPD_RULE, provider=POST_EOS_EXACT,
    notes="OmniOPD 3.2.3 Eq 6-7, faithful reference. Exact but post-EOS: reproduces the algorithm "
          "and forecloses same-trajectory overlap. The Phase-0 baseline.")

OMNIOPD_ONLINE_EXACT = SelectorSpec(
    **_OMNIOPD_RULE, provider=ROLLOUT_EXACT_VLLM_0151,
    notes="THE SAME LOGICAL SELECTOR, computed during decode -- a CANDIDATE IMPLEMENTATION until an "
          "equivalence certificate exists. Full-support (bf16 logits, fp32 reduction), which is not "
          "the same as bit-identical to the fp32 post-EOS forward, so equivalence must be measured "
          "on final selections rather than inferred from the fidelity label.")

OMNIOPD_ONLINE_TOPK = SelectorSpec(
    **_OMNIOPD_RULE,
    provider=ProviderFingerprint(signal_provider="rollout_topk_approx", engine="vllm",
                                 engine_version="0.15.1"),
    notes="Optional fallback and an explicit DEVIATION: omniopd_chunks.py:120 refuses entropy_topk "
          "!= 0 in favour of exact full-vocab (FID-1). At V=151936, K=20 observes 0.013% of the "
          "support and the unobserved tail is largest exactly at the high-entropy anchors this rule "
          "selects -- so it must be validated by selection agreement, never by entropy MSE.")

GAPSELECT_M10 = SelectorSpec(
    signal_semantics="teacher_gap_abs", distribution="raw_model", aggregation="mean",
    chunk_tokens=50, M=10, N=10,
    provider=ProviderFingerprint(signal_provider="teacher_scoring_k0"),
    notes="Teacher-side plugin and scheduler stress workload: the one policy whose signal does not "
          "exist until after teacher scoring, so it exercises the deferred-selection path. NOT an "
          "algorithmic improvement claim.")

GAPSELECT_M8_CLOSED = SelectorSpec(
    signal_semantics="teacher_gap_abs", distribution="raw_model", aggregation="mean",
    chunk_tokens=50, M=8, N=10,
    provider=ProviderFingerprint(signal_provider="teacher_scoring_k0"),
    notes="CLOSED. Failed both pre-registered gates against OMNIOPD_PUBLISHED "
          "(docs/genopdflow_phase2_gapselect.md). Retained so the closed claim stays citable.")
