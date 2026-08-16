"""Per-token student entropy: the GPU kernel, and the verl-boundary accumulator that consumes it.

WHAT THIS IS FOR. OmniOPD selects audit chunks by peak student entropy. The only exact implementation
today re-forwards the finished response through the actor, so chunk 1's score is unknown until EOS --
which forecloses starting teacher generation before the student finishes. vLLM already holds the
full-vocabulary logits on the GPU at every decode step; this module is the reduction that turns them
into one scalar per emitted token, plus the accumulator that reassembles those scalars per request.

TWO HALVES, DELIBERATELY SPLIT:

  entropy_from_logits    goes INSIDE vLLM. Kept narrow and specific -- one tensor in, one tensor out
                         -- so the patch stays small and reviewable against upstream.
  TokenEntropyStream     lives at the VERL BOUNDARY. Maps vLLM's specific field into OPDFlow's
                         generic student-signal event, and owns every alignment invariant.

The invariants are in the accumulator rather than the kernel because that is where they can actually
be checked: the kernel sees an anonymous [B, V] tensor with no request identity at all.

ALIGNMENT IS THE WHOLE RISK. This project has already shipped four off-by-one defects of exactly this
shape. So: one value per EMITTED TOKEN (never per engine step), keyed by request id (never by batch
row, which continuous batching reorders every step), and `len(entropies) == len(token_ids)` asserted
per request at every append and again at finish.
"""
from __future__ import annotations

from dataclasses import dataclass, field


def entropy_from_logits(logits, *, chunk_rows: int = 0):
    """Exact Shannon entropy in nats, one scalar per row, over the FULL unpadded vocabulary.

        H = logsumexp(z) - sum_v softmax(z)_v * z_v

    This identity rather than the textbook `-(p * log p).sum(-1)`: it needs ONE transient [B, V]
    buffer instead of two, and `torch.compile` fuses it into the softmax epilogue so a compiled call
    allocates no [B, V] at all. vLLM does the same thing for `batched_count_greater_than`, explicitly
    to avoid extra copies. The naive form allocates up to 311 MB at B=256, V=151936, which would
    surface in the sampler's warm-up OOM probe rather than as a clean error.

    Reduction is fp32 regardless of input dtype: bf16 logits carry ~3 decimal digits, and summing
    151936 terms in bf16 loses far more than the entropy differences that decide anchor selection.

    NEVER MUTATES `logits` -- the caller's tensor is the one sampling uses, and a selection signal
    that perturbs sampling would change the rollout it is supposed to be observing.
    """
    import torch

    if logits.ndim != 2:
        raise ValueError(f"expected [num_rows, vocab], got {tuple(logits.shape)}")
    if chunk_rows and logits.shape[0] > chunk_rows:
        # Row-chunked for very large batches; the vocab dim is never split, so each row's entropy is
        # still an exact full-vocabulary reduction.
        return torch.cat([entropy_from_logits(logits[i:i + chunk_rows])
                          for i in range(0, logits.shape[0], chunk_rows)])
    z = logits.to(torch.float32)
    return torch.logsumexp(z, dim=-1) - (z.softmax(dim=-1) * z).sum(dim=-1)


def entropy_reference(logits):
    """Textbook fp32 reference: -(p * log p).sum(-1). Slower, allocates more, used only by tests."""
    import torch

    z = logits.to(torch.float32)
    logp = torch.log_softmax(z, dim=-1)
    return -(logp.exp() * logp).sum(dim=-1)


class EntropyAlignmentError(RuntimeError):
    """Raised when entropy values and token ids fall out of step for a request.

    Always fatal. A dropped, duplicated or shifted entropy value produces a selection that is subtly
    wrong everywhere and obviously wrong nowhere -- the failure mode this project keeps paying for.
    """


@dataclass
class _Req:
    token_ids: list = field(default_factory=list)
    entropies: list = field(default_factory=list)
    finished: bool = False


@dataclass
class TokenEntropyStream:
    """Accumulates per-emitted-token entropy per request, at the verl boundary.

    Keyed by REQUEST ID, never by batch row: continuous batching adds, removes and moves rows every
    engine step, so a row index is meaningful only within the step that produced it.

    Handles multi-token steps (speculative / jump decoding) by requiring the caller to pass the tokens
    and entropies for the step together and in emission order. If a step's lengths disagree, that is
    a bug in the producer and it raises here rather than silently truncating.
    """

    strict: bool = True                  # False only for diagnostics; never in a run that reports numbers
    reqs: dict = field(default_factory=dict)

    def append(self, request_id: str, token_ids, entropies):
        """One engine step's emission for one request. `entropies[i]` belongs to `token_ids[i]`."""
        token_ids, entropies = list(token_ids), list(entropies)
        if len(token_ids) != len(entropies):
            raise EntropyAlignmentError(
                f"{request_id}: step emitted {len(token_ids)} tokens but {len(entropies)} entropy "
                f"values. One value per EMITTED TOKEN is required -- a per-engine-step value silently "
                f"drops the extra tokens of a multi-token step.")
        r = self.reqs.setdefault(request_id, _Req())
        if r.finished:
            raise EntropyAlignmentError(f"{request_id}: append after finish -- duplicated delivery")
        r.token_ids.extend(token_ids)
        r.entropies.extend(entropies)
        if len(r.token_ids) != len(r.entropies):
            raise EntropyAlignmentError(
                f"{request_id}: {len(r.token_ids)} tokens vs {len(r.entropies)} entropies")

    def finish(self, request_id: str, expected_token_ids=None):
        """Close a request and return its entropy series, after checking it against the tokens.

        `expected_token_ids` is the authoritative sequence from the engine's own output. Comparing
        against it catches the case the incremental path cannot see by itself: values that were
        dropped or reordered *before* reaching this accumulator.
        """
        r = self.reqs.get(request_id)
        if r is None:
            raise EntropyAlignmentError(f"{request_id}: finish() before any append()")
        if expected_token_ids is not None:
            exp = list(expected_token_ids)
            if len(exp) != len(r.entropies):
                raise EntropyAlignmentError(
                    f"{request_id}: engine reported {len(exp)} tokens but {len(r.entropies)} "
                    f"entropy values arrived. Values were dropped, duplicated or never sent -- "
                    f"check for a transport hop that zips against a fixed-width iterator.")
            if self.strict and exp != r.token_ids:
                raise EntropyAlignmentError(
                    f"{request_id}: token ids from the incremental stream do not match the engine's "
                    f"final output; the stream is misaligned, not merely incomplete.")
        r.finished = True
        return list(r.entropies)

    def pending(self):
        return [rid for rid, r in self.reqs.items() if not r.finished]


def to_signal_series(entropies, prompt_len: int):
    """Map a response-indexed entropy series into the whole-sequence layout selectors expect.

    `select` reads response position t at `signal[prompt_len + t + anchor_offset]`, with OmniOPD's
    offset of -1 meaning "the distribution that PRODUCED y_t". The sampler's step entropy is exactly
    that distribution, so `entropies[t]` belongs at sequence index `prompt_len + t - 1`.

    Getting this wrong by one is the fourth defect of this shape in this project, which is why the
    mapping lives in one named function with a test pinning it rather than inline at call sites.
    """
    return [float("-inf")] * (prompt_len - 1) + list(entropies)


class SpecDecUnsupported(RuntimeError):
    """Raised at ENGINE INIT when token entropy is requested alongside speculative decoding.

    Phase 0 instruments `Sampler.forward`, which emits exactly one token per request per step.
    Speculative and jump decoding emit several through the REJECTION SAMPLER -- a different code path
    that this patch does not touch. Instrumenting one and not the other would silently drop the extra
    tokens' entropy, leaving a series shorter than the token list, which is precisely the misalignment
    class this project keeps paying for.

    Rejecting up front rather than per step matters: a per-step check burns a branch on the hot path
    and, worse, fails *after* generation has begun, when a partial run already exists to be
    misread. Supporting the rejection-sampler path is a separate milestone, not Phase 0 scope.
    """


def assert_entropy_config_supported(*, return_token_entropy: bool, speculative_config=None,
                                    logprobs_mode: str = "raw_logprobs",
                                    distribution: str = "raw_model") -> dict:
    """Validate at engine construction. Returns manifest facts to record; raises rather than warns.

    Call this ONCE, before the engine starts generating. Everything it checks is a property of the
    configuration, so nothing here needs to be re-checked per step.
    """
    if not return_token_entropy:
        return {"token_entropy": False}
    if speculative_config is not None:
        raise SpecDecUnsupported(
            "return_token_entropy=True is not supported with speculative decoding. Phase 0 "
            "instruments Sampler.forward (one token per request per step); multi-token emission goes "
            "through the rejection sampler, which is uninstrumented, so entropy values would be "
            "silently missing for accepted draft tokens. Disable speculative decoding, or disable "
            "token entropy. Supporting the rejection-sampler path is a separate milestone.")
    if distribution == "raw_model" and logprobs_mode != "raw_logprobs":
        raise ValueError(
            f"distribution='raw_model' requires logprobs_mode='raw_logprobs', got "
            f"{logprobs_mode!r}. Under processed_* modes vLLM substitutes post-temperature and "
            f"post-top-k values, so the emitted quantity would be the BEHAVIOR distribution's entropy "
            f"while every downstream name still said raw_model.")
    return {"token_entropy": True, "speculative_decoding": False,
            "logprobs_mode": logprobs_mode, "distribution": distribution}


def active_row_mask(num_reqs: int, active_indices):
    """Rows of a [num_reqs, ...] batch that belong to a live request this step.

    A finished or unscheduled request still occupies a row until the batch is condensed. Emitting
    entropy for those rows would attach values to requests that generated no token, which the
    accumulator would then see as duplicates.
    """
    m = [False] * num_reqs
    for i in active_indices:
        if not (0 <= i < num_reqs):
            raise EntropyAlignmentError(f"active row {i} outside batch of {num_reqs}")
        m[i] = True
    return m


def emit_for_step(req_ids, sampled_token_ids, entropies, *, active_indices=None,
                  placeholder_token_id: int = -1):
    """Turn one engine step's tensors into per-request (token_ids, entropies) pairs.

    Handles the two ways a `[num_reqs, max_num_generated_tokens]` block lies about its own contents:
    PADDED CELLS (requests that produced fewer tokens than the widest one this step) and INACTIVE
    ROWS (finished or unscheduled requests still occupying a slot). Neither may receive or emit an
    entropy value.

    `sampled_token_ids` is a list of per-request token lists, already trimmed of padding by the
    caller, or padded with `placeholder_token_id`, which is stripped here.
    """
    if not (len(req_ids) == len(sampled_token_ids) == len(entropies)):
        raise EntropyAlignmentError(
            f"step shape mismatch: {len(req_ids)} req_ids, {len(sampled_token_ids)} token rows, "
            f"{len(entropies)} entropy rows")
    mask = (active_row_mask(len(req_ids), active_indices) if active_indices is not None
            else [True] * len(req_ids))
    out = []
    for i, rid in enumerate(req_ids):
        if not mask[i]:
            continue
        toks = [t for t in sampled_token_ids[i] if t != placeholder_token_id]
        ents = list(entropies[i])[:len(toks)]
        if len(ents) != len(toks):
            raise EntropyAlignmentError(
                f"{rid}: {len(toks)} real tokens but {len(ents)} entropy values after stripping "
                f"padding -- the entropy row is shorter than the token row")
        if toks:
            out.append((rid, toks, ents))
    return out


def assert_full_vocab(logits, vocab_size: int):
    """The reduction must see the real vocabulary, not the TP-padded one.

    vLLM pads the vocab dimension for tensor parallelism and slices it back to `org_vocab_size`
    after the all-gather. Reducing over the padded width would fold `-inf` (or worse, garbage)
    columns into the distribution and shift every entropy value by a constant that depends on the
    padding, which is invisible in the output and fatal to selection.
    """
    if logits.shape[-1] != vocab_size:
        raise ValueError(
            f"entropy would be computed over {logits.shape[-1]} columns but the real vocabulary is "
            f"{vocab_size}. Slice padding off before reducing (vLLM does this at "
            f"logits_processor.py:99, `logits[..., :self.org_vocab_size]`).")
    return logits
