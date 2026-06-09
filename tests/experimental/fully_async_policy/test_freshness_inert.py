# Falsification artifact: under the supervised forward-KL top-k distillation loss
# (the default OPD objective; policy-gradient and task-rewards OFF), a chunk's
# `policy_version` / staleness is CAUSALLY INERT -- it never enters the loss value
# or the gradient. Chunk streaming's entire reason to exist (freshness) therefore
# provides ZERO gradient leverage in this objective: a chunk lagged by N policy
# versions trains identically to a fresh one (only the rollouter's drop/staleness
# GATE ever reads policy_version, and that happens before the loss).
#
# This is the cheap (~0 GPU) probe behind the claim that any chunk-streaming
# efficiency win under this loss must come from throughput/waste engineering, not
# freshness. Run inside the training container (needs torch + the verl package):
#   PYTHONPATH=$R/verl:$R:/opt/venv/lib/python3.12/site-packages: \
#     $R/.venv/bin/python verl/tests/experimental/fully_async_policy/test_freshness_inert.py
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from verl.trainer.distillation.fsdp.losses import compute_forward_kl_topk, kl_divergence


def _cfg(log_prob_min_clamp=None):
    # Duck-typed stand-in for DistillationConfig: the loss only reads
    # config.distillation_loss.log_prob_min_clamp.
    return SimpleNamespace(distillation_loss=SimpleNamespace(log_prob_min_clamp=log_prob_min_clamp))


def _teacher_targets(seqlen, vocab, topk, seed):
    g = torch.Generator().manual_seed(seed)
    teacher_logits = torch.randn(seqlen, vocab, generator=g, dtype=torch.float32)
    tlp = F.log_softmax(teacher_logits, dim=-1)
    tk_lp, tk_ids = torch.topk(tlp, k=topk, dim=-1)
    # compute_forward_kl_topk requires nested (jagged) teacher tensors; it does
    # .values().unsqueeze(0) internally -> (1, seqlen, topk).
    nlp = torch.nested.nested_tensor([tk_lp], layout=torch.jagged)
    nids = torch.nested.nested_tensor([tk_ids], layout=torch.jagged)
    return nlp, nids


def _grad_of_chunk(student_logits_base, chunk):
    """Mirror train time: student logits are recomputed (here: a leaf we backprop
    through); the loss reads ONLY the teacher targets off the chunk. `policy_version`
    is present on the chunk (as in the real ChunkSample) but never consulted here --
    exactly as in verl's loss path."""
    s = student_logits_base.clone().requires_grad_(True)
    out = compute_forward_kl_topk(
        student_logits=s,
        teacher_topk_log_probs=chunk["teacher_topk_log_probs"],
        teacher_topk_ids=chunk["teacher_topk_ids"],
        config=_cfg(),
        data_format="bshd",
    )
    out["distillation_losses"].sum().backward()
    return out["distillation_losses"].detach().clone(), s.grad.clone()


def test_kl_divergence_is_pure_function_of_logprobs():
    # The foundational loss takes ONLY (log_q, log_p): there is structurally no
    # argument through which a version/staleness value could enter.
    g = torch.Generator().manual_seed(7)
    lq = F.log_softmax(torch.randn(4, 16, generator=g), dim=-1)
    lp = F.log_softmax(torch.randn(4, 16, generator=g), dim=-1)
    a = kl_divergence(log_q=lq, log_p=lp)
    b = kl_divergence(log_q=lq, log_p=lp)
    assert torch.equal(a, b)
    import inspect

    params = list(inspect.signature(kl_divergence).parameters)
    assert params == ["log_q", "log_p"], params


def test_policy_version_does_not_change_loss_or_gradient():
    seqlen, vocab, topk = 12, 256, 16
    nlp, nids = _teacher_targets(seqlen, vocab, topk, seed=0)
    base = torch.randn(1, seqlen, vocab, dtype=torch.float32, generator=torch.Generator().manual_seed(3))

    # Two chunks: byte-identical token content + teacher targets, differing ONLY in
    # policy_version (a 99-version lag -- well beyond any sigma). If freshness mattered
    # to the gradient, these would differ.
    fresh = {"policy_version": 100, "teacher_topk_log_probs": nlp, "teacher_topk_ids": nids}
    stale = {"policy_version": 1, "teacher_topk_log_probs": nlp, "teacher_topk_ids": nids}

    loss_fresh, grad_fresh = _grad_of_chunk(base, fresh)
    loss_stale, grad_stale = _grad_of_chunk(base, stale)

    assert torch.equal(loss_fresh, loss_stale), "policy_version changed the LOSS -> not inert"
    assert torch.equal(grad_fresh, grad_stale), "policy_version changed the GRADIENT -> not inert"


def test_no_version_reference_in_loss_modules_but_present_in_staleness_gate():
    # Source-level proof there is no code path from policy_version to the loss.
    import verl.trainer.distillation.fsdp.losses as fsdp_losses
    import verl.trainer.distillation.losses as orch_losses

    for mod in (fsdp_losses, orch_losses):
        src = open(mod.__file__).read()
        assert "policy_version" not in src, f"policy_version referenced in {mod.__file__}"
        assert "staleness" not in src, f"staleness referenced in {mod.__file__}"

    # Positive control: policy_version DOES drive the rollouter/trainer staleness gate.
    import verl.experimental.fully_async_policy.chunk_sample as chunk_sample

    cs_src = open(chunk_sample.__file__).read()
    assert "policy_version" in cs_src and "def is_stale" in cs_src


if __name__ == "__main__":
    test_kl_divergence_is_pure_function_of_logprobs()
    test_policy_version_does_not_change_loss_or_gradient()
    test_no_version_reference_in_loss_modules_but_present_in_staleness_gate()
    print("FRESHNESS-INERT TEST PASS: policy_version does not affect loss or gradient under forward_kl_topk")
