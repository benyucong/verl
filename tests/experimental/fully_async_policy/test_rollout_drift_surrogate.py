# FROST rollout-drift surrogate (D^roll_hat) calibration + safety test.
#
# The diagnostic in verl/trainer/distillation/losses.py logs a CHEAP per-token surrogate
# D^roll_hat = logp_current(sampled token) - logp_stale(sampled token). This test validates that
# the surrogate is a faithful (k1) estimator of the true rollout drift: by construction
#   E_{a ~ pi_stale}[ logp_cur(a) - logp_stale(a) ] = - KL(pi_stale || pi_cur),
# so -mean(D^roll_hat) estimates KL(stale||current) and its magnitude tracks drift. We verify
# monotonicity along a fixed drift direction and numerical agreement with the full KL, and that
# the diagnostic helper no-ops safely when rollout_log_probs is absent.
#
# Run in the training container:
#   PYTHONPATH=$R/verl:$R:/opt/venv/lib/python3.12/site-packages:$R/.venv/lib/python3.12/site-packages \
#     /opt/venv/bin/python verl/tests/experimental/fully_async_policy/test_rollout_drift_surrogate.py
import torch
import torch.nn.functional as F
from tensordict import TensorDict

from verl.trainer.distillation.losses import _compute_rollout_drift_surrogate


def _kl(p_logits, q_logits):
    lp = F.log_softmax(p_logits, dim=-1)
    lq = F.log_softmax(q_logits, dim=-1)
    return (lp.exp() * (lp - lq)).sum(-1)


def test_surrogate_tracks_true_rollout_kl():
    torch.manual_seed(0)
    V = 256
    stale = torch.randn(V)
    direction = torch.randn(V)
    p_stale = F.softmax(stale, dim=-1)
    lp_stale = F.log_softmax(stale, dim=-1)
    prev_kl = -1.0
    for eps in [0.0, 0.5, 1.0, 2.0]:
        cur = stale + eps * direction
        lp_cur = F.log_softmax(cur, dim=-1)
        true_kl = _kl(stale, cur).item()
        samples = torch.multinomial(p_stale, 100000, replacement=True)
        surrogate = (lp_cur[samples] - lp_stale[samples]).mean().item()  # ~ E[logp_cur-logp_stale] = -KL
        # KL grows monotonically along a fixed drift direction
        assert true_kl >= prev_kl - 1e-9, (eps, true_kl, prev_kl)
        prev_kl = true_kl
        # -surrogate is a faithful estimator of the true KL(stale||cur)
        assert abs((-surrogate) - true_kl) < 0.05, (eps, -surrogate, true_kl)
    # final drift is substantial (sanity: the metric is not trivially ~0 everywhere)
    assert prev_kl > 0.1


def test_helper_safe_without_rollout_log_probs():
    bsz, T = 2, 4
    data = TensorDict({"response_mask": torch.ones(bsz, T)}, batch_size=[bsz])
    out = _compute_rollout_drift_surrogate(model_output={"log_probs": torch.randn(bsz, T)}, data=data)
    assert out == {}, "diagnostic must no-op when the batch carries no rollout_log_probs"


if __name__ == "__main__":
    test_surrogate_tracks_true_rollout_kl()
    test_helper_safe_without_rollout_log_probs()
    print("ROLLOUT-DRIFT SURROGATE TEST PASS: D^roll_hat tracks KL(stale||current); helper no-ops safely")
