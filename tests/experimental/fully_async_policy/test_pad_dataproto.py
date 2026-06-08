# Correctness test for pad_dataproto_to_prompt_response_width (the fix for the
# DataProto.concat shape-mismatch crash on variable prompt/response lengths).
#
# Verifies that padding two samples with DIFFERENT prompt AND response widths to a common
# (prompt, response) width: (a) makes DataProto.concat succeed, (b) preserves the real prompt
# tokens at the RIGHT of the prompt span (left-pad) and real response tokens at the LEFT of the
# response span (right-pad), (c) builds attention_mask=0 over pads, (d) recomputes position_ids
# in the standard left-pad convention, (e) right-pads response-aligned tensors with 0.
#
# Needs torch + tensordict + the verl DataProto -> run inside the training container.
import torch
from tensordict import TensorDict

from verl import DataProto
from verl.experimental.fully_async_policy.detach_utils import pad_dataproto_to_prompt_response_width as padfn


def _mk(prompt, resp, logp):
    p, r = len(prompt), len(resp)
    bt = TensorDict(
        {
            "prompts": torch.tensor([prompt]),
            "responses": torch.tensor([resp]),
            "input_ids": torch.tensor([prompt + resp]),
            "attention_mask": torch.tensor([[1] * (p + r)]),
            "position_ids": torch.tensor([list(range(p + r))]),
            "response_mask": torch.tensor([[1] * r]),
            "old_log_probs": torch.tensor([logp]),
        },
        batch_size=[1],
    )
    return DataProto(batch=bt, non_tensor_batch={}, meta_info={})


def test_pad_and_concat_mixed_widths():
    A = _mk([10, 11, 12], [20, 21, 22, 23], [0.1, 0.2, 0.3, 0.4])      # p=3 r=4
    B = _mk([30, 31, 32, 33, 34], [40, 41], [0.5, 0.6])                # p=5 r=2
    PA = padfn(A, 5, 4, 0)
    PB = padfn(B, 5, 4, 0)
    cat = DataProto.concat([PA, PB])
    b = cat.batch

    # (a) uniform shapes -> concat succeeded
    assert tuple(b["prompts"].shape) == (2, 5)
    assert tuple(b["responses"].shape) == (2, 4)
    assert tuple(b["input_ids"].shape) == (2, 9)

    # (b) real tokens preserved at the right positions
    assert b["prompts"][0].tolist() == [0, 0, 10, 11, 12]              # left-pad
    assert b["responses"][0].tolist() == [20, 21, 22, 23]
    assert b["input_ids"][0].tolist() == [0, 0, 10, 11, 12, 20, 21, 22, 23]
    assert b["prompts"][1].tolist() == [30, 31, 32, 33, 34]
    assert b["responses"][1].tolist() == [40, 41, 0, 0]               # right-pad
    assert b["input_ids"][1].tolist() == [30, 31, 32, 33, 34, 40, 41, 0, 0]

    # (c) attention_mask: 0 over pads only
    assert b["attention_mask"][0].tolist() == [0, 0, 1, 1, 1, 1, 1, 1, 1]
    assert b["attention_mask"][1].tolist() == [1, 1, 1, 1, 1, 1, 1, 0, 0]

    # (d) position_ids: standard left-pad convention (cumsum(attn)-1).clamp(0)
    assert b["position_ids"][0].tolist() == [0, 0, 0, 1, 2, 3, 4, 5, 6]
    assert b["position_ids"][1].tolist() == [0, 1, 2, 3, 4, 5, 6, 6, 6]

    # (e) response-aligned tensors right-padded with 0; reals intact
    assert b["response_mask"][0].tolist() == [1, 1, 1, 1]
    assert b["response_mask"][1].tolist() == [1, 1, 0, 0]
    lp = b["old_log_probs"][1].tolist()
    assert abs(lp[0] - 0.5) < 1e-6 and abs(lp[1] - 0.6) < 1e-6 and lp[2] == 0.0 and lp[3] == 0.0

    # no-op path: already at target width returns an equal batch
    same = padfn(A, 3, 4, 0)
    assert same.batch["input_ids"][0].tolist() == [10, 11, 12, 20, 21, 22, 23]


if __name__ == "__main__":
    test_pad_and_concat_mixed_widths()
    print("PAD DATAPROTO TEST PASS")
