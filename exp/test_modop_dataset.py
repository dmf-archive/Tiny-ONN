import torch

from exp.sia_plsd_fast_test import build_mod_addition


def test_mod_addition_mixed_ratio() -> None:
    dataset, _ = build_mod_addition(3, 1.0, 0)
    assert dataset.pairs.shape[1] == 3
    ops = dataset.pairs[:, 2]
    add_token = 3
    sub_token = 4
    assert set(ops.tolist()) == {add_token, sub_token}
    total = 3 * 3
    add_count = int((ops == add_token).sum().item())
    sub_count = int((ops == sub_token).sum().item())
    expected_add = int(total * 0.55)
    expected_sub = total - expected_add
    assert add_count == expected_add
    assert sub_count == expected_sub
