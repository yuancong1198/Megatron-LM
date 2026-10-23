# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Tests for the `muon_zero_parallelism` (Z) -> `num_distributed_optimizer_instances` (N)
conversion used by hybrid ZeRO.

The conversion lives in `_derive_muon_num_distributed_optimizer_instances`
(extracted from `validate_args`) so it can be exercised without a full
argument parse or model-parallel setup. These tests are pure CPU.
"""

import pytest

from megatron.training.arguments import _derive_muon_num_distributed_optimizer_instances

# (muon_zero_parallelism, dp_cp, expert_dp) -> expected N
_CONVERSION_CASES = [
    # Z == 0: no ZeRO parallelism limit -> full ZeRO, N == 1.
    (0, 8, None, 1),
    # Z == 1: every rank is its own replica group -> N == dp_cp.
    (1, 8, None, 8),
    # Z == 2: ceil(8 / 2) == 4 replicas.
    (2, 8, None, 4),
    # Z == 4: ceil(8 / 4) == 2 replicas.
    (4, 8, None, 2),
    # Z == dp_cp: limit does not constrain the group -> N == 1.
    (8, 8, None, 1),
    # Z == dp_cp: limit does not constrain the group -> N == 1.
    (16, 8, None, 1),
    # Odd dp_cp with an even divisor: ceil(12 / 2) == 6, 12 % 6 == 0.
    (2, 12, None, 6),
    # Z == 3 over dp_cp == 9: ceil(9 / 3) == 3, 9 % 3 == 0.
    (3, 9, None, 3),
]


@pytest.mark.parametrize("z,dp_cp,expert_dp,expected", _CONVERSION_CASES)
def test_muon_zero_parallelism_to_num_instances(z, dp_cp, expert_dp, expected):
    assert _derive_muon_num_distributed_optimizer_instances(z, dp_cp, expert_dp) == expected


def test_muon_zero_parallelism_with_expert_dp_divisible():
    # expert_dp == 4 evenly divides the derived N == 4 -> accepted.
    assert _derive_muon_num_distributed_optimizer_instances(2, 8, expert_dp=4) == 4
    # expert_dp == 2 evenly divides the derived N == 4 -> accepted.
    assert _derive_muon_num_distributed_optimizer_instances(2, 8, expert_dp=2) == 4


@pytest.mark.parametrize(
    "z,dp_cp,expert_dp",
    [
        # ceil(8 / 3) == 3, but 8 % 3 != 0 -> invalid (N must evenly divide dp_cp).
        (3, 8, None),
        # ceil(8 / 2) == 4, but expert_dp == 6 is not divisible by 4 -> invalid.
        (2, 8, 6),
        # ceil(10 / 4) == 3, but 10 % 3 != 0 -> invalid.
        (4, 10, None),
    ],
)
def test_muon_zero_parallelism_divisibility_failure(z, dp_cp, expert_dp):
    with pytest.raises(AssertionError):
        _derive_muon_num_distributed_optimizer_instances(z, dp_cp, expert_dp)


def test_muon_zero_parallelism_negative_is_no_limit():
    # Negative Z is treated as "no limit" exactly like Z == 0.
    assert _derive_muon_num_distributed_optimizer_instances(-1, 8) == 1
