# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""CPU unit tests for batched Newton-Schulz in TensorParallelMuon.

Verifies that batching same-shape matrix parameters into a single 3D Newton-Schulz call produces
the same parameter updates as the per-parameter path, up to floating-point reordering between
`matmul` (2D) and `baddbmm` (3D) kernels.
"""

import pytest
import torch

from megatron.core.optimizer.emerging_optimizers import HAVE_EMERGING_OPTIMIZERS, TensorParallelMuon

pytestmark = pytest.mark.skipif(
    not HAVE_EMERGING_OPTIMIZERS, reason="emerging_optimizers is not installed"
)

def _make_muon(params, use_batched_ns):
    return TensorParallelMuon(
        params,
        lr=3e-4,
        momentum=0.95,
        nesterov=True,
        weight_decay=0.01,
        split_qkv=False,
        fp32_matmul_prec="highest",
        num_ns_steps=3,
        use_batched_ns=use_batched_ns,
    )


def test_batched_ns_matches_unbatched():
    torch.manual_seed(0)
    # Three same-shape matrices (to exercise the batched path) plus one different shape.
    shapes = [(32, 32), (32, 32), (32, 32), (64, 64)]

    params_batched = [torch.nn.Parameter(torch.randn(s)) for s in shapes]
    params_unbatched = [torch.nn.Parameter(p.detach().clone()) for p in params_batched]
    grads = [torch.randn(s) for s in shapes]

    opt_batched = _make_muon(params_batched, use_batched_ns=True)
    opt_unbatched = _make_muon(params_unbatched, use_batched_ns=False)

    for p, g in zip(params_batched, grads):
        p.grad = g
    for p, g in zip(params_unbatched, grads):
        p.grad = g.clone()

    opt_batched.step()
    opt_unbatched.step()

    for p_batched, p_unbatched in zip(params_batched, params_unbatched):
        torch.testing.assert_close(p_batched, p_unbatched, rtol=1e-3, atol=1e-5)
