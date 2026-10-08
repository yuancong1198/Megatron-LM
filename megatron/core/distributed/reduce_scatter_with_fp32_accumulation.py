# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.


from typing import Any

import torch


class _ReduceScatterWithFP32AccumulationWorkHandle:
    """Work handle to return to user when using reduce_scatter_with_fp32_accumulation with
    async_op=True."""

    def __init__(
        self,
        all_to_all_handle: Any,
        all_to_all_output_tensor: torch.Tensor,
        output_tensor: torch.Tensor,
        world_size: int,
        op: torch.distributed.ReduceOp,
    ):
        """Initialize WorkHandle object."""
        self.all_to_all_handle = all_to_all_handle
        self.all_to_all_output_tensor = all_to_all_output_tensor
        self.output_tensor = output_tensor
        self.world_size = world_size
        self.op = op

    def wait(self):
        """Wait until communication (and associated computation) is completed."""
        # Wait for communication to complete if needed.
        if self.all_to_all_handle is not None:
            self.all_to_all_handle.wait()

        # Accumulate into a fp32 sum. The all-to-all output may be a lower-precision
        # (e.g. bf16 after stochastic-rounding quantization); summing with dtype=fp32
        # promotes it to FP32 so the accumulation stays exact.
        output_tensor_in_fp32 = torch.sum(
            self.all_to_all_output_tensor.view((self.world_size, -1)), dim=0, dtype=torch.float32
        )
        assert output_tensor_in_fp32.dtype == torch.float32

        # For ReduceOp.AVG, the collective averages the shard sum over the group: divide by
        # world_size after the local FP32 accumulation (numerically equivalent to NCCL's AVG).
        if self.op == torch.distributed.ReduceOp.AVG:
            output_tensor_in_fp32 = output_tensor_in_fp32 / self.world_size

        # Copy downcasted sum into output_tensor.
        self.output_tensor.copy_(output_tensor_in_fp32)


def stochastic_round_to_bf16(x: torch.Tensor) -> torch.Tensor:
    """Stochastically round a float32 tensor to bfloat16.

    BF16 keeps the top 16 bits of the FP32 bit pattern. The dropped low 16 bits are rounded up
    with probability equal to their fractional value: add a uniform 16-bit jitter to the integer
    representation, then truncate the low 16 bits. For finite values the carry never reaches the
    sign bit (bit 31), so plain integer add/mask is correct for negative values as well.
    """
    if x.dtype != torch.float32:
        raise TypeError(f"stochastic_round_to_bf16 expects float32 input, got {x.dtype}")
    x_int = x.view(torch.int32)
    jitter = torch.randint(0, 1 << 16, x_int.shape, device=x.device, dtype=torch.int32)
    rounded_int = (x_int + jitter) & ~0xFFFF
    return rounded_int.view(torch.float32).to(torch.bfloat16)

def _all_to_all_reduce_scatter_with_local_fp32_sum(
    output_tensor: torch.Tensor,
    input_tensor: torch.Tensor,
    op: torch.distributed.ReduceOp,
    group: torch.distributed.ProcessGroup,
    async_op: bool,
):
    """All-to-all the input shards, then sum them locally in FP32 into output_tensor.

    The all-to-all runs in the dtype of `input_tensor` (callers may quantize it to a lower
    precision first to halve the communicated volume); the local accumulation is always done in
    FP32 to avoid low-precision accumulation error, and the final sum is downcast back into
    `output_tensor`.
    """
    # Get world_size.
    if group is None:
        world_size = torch.distributed.get_world_size()
    else:
        world_size = group.size()

    # Make sure input_tensor size is divisible by world size.
    assert input_tensor.numel() % world_size == 0

    # Call all_to_all (every rank should have their respective gradient shards collected from
    # all ranks). We also create a tensor for the all-to-all output (the all-to-all collective
    # cannot be performed in-place).
    all_to_all_output_tensor = torch.empty_like(input_tensor)
    all_to_all_handle = torch.distributed.all_to_all_single(
        output=all_to_all_output_tensor, input=input_tensor, group=group, async_op=async_op
    )

    # Create a work handle to finish communication and reduction.
    reduce_scatter_handle = _ReduceScatterWithFP32AccumulationWorkHandle(
        all_to_all_handle, all_to_all_output_tensor, output_tensor, world_size, op
    )
    if async_op:
        # Return work handle; consumers can call .wait() to ensure communication and associated
        # reduction complete.
        return reduce_scatter_handle
    else:
        # Wait on work handle.
        reduce_scatter_handle.wait()

def reduce_scatter_with_fp32_accumulation(
    output_tensor: torch.Tensor,
    input_tensor: torch.Tensor,
    op: torch.distributed.ReduceOp,
    group: torch.distributed.ProcessGroup,
    async_op: bool,
):
    """Reduce-scatter with FP32 accumulation.

    Collects input_tensor in lower precision using an all-to-all, then locally accumulates in FP32
    precision, then downcasts final sum back into right location in input_tensor.

    Args:
        output_tensor (torch.Tensor): Output tensor with reduce-scattered output (only the shard).
        input_tensor (torch.Tensor): Input tensor that needs to be reduce-scattered.
        op (torch.distributed.ReduceOp): SUM or AVG. AVG divides the local FP32 sum by the group size.
        group (torch.distributed.ProcessGroup): Process group to use for reduce-scatter.
        async_op (bool): If True, return a work handle; if False, block until the collective completes.
    """
    # Make sure arguments conform to the implementation.
    assert op in (torch.distributed.ReduceOp.SUM, torch.distributed.ReduceOp.AVG)

    return _all_to_all_reduce_scatter_with_local_fp32_sum(
        output_tensor, input_tensor, op, group, async_op
    )

def reduce_scatter_with_bf16_stochastic_rounding(
    output_tensor: torch.Tensor,
    input_tensor: torch.Tensor,
    op: torch.distributed.ReduceOp,
    group: torch.distributed.ProcessGroup,
    async_op: bool,
):
    """Reduce-scatter with BF16 stochastic-rounding quantization and FP32 accumulation.

    Quantizes `input_tensor` from FP32 to BF16 using stochastic rounding (halving the volume
    communicated by the all-to-all), then locally accumulates the shards in FP32 to avoid
    low-precision accumulation error. Intended for MoE (expert-parallel) gradients.

    Args:
        output_tensor (torch.Tensor): Output tensor with reduce-scattered output (only the shard).
        input_tensor (torch.Tensor): Input tensor that needs to be reduce-scattered (FP32).
        op (torch.distributed.ReduceOp): SUM or AVG. AVG divides the local FP32 sum by the group size.
        group (torch.distributed.ProcessGroup): Process group to use for reduce-scatter.
        async_op (bool): If True, return a work handle; if False, block until the collective completes.
    """
    # Make sure arguments conform to the implementation.
    assert op in (torch.distributed.ReduceOp.SUM, torch.distributed.ReduceOp.AVG)

    return _all_to_all_reduce_scatter_with_local_fp32_sum(
        output_tensor, stochastic_round_to_bf16(input_tensor), op, group, async_op
    )