# SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
import pytest
from nvfuser import FusionDefinition, DataType
from nvfuser.pytorch_utils import torch_dtype_to_nvfuser_dtype
from .core import run_benchmark, clear_dynamo_cache, with_executor, DEFAULT_EXECUTORS
import torch
from .global_params import generate_input_sizes, FLOAT_DTYPES, PROMOTE_DTYPES
from .torch_ops import gelu


def gelu_fwd_fusion(
    fd: FusionDefinition,
    dtype: DataType,
) -> None:
    input = fd.define_tensor(
        shape=[-1, -1], contiguity=[True, True], dtype=dtype, is_cpu=False
    )
    bias = fd.define_tensor(shape=[-1], contiguity=[True], dtype=dtype, is_cpu=False)
    if dtype in PROMOTE_DTYPES:
        input = fd.ops.cast(input, dtype=DataType.Float)
        bias = fd.ops.cast(bias, dtype=DataType.Float)
    S_079 = fd.define_scalar(0.79788456)
    S_004 = fd.define_scalar(0.044715)
    V1 = fd.define_vector([1, input.size(-1)], dtype=DataType.Int)
    bias = fd.ops.broadcast_in_dim(bias, shape=V1, broadcast_dims=[1])
    T1 = fd.ops.add(input, bias)
    T2 = fd.ops.mul(S_079, T1)
    T3 = fd.ops.mul(S_004, T1)
    T4 = fd.ops.mul(T3, T1)
    S1 = fd.define_scalar(1.0)
    T5 = fd.ops.add(T4, S1)
    T6 = fd.ops.mul(T2, T5)
    T7 = fd.ops.tanh(T6)
    T8 = fd.ops.add(S1, T7)
    T9 = fd.ops.mul(T8, T1)
    S2 = fd.define_scalar(0.50)
    T10 = fd.ops.mul(S2, T9)
    if dtype in PROMOTE_DTYPES:
        T10 = fd.ops.cast(T10, dtype=dtype)
    fd.add_output(T10)


@pytest.mark.parametrize("size", generate_input_sizes(dims=2))
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_gelu_fwd_nvf_benchmark(
    benchmark,
    size: tuple,
    dtype: torch.dtype,
    disable_validation: bool,
    disable_benchmarking: bool,
):
    inputs = [
        torch.randn(size, device="cuda", dtype=dtype, requires_grad=True),  # in_tensor
        torch.ones(size[-1], device="cuda", dtype=dtype),  # bias
    ]
    with FusionDefinition() as fd:
        gelu_fwd_fusion(fd, torch_dtype_to_nvfuser_dtype(dtype))
    if not disable_validation:
        eager_output = gelu(inputs)
        fd.validate(inputs, [eager_output])
    if not disable_benchmarking:
        run_benchmark(benchmark, fd.execute, inputs)


@pytest.mark.parametrize("executor", DEFAULT_EXECUTORS)
@pytest.mark.parametrize("size", generate_input_sizes(dims=2))
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_gelu_fwd_baseline_benchmark(
    benchmark,
    size: tuple,
    dtype: torch.dtype,
    executor: str,
):
    if executor == "torchcompile":
        clear_dynamo_cache()
    inputs = [
        torch.randn(size, device="cuda", dtype=dtype, requires_grad=True),  # in_tensor
        torch.ones(size[-1], device="cuda", dtype=dtype),  # bias
    ]

    benchmark_fn = with_executor(executor, gelu)

    # Inputs and outputs are same as nvFuser, no need for manual IOByte computation
    run_benchmark(benchmark, benchmark_fn, inputs)
