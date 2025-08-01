# SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Owner(s): ["module: nvfuser"]

import itertools
from functools import partial, wraps

import math
import torch
from torch.testing import make_tensor
import random
from numbers import Number

from opinfo_core import OpInfo, SampleInput, ErrorSample, Domain
from opinfo_utils import (
    make_number,
    find_nonmatching_dtype,
    is_floating_dtype,
    float_complex_dtypes,
    complex_dtypes,
)
import sys

if "nvfuser" in sys.modules:
    from nvfuser import DataType
    from nvfuser.pytorch_utils import torch_dtype_to_nvfuser_dtype
else:
    from nvfuser_direct import DataType
    from nvfuser_direct.pytorch_utils import torch_dtype_to_nvfuser_dtype

MINIMUM_SYMBOLIC_SIZE = -1
INT64_MAX = 2**63 - 1
MAX_TENSOR_DIMS = 8


# Determine if a number is with desired Domain [low, high)
# The domain is half-open. The lower limit is inclusive while the upper limit is exclusive.
def is_within_domain(domain: Domain, a: Number, exclude_zero: bool = False):
    # comparison operators are not defined for complex numbers
    if isinstance(a, complex):
        return True

    if domain.low is not None and domain.low > a:
        return False

    if domain.high is not None and a >= domain.high:
        return False

    if exclude_zero and a == 0:
        return False

    return True


def _extremal_values(dtype: torch.dtype):
    _float_vals = (float("inf"), float("-inf"), float("nan"))
    _complex_vals = tuple(
        complex(*x) for x in itertools.product(_float_vals, _float_vals)
    )
    _int16_vals = (-32768, 32767)
    _int32_vals = (-2147483648, 2147483647)
    _int64_vals = (-9223372036854775808, 9223372036854775807)

    if dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        return _float_vals
    elif dtype in (torch.complex64, torch.complex128):
        return _complex_vals
    elif dtype is torch.int16:
        return _int16_vals
    elif dtype is torch.int32:
        return _int32_vals
    elif dtype is torch.int64:
        return _int64_vals
    else:
        raise ValueError(f"Unsupported dtype --- {dtype}")


def _large_values(dtype: torch.dtype):
    _int_vals = (-1113, 1113, -10701, 10701)
    _float16_vals = (-501, 501, -1001.2, 1001.2, -13437.7, 13437.7)
    _float_vals = _float16_vals + (-4988429.2, 4988429.2, -1e20, 1e20)
    _complex_vals = tuple(
        complex(*x) for x in itertools.product(_float_vals, _float_vals)
    )

    if dtype == torch.float16:
        return _float16_vals
    elif dtype in (torch.bfloat16, torch.float32, torch.float64):
        return _float_vals
    elif dtype in (torch.complex64, torch.complex128):
        print(_complex_vals)
        return _complex_vals
    elif dtype in (torch.int16, torch.int32, torch.int64):
        return _int_vals
    else:
        raise ValueError(f"Unsupported dtype --- {dtype}")


def _small_values(dtype: torch.dtype):
    eps = 1e-5
    _int_vals = (0, -1, 1, -55, 55, -127, 127, -128)
    _float_vals = (
        0.0,
        -0.0,
        -1e-3,
        1e-3,
        -0.25,
        0.25,
        -1.0,
        1.0,
        -math.e / 2.0,
        math.e / 2.0,
        -math.e + eps,
        math.e - eps,
        -math.e,
        math.e,
        -math.e - eps,
        math.e + eps,
    )
    _complex_vals = tuple(
        complex(*x) for x in itertools.product(_float_vals, _float_vals)
    )

    if dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        return _float_vals
    elif dtype in (torch.complex64, torch.complex128):
        return _complex_vals
    elif dtype in (torch.int16, torch.int32, torch.int64):
        return _int_vals
    else:
        raise ValueError(f"Unsupported dtype --- {dtype}")


def broadcast_error_generator(
    op: OpInfo, dtype: torch.dtype, requires_grad: bool = False, **kwargs
):
    # jax.lax.broadcast(operand, sizes)
    # add new dimensions to left-hand-side of tensor
    # dims = tuple(range(len(sizes), len(sizes) + np.ndim(operand)))
    # return broadcast_in_dim(operand, tuple(sizes) + np.shape(operand), dims)

    make_arg = partial(
        make_tensor, device="cuda", dtype=dtype, requires_grad=requires_grad
    )

    fewer_original_axes = (
        ([2, 3], [True, False]),
        RuntimeError,
        "Invalid broadcast, number of false entries in is_broadcast_dim expected to be",
    )

    greater_original_axes = (
        ([2, 3], [True, False, False, False]),
        RuntimeError,
        "Invalid broadcast, number of false entries in is_broadcast_dim expected to be",
    )

    error_cases = [
        fewer_original_axes,
        greater_original_axes,
    ]
    for es in error_cases:
        ex_case, ex_type, ex_str = es
        input_shape, bcast_dims = ex_case
        input_tensor = make_arg(input_shape)
        yield SampleInput(input_tensor, bcast_dims), ex_type, ex_str


def broadcast_in_dim_generator(
    op: OpInfo, dtype: torch.dtype, requires_grad: bool = False, **kwargs
):
    make_arg = partial(
        make_tensor, device="cuda", dtype=dtype, requires_grad=requires_grad
    )

    # The first 5 test cases below are taken from JAX's broadcast_in_dim tests
    #   https://github.com/google/jax/blob/main/tests/lax_test.py#L1171
    # input shape, output shape, bcast_dims
    cases = (
        ([2], [2, 2], [0]),
        ([2], [2, 2], [1]),
        ([2], [2, 3], [0]),
        ([], [2, 3], []),
        ([1], [2, 3], [1]),
        ((4, 6, 3, 1), (5, 4, 7, 6, 3, 6, 6), (1, 3, 4, 5)),
    )

    for input_shape, output_shape, bcast_dims in cases:
        input_tensor = make_arg(input_shape)
        if op.name == "broadcast_in_dim_symbolic":
            bcast_shaped_tensor = make_arg(output_shape)
            yield SampleInput(input_tensor, bcast_shaped_tensor, bcast_dims)
        else:
            yield SampleInput(input_tensor, output_shape, bcast_dims)


def broadcast_in_dim_error_generator(
    op: OpInfo, dtype: torch.dtype, requires_grad: bool = False, **kwargs
):
    # jax.lax.broadcast_in_dim(operand, shape, broadcast_dimensions)
    make_arg = partial(
        make_tensor, device="cuda", dtype=dtype, requires_grad=requires_grad
    )

    # 1. Every dimension in the input tensor must be used in broadcast_dimensions.
    missing_axis_in_bcast_dims = (
        ([2, 2], [2, 2, 3], [0]),
        RuntimeError,
        "The broadcast dimensions should match the input dimensions.",
    )

    # 2. New shape has weakly more dimentions than the original tensor.
    fewer_dims_in_output_shape = (
        ([2, 2], [2], [0]),
        RuntimeError,
        "The new shape is expected to be greater-then-or-equal to the input",
    )

    # 3. broadcast_dimensions is an ascending sequence of integers.
    descending_broadcast_dimensions = (
        ([2, 2], [2, 2], [1, 0]),
        RuntimeError,
        "Broadcast dimension is not greater than the previous value.",
    )

    # 4. Each broadcast dimension is within the new shape.
    out_of_bounds_broadcast_dimensions = (
        ([2, 2], [2, 2], [0, 2]),
        RuntimeError,
        "Invalid broadcast_dims value.",
    )

    # 5. The original tensor is not broadcastable to desired shape.
    # tensor.shape[idx] == 1 or tensor.shape[idx] == output_shape[new_idx]
    #
    # Jax Exception:
    # TypeError: broadcast_in_dim operand dimension sizes must either be 1,
    # or be equal to their corresponding dimensions in the target broadcast shape;
    # got operand of shape (2, 3), target broadcast shape (2, 3, 4), broadcast_dimensions (0, 2)
    not_broadcastable = (
        ([2, 3], [2, 3, 4], [0, 2]),
        RuntimeError,
        "Invalid broadcast_dims value.",
    )

    # 6. TypeError: broadcast_in_dim shape must have every element be nonnegative, got (-1, 2, 3).
    negative_shape = (
        ([2, 3], [2, 3, -1], [0, 1]),
        RuntimeError,
        "Invalid broadcast_dims value.",
    )

    # TODO add exceptions for not_broadcastable, negative output shape
    error_cases = [
        missing_axis_in_bcast_dims,
        fewer_dims_in_output_shape,
        descending_broadcast_dimensions,
        out_of_bounds_broadcast_dimensions,
        # not_broadcastable,
        # negative_shape,
    ]
    for es in error_cases:
        ex_case, ex_type, ex_str = es
        input_shape, output_shape, bcast_dims = ex_case
        input_tensor = make_arg(input_shape)
        if op.name == "broadcast_in_dim_symbolic":
            bcast_shaped_tensor = make_arg(output_shape)
            yield SampleInput(
                input_tensor, bcast_shaped_tensor, bcast_dims
            ), ex_type, ex_str
        else:
            yield SampleInput(input_tensor, output_shape, bcast_dims), ex_type, ex_str


def cat_generator(
    op: OpInfo, dtype: torch.dtype, requires_grad: bool = False, **kwargs
):
    make_arg = partial(
        make_tensor, device="cuda", dtype=dtype, requires_grad=requires_grad
    )

    # concatenating tensors along singleton, broadcast dimensions is unsupported by nvFuser.
    # https://github.com/NVIDIA/Fuser/issues/224
    # shapes, dim
    cases = [
        ([(3,)], 0),  # single tensor provided
        # 1D
        ([(2,), (3,)], 0),
        ([(2,), (4,)], 0),
        ([(0,), (2,)], 0),
        ([(0,), (2,)], -1),
        ([(2, 3), (2, 4)], 1),
        ([(2, 3), (2, 4), (2, 5)], 1),
    ]

    for shapes, dim in cases:
        yield SampleInput([make_arg(s) for s in shapes], dim)


def cat_error_generator(op, dtype=torch.float32, requires_grad: bool = False, **kwargs):
    make_arg = partial(
        make_tensor, device="cuda", dtype=dtype, requires_grad=requires_grad
    )
    # shapes, dim, exception type, exception string
    empty_input_tensors = (
        ([], 0),
        RuntimeError,
        "Attempting to concatenate empty list of tensors",
    )
    positive_dim = (([(1,), (2,)], 1), RuntimeError, "Invalid dimension to cat")
    negative_dim = (([(2,), (2,)], -2), RuntimeError, "Invalid dimension to cat")
    # All tensors must have same number of dimension"
    ndims_mismatch = (
        ([(2,), (2, 3)], 0),
        RuntimeError,
        "Unexpected number of dimensions",
    )
    # All tensors must have same shape except for the cat dimension
    shape_mismatch = (
        ([(2, 3), (4, 5)], 0),
        RuntimeError,
        "a conflict was found with 2 different sizes",
    )

    error_cases = [
        empty_input_tensors,
        positive_dim,
        negative_dim,
        ndims_mismatch,
        shape_mismatch,
    ]

    for case, ex_type, ex_str in error_cases:
        shapes, dim = case
        yield SampleInput([make_arg(s) for s in shapes], dim), ex_type, ex_str


def define_tensor_error_generator(
    op: OpInfo, dtype: torch.dtype, requires_grad: bool = False, **kwargs
):
    """
    "define_tensor",
    [](FusionDefinition& self,
        std::vector<int64_t>& sizes,
        std::vector<int64_t>& strides,
        PrimDataType dtype = DataType::Float,
        bool static_sizes = false,
        bool is_cpu = false) -> Tensor {
    ---
    "define_tensor",
    [](FusionDefinition& self,
        std::vector<int64_t>& shape,
        std::vector<std::optional<bool>>& contiguity,
        PrimDataType dtype = DataType::Float,
        bool is_cpu = false) -> Tensor {
    """

    check_size_contiguity_match = ErrorSample(
        {
            "shape": [-1, -1],
            "contiguity": [True, True, True],
            "dtype": DataType.Float,
        },
        "Length of contiguity argument (3) must match that of shape argument (2)",
    )

    check_empty_tensor_size = ErrorSample(
        {"shape": [], "contiguity": []},
        "Empty tensor is unsupported.",
    )

    check_max_tensor_size = ErrorSample(
        {
            "shape": [-1 for _ in range(MAX_TENSOR_DIMS + 1)],
            "contiguity": [True for _ in range(MAX_TENSOR_DIMS + 1)],
        },
        "The specified tensor dimensionality exceeds the max tensor size for nvfuser.",
    )

    check_above_size_range = ErrorSample(
        {"shape": [INT64_MAX + 1], "contiguity": [True]},
        "define_tensor(): incompatible function arguments",
        TypeError,
    )

    check_below_size_range = ErrorSample(
        {"shape": [MINIMUM_SYMBOLIC_SIZE - 1], "contiguity": [True]},
        "The value -2 at index 0 was neither symbolic(-1), zero_element(0), broadcast(1), or static(>1)",
    )

    check_contiguity_unknown_values = ErrorSample(
        {"shape": [10], "contiguity": [-1]},
        "define_tensor(): incompatible function arguments.",
        TypeError,
    )

    check_shape_unknown_dtypes = ErrorSample(
        {"shape": [10.0], "contiguity": [True]},
        "define_tensor(): incompatible function arguments.",
        TypeError,
    )

    check_stride_order_duplicate = ErrorSample(
        {
            "shape": [-1, -1, -1],
            "contiguity": [True, True, True],
            "stride_order": [0, 1, 1],
        },
        "duplicated stride_order entries",
        RuntimeError,
    )

    check_stride_order_out_of_range = ErrorSample(
        {
            "shape": [-1, -1, -1],
            "contiguity": [True, True, True],
            "stride_order": [0, 1, 5],
        },
        "stride_order argument is out of range",
        RuntimeError,
    )

    check_stride_order_out_of_negative_range = ErrorSample(
        {
            "shape": [-1, -1, -1],
            "contiguity": [True, True, True],
            "stride_order": [0, 1, -4],
        },
        "stride_order argument is out of range",
        RuntimeError,
    )

    # TODO: Fix empty and maximum tensor dimensionality error checks.
    # TODO: Add invalid argument checks for contiguity.
    error_cases = [
        check_size_contiguity_match,
        # check_empty_tensor_size,
        # check_max_tensor_size,
        check_above_size_range,
        check_below_size_range,
        # check_contiguity_unknown_values,
        check_shape_unknown_dtypes,
    ]

    input_tensor = make_tensor(
        (10, 10), device="cuda", dtype=dtype, requires_grad=requires_grad
    )
    for es in error_cases:
        yield SampleInput(input_tensor, **es.kwargs), es.ex_type, es.ex_str


def define_vector_constant_error_generator(
    op: OpInfo, dtype: torch.dtype, requires_grad: bool = False, **kwargs
):
    """
    "define_vector",
    [](FusionDefinition& self, py::list& values) -> Vector {
    """

    check_above_size_range = ErrorSample(
        {"values": [INT64_MAX + 1]},
        "define_vector(): incompatible function arguments",
        TypeError,
    )

    check_below_size_range = ErrorSample(
        {"values": [MINIMUM_SYMBOLIC_SIZE - 1]},
        "The value -2 at index 0 was neither symbolic(-1), zero_element(0), broadcast(1), or static(>1)",
    )

    error_cases = [
        # FIXME: The above_size_range case gives a non-sensical error message.
        # "Unable to cast Python instance to C++ type (#define PYBIND11_DETAILED_ER"
        # check_above_size_range,
        check_below_size_range,
    ]

    for es in error_cases:
        yield SampleInput(**es.kwargs), es.ex_type, es.ex_str


def _special_value_binary_generator(
    lhs_generator_fn, rhs_generator_fn, dtype, requires_grad
):
    lhs_vals, rhs_vals = zip(*itertools.product(lhs_generator_fn, rhs_generator_fn))
    lhs = torch.tensor(
        lhs_vals, device="cuda", dtype=dtype, requires_grad=requires_grad
    )
    rhs = torch.tensor(
        rhs_vals, device="cuda", dtype=dtype, requires_grad=requires_grad
    )
    return SampleInput(lhs, rhs)


def elementwise_binary_generator(
    op: OpInfo,
    dtype: torch.dtype,
    requires_grad: bool = False,
    *,
    supports_numbers: bool = True,
    enable_broadcast_testing: bool = True,
    enable_extremal_value_testing: bool = True,
    enable_large_value_testing: bool = True,
    enable_small_value_testing: bool = True,
    **kwargs,
):
    low = None if op.domain.low is None else max(-9, op.domain.low)
    high = None if op.domain.high is None else min(9, op.domain.high)
    make_arg = partial(
        make_tensor,
        device="cuda",
        dtype=dtype,
        low=low,
        high=high,
        requires_grad=requires_grad,
        **kwargs,
    )

    shapes = (
        (0, 2, 1),
        (5, 0, 3),
        (),
        (11,),
        (4, 4),
        (1024, 1024),
        (64, 64, 64),
    )

    # Typical inputs
    for shape in shapes:
        yield SampleInput(make_arg(shape), make_arg(shape))
        yield SampleInput(
            make_arg(shape, noncontiguous=True), make_arg(shape, noncontiguous=True)
        )

    if enable_broadcast_testing:
        broadcast_shapes = (
            ((1,), ()),
            ((2,), ()),
            ((1,), (2,)),
            ((2, 1), (2,)),
            ((1, 2), (2,)),
            ((3, 2), (2,)),
            ((1, 3, 2), (2,)),
            ((1, 3, 2), (3, 2)),
            ((3, 1, 2), (3, 2)),
            ((2, 3, 2), ()),
            ((3, 1, 2), (1, 3, 2)),
        )
        for lhs_shape, rhs_shape in broadcast_shapes:
            yield SampleInput(make_arg(lhs_shape), make_arg(rhs_shape))
            yield SampleInput(
                make_arg(lhs_shape, noncontiguous=True),
                make_arg(rhs_shape, noncontiguous=True),
            )

    # Create filtered special inputs for this operation's domain
    def _filter_lhs_domain(values):
        return [v for v in values if is_within_domain(op.domain, v)]

    def _filter_rhs_domain(values):
        # NOTE: Check exclude_zero flag to avoid undefined behavior such as ZeroDivisionError: division by zero
        exclude_zero = kwargs.get("exclude_zero", False)
        return [v for v in values if is_within_domain(op.domain, v, exclude_zero)]

    if (
        enable_large_value_testing
        and dtype != torch.bool
        and dtype not in complex_dtypes
    ):
        lhs_large_values = _filter_lhs_domain(_large_values(dtype))
        rhs_large_values = _filter_rhs_domain(_large_values(dtype))
        yield _special_value_binary_generator(
            lhs_large_values, rhs_large_values, dtype, requires_grad
        )

    if enable_small_value_testing and dtype != torch.bool:
        lhs_small_values = _filter_lhs_domain(_small_values(dtype))
        rhs_small_values = _filter_rhs_domain(_small_values(dtype))
        yield _special_value_binary_generator(
            lhs_small_values, rhs_small_values, dtype, requires_grad
        )

    if enable_extremal_value_testing and dtype in float_complex_dtypes:
        lhs_extremal_values = _filter_lhs_domain(_extremal_values(dtype))
        rhs_extremal_values = _filter_rhs_domain(_extremal_values(dtype))
        yield _special_value_binary_generator(
            lhs_extremal_values, rhs_extremal_values, dtype, requires_grad
        )

        # Test interactions between extreme and normal values
        make_cuda_tensor = partial(
            torch.tensor, device="cuda", dtype=dtype, requires_grad=requires_grad
        )
        rhs_normal = [random.uniform(-10, 10) for _ in range(len(lhs_extremal_values))]
        lhs_normal = [random.uniform(-10, 10) for _ in range(len(rhs_extremal_values))]
        yield SampleInput(
            make_cuda_tensor(lhs_extremal_values), make_cuda_tensor(rhs_normal)
        )
        yield SampleInput(
            make_cuda_tensor(lhs_normal), make_cuda_tensor(rhs_extremal_values)
        )


def _elementwise_binary_torch(op):
    @wraps(op)
    def _fn(x, y):
        if isinstance(x, torch.Tensor) or isinstance(y, torch.Tensor):
            return op(x, y)
        return op(torch.tensor(x), torch.tensor(y)).item()

    return _fn


def elementwise_unary_generator(
    op: OpInfo,
    dtype: torch.dtype,
    requires_grad: bool = False,
    *,
    supports_numbers: bool = True,
    enable_extremal_value_testing: bool = True,
    enable_large_value_testing: bool = True,
    enable_small_value_testing: bool = True,
    **kwargs,
):
    low = None if op.domain.low is None else max(-9, op.domain.low)
    high = None if op.domain.high is None else min(9, op.domain.high)
    make_arg = partial(
        make_tensor,
        device="cuda",
        dtype=dtype,
        low=low,
        high=high,
        requires_grad=requires_grad,
        **kwargs,
    )

    shapes = (
        (0, 2, 1),
        (5, 0, 3),
        (),
        (11,),
        (4, 4),
        (1024, 1024),
        (64, 64, 64),
    )

    # Typical inputs
    for shape in shapes:
        yield SampleInput(make_arg(shape))
        yield SampleInput(make_arg(shape, noncontiguous=True))

    # Create filtered special inputs for this operation's domain
    def _filter_domain(values):
        return [v for v in values if is_within_domain(op.domain, v)]

    if (
        enable_large_value_testing
        and dtype != torch.bool
        and dtype not in complex_dtypes
    ):
        filtered_large_values = _filter_domain(_large_values(dtype))
        yield SampleInput(
            torch.tensor(
                filtered_large_values,
                device="cuda",
                dtype=dtype,
                requires_grad=requires_grad,
            )
        )

    if enable_small_value_testing and dtype != torch.bool:
        filtered_small_values = _filter_domain(_small_values(dtype))
        yield SampleInput(
            torch.tensor(
                filtered_small_values,
                device="cuda",
                dtype=dtype,
                requires_grad=requires_grad,
            )
        )

    if enable_extremal_value_testing and dtype in float_complex_dtypes:
        filtered_extremal_values = _filter_domain(_extremal_values(dtype))
        yield SampleInput(
            torch.tensor(
                filtered_extremal_values,
                device="cuda",
                dtype=dtype,
                requires_grad=requires_grad,
            )
        )


def _elementwise_unary_torch(op):
    @wraps(op)
    def _fn(x):
        if isinstance(x, torch.Tensor):
            return op(x)
        return op(torch.tensor(x)).item()

    return _fn


def full_error_generator(
    op: OpInfo, dtype: torch.dtype, requires_grad: bool = False, **kwargs
):
    # torch.full(size, fill_value, dtype=None)
    # Error: Trying to create tensor with negative dimension
    negative_input_shape = [2, -2]
    yield SampleInput(
        negative_input_shape, make_number(dtype), dtype
    ), RuntimeError, "The value -2 at index 1 was neither symbolic(-1), zero_element(0), broadcast(1), or static(>1)."


def scatter_generator(
    op: OpInfo, dtype: torch.dtype, requires_grad: bool = False, **kwargs
):
    # torch.scatter(input: Tensor, dim: int, index: LongTensor, src: LongTensor)
    # * input, index and src tensors have same ndims.
    # * index tensors must be <= input tensor along all dims.
    # * index tensors must be == src tensor along all dims.
    # * index tensors must have unique value across specified axis.

    make_arg = partial(
        make_tensor, device="cuda", dtype=dtype, requires_grad=requires_grad
    )

    def make_unique_index(shape_b, dim, extent):
        logits_shape = list(shape_b)
        logits_shape[dim] = extent
        logits = make_tensor(logits_shape, device="cuda", dtype=torch.float)
        # return index tensor with unique entry
        return logits.argsort(dim).narrow(dim, 0, shape_b[dim])

    make_index = partial(
        make_tensor, device="cuda", dtype=torch.long, requires_grad=False
    )

    # a.shape, dim, b.shape
    cases = (
        ((8, 2, 3), 0, (8, 2, 3)),
        ((8, 2, 3), 1, (8, 2, 3)),
        ((8, 2, 3), 2, (8, 2, 3)),
        # TODO: enable the test below when we fix mapping for scatter.
        # scatter supporting unmatched scatter dim
        # ((8, 2, 3), 0, (4, 2, 3)),
        # ((8, 2, 3), 1, (8, 1, 3)),
        # ((8, 2, 3), 2, (8, 2, 2)),
        # ((8,), 0, (8)),
        # ((8,), 0, (4)),
        # ((8,), 0, (1)),
        # ((8, 2, 3), -3, (4, 2, 3)),
        # ((8, 2, 3), -2, (8, 1, 3)),
        # ((8, 2, 3), -1, (8, 2, 2)),
        # TODO: should we support mismatching dims?
        # scatter supporting unmatched all dims
        # ((8, 2, 3), 0, (4, 2, 3)),
        # ((4, 2, 3), 1, (4, 2, 3)),
        # ((4, 2, 3), 2, (4, 2, 2)),
        # ((8,), 0, (8)),
        # ((8,), 0, (4)),
        # ((8,), 0, (1)),
        # ((4, 1), 0, (1, 1)),
        # ((4, 5), 1, (4, 1)),
        # ((8, 2, 3), 0, (4, 1, 2)),
        # ((8, 2, 3), 1, (4, 1, 2)),
        # ((8, 2, 3), 2, (4, 1, 2)),
        # ((8, 2, 3), -3, (4, 2, 3)),
        # ((4, 2, 3), -2, (4, 2, 3)),
        # ((4, 2, 3), -1, (4, 2, 2)),
    )

    for shape_a, dim, shape_b in cases:
        a = make_arg(shape_a)
        b = make_unique_index(shape_b, dim, shape_a[dim])
        c = make_arg(shape_b)
        yield SampleInput(a, b, c, dim)


def gather_generator(
    op: OpInfo, dtype: torch.dtype, requires_grad: bool = False, **kwargs
):
    # torch.gather(input: Tensor, dim: int, index: LongTensor)
    # * input and index tensors have same ndims.
    # * index tensors must be smaller than input tensor along all dims except specified axis.

    make_arg = partial(
        make_tensor, device="cuda", dtype=dtype, requires_grad=requires_grad
    )
    make_index = partial(
        make_tensor, device="cuda", dtype=torch.long, requires_grad=False
    )

    # a.shape, dim, b.shape
    cases = (
        ((4, 2, 3), 0, (8, 2, 3)),
        ((4, 2, 3), 1, (4, 1, 3)),
        ((4, 2, 3), 2, (4, 2, 5)),
        ((4,), 0, (8)),
        ((4,), 0, (1)),
        ((4, 1), 0, (3, 1)),
        ((4, 1), 1, (4, 5)),
        # negative dim
        ((4, 2, 3), -3, (8, 2, 3)),
        ((4, 2, 3), -2, (4, 1, 3)),
        ((4, 2, 3), -1, (4, 2, 5)),
        ((4,), -1, (8)),
        ((4,), -1, (1)),
        ((4, 1), -2, (3, 1)),
        ((4, 1), -1, (4, 5)),
        # nvfuser gather does not support broadcast non-axis dimensions
    )

    for shape_a, dim, shape_b in cases:
        a = make_arg(shape_a)
        b = make_index(shape_b, low=0, high=shape_a[dim])
        yield SampleInput(a, b, dim)


def argsort_generator(
    op: OpInfo, dtype: torch.dtype, requires_grad: bool = False, **kwargs
):
    make_arg = partial(
        make_tensor, device="cuda", dtype=dtype, requires_grad=requires_grad
    )

    # a.shape, dim
    cases = (
        (list(), 0),
        ((128,), 0),
        ((128, 7, 32), 0),
        ((128, 7, 32), 1),
        ((128, 7, 32), 2),
        ((128, 7, 32), -1),
        ((128, 7, 32), -2),
        ((128, 7, 32), -3),
    )

    for shape, dim in cases:
        a = make_arg(shape)
        for descending, stable in itertools.product([True, False], repeat=2):
            yield SampleInput(a, dim, descending, stable)


def topk_generator(
    op: OpInfo, dtype: torch.dtype, requires_grad: bool = False, **kwargs
):
    """
    Generate valid test cases for topk operation.

    Creates test tensors of various shapes and tests different combinations of:
    - k values (ensuring k <= dimension size)
    - largest/smallest selection
    - sorted/unsorted output

    Args:
        op: OpInfo object for the topk operation
        dtype: Data type for test tensors
        requires_grad: Whether tensors should require gradients

    Yields:
        SampleInput objects with valid topk parameters
    """
    make_arg = partial(
        make_tensor, device="cuda", dtype=dtype, requires_grad=requires_grad
    )

    # a.shape, dim, k_values
    cases = (
        # NOTE: aten supports topk on scalar tensor. Not sure if we would want to support this.
        # (list(), 0, [0, 1]),
        ((128,), 0, [5, 10, 64]),
        ((128, 7, 32), 0, [5, 1, 128]),
        ((128, 7, 32), 1, [5, 1, 7]),
        ((128, 7, 32), 2, [5, 1, 32]),
        ((128, 7, 32), -1, [5, 1, 32]),
        ((128, 7, 32), -2, [5, 1, 7]),
        ((128, 7, 32), -3, [5, 1, 128]),
    )

    for shape, dim, k_values in cases:
        a = make_arg(shape)
        for k in k_values:
            for largest in [True, False]:
                # NOTE: we do not test unsorted result, because reference implementation is not stable.
                yield SampleInput(a, k, dim, largest, True)


def topk_error_generator(
    op: OpInfo, dtype: torch.dtype, requires_grad: bool = False, **kwargs
):
    """
    Generate test cases that should produce errors for topk operation.

    Args:
        op: OpInfo object for the topk operation
        dtype: Data type for test tensors
        requires_grad: Whether tensors should require gradients

    Yields:
        Tuples of (SampleInput, expected_exception_type, error_message_pattern)
    """
    make_arg = partial(
        make_tensor, device="cuda", dtype=dtype, requires_grad=requires_grad
    )

    a = make_arg((128, 7, 32))

    # Out of bounds dimension access
    yield SampleInput(
        a, 3, 3, True, False
    ), RuntimeError, "Tried to access out of boundary index"
    yield SampleInput(
        a, 3, -4, True, False
    ), RuntimeError, "Tried to access out of boundary index"

    # Concretization should detect the negative K as an error
    yield SampleInput(a, -5, 1, True, False), RuntimeError, "Invalid TopK K parameter"

    #  error coming from aten fallback.
    yield SampleInput(
        a, 16, 1, True, False
    ), RuntimeError, "selected index k out of range"


def index_select_generator(
    op: OpInfo, dtype: torch.dtype, requires_grad: bool = False, **kwargs
):
    make_arg = partial(
        make_tensor, device="cuda", dtype=dtype, requires_grad=requires_grad
    )
    make_index = partial(make_tensor, device="cuda", requires_grad=False)

    # a.shape, dim, b.shape
    cases = (
        ((4, 2, 3), 0, (8)),
        ((4, 2, 3), 1, (7)),
        ((4, 2, 3), 2, (2)),
        ((4,), 0, (8)),
        ((4,), 0, (1)),
        ((4, 1), 0, (3)),
        ((4, 1), 1, (5)),
        ((1, 0, 3), 0, (8)),
    )

    for shape_a, dim, shape_b in cases:
        for index_dtype in [torch.int, torch.long]:
            a = make_arg(shape_a)
            b = make_index(shape_b, low=0, high=shape_a[dim], dtype=index_dtype)
            yield SampleInput(a, b, dim)


def index_select_error_generator(
    op: OpInfo, dtype: torch.dtype, requires_grad: bool = False, **kwargs
):
    # torch.index_select(input: Tensor, dim: int, index: LongTensor)
    # * dim is within bounds
    # * index is a 1D vector
    # * index array can't have zero elements
    make_arg = partial(
        make_tensor, device="cuda", dtype=dtype, requires_grad=requires_grad
    )
    make_index = partial(make_tensor, device="cuda", requires_grad=False)

    input_shape = (4, 2)
    index_shape = (8,)

    a = make_arg(input_shape)

    # dim, exception type, exception string
    positive_axis = (
        2,
        RuntimeError,
        "Tried to access out of boundary index 2. total index: 2",
    )
    negative_axis = (
        -3,
        RuntimeError,
        "Tried to access out of boundary index -1. total index: 2",
    )

    error_cases = [
        positive_axis,
        negative_axis,
    ]

    for dim, ex_type, ex_str in error_cases:
        b = make_index(index_shape, low=0, high=10, dtype=torch.long)
        yield SampleInput(a, b, dim), ex_type, ex_str

    # TODO add index dtype check
    # b = make_index(index_shape, low=0, high=input_shape[0], dtype=torch.float)
    # yield SampleInput(a, b, 0), RuntimeError, "index tensor can only be int or long dtype."

    # TODO add index out-of-bounds check
    # b = make_index(index_shape, low=10, high=100, dtype=torch.long)
    # yield SampleInput(a, b, 0), RuntimeError, "out of bounds index value."


def index_put_accumulate_generator(
    op: OpInfo, dtype: torch.dtype, requires_grad: bool = False, **kwargs
):
    make_arg = partial(
        make_tensor, device="cuda", dtype=dtype, requires_grad=requires_grad
    )
    make_index = partial(make_tensor, device="cuda", requires_grad=False)

    # vocab_size, hidden_size, seq_size
    cases = ((1024, 12, 300),)

    for vocab, hidden, seq in cases:
        for index_dtype in [torch.int, torch.long]:
            acc = make_arg((vocab, hidden))
            index = make_index((seq,), low=0, high=vocab, dtype=index_dtype)
            value = make_arg((seq, hidden))
            yield SampleInput(acc, index, value)


def iota_error_generator(
    op: OpInfo, dtype: torch.dtype, requires_grad: bool = False, **kwargs
):
    # torch.arange(start=0, end, step=1, dtype=None)
    # nvfuser.iota(length, start, step, dtype)
    #
    # length, start, step are not complex numbers and are finite numbers.
    # step cannot be 0

    yield SampleInput(
        make_number(torch.complex64, low=1),
        make_number(dtype, low=0),
        make_number(dtype, low=0),
        dtype,
    ), RuntimeError, "length must be integer"

    yield SampleInput(
        make_number(torch.int64, low=1),
        make_number(torch.complex64),
        make_number(dtype, low=0),
        dtype,
    ), RuntimeError, "iota: start dtype does not match specified dtype argument"

    yield SampleInput(
        make_number(torch.int64, low=1),
        make_number(dtype, low=0),
        make_number(torch.complex64),
        dtype,
    ), RuntimeError, "iota: step dtype does not match specified dtype argument"

    if is_floating_dtype(dtype):
        yield SampleInput(
            make_number(torch.int64, low=1),
            float("inf"),
            float("inf"),
            dtype,
        ), RuntimeError, "iota: length, start, step must be finite numbers."

    zero_step = torch.tensor([0], dtype=dtype).item()
    yield SampleInput(
        10, make_number(dtype), zero_step, dtype
    ), RuntimeError, "iota: step value must not equal zero."


def pad_error_generator(
    op: OpInfo, dtype: torch.dtype, requires_grad: bool = False, **kwargs
):
    # Nvfuser - fd.ops.pad(Tensor arg, std::vector<int64_t>& pad_widths, std::optional<Scalar> value)
    # Jax ----- jax.lax.pad(operand, padding_value, padding_config)
    # PyTorch - torch.nn.functional.pad(input, pad, mode='constant', value=None)
    #
    # Note: Nvfuser does not support interior (between-element) padding.
    #
    # Nvfuser errors
    # 1) Tensor arg and pad value must have the same dtype
    # 2) Number of pad widths must be at most twice the input dimension - NvFuser
    # 3) Dimension size after padding is not at least 0
    #
    # Jax and PyTorch errors
    # 1) Interior padding is non-negative
    # 2) Length of pad_widths is equal to number of operands

    make_arg = partial(
        make_tensor, device="cuda", dtype=dtype, requires_grad=requires_grad
    )

    input_shape = (2, 2)
    valid_pad_width = [1, 1, -1, 2]

    yield SampleInput(
        make_arg(input_shape),
        valid_pad_width,
        make_number(find_nonmatching_dtype(dtype)),
    ), RuntimeError, "Tensor arg and pad value must have the same dtype."

    # TODO Add better error message.
    # Dimension size after padding is not at least 0
    delete_all_pad_width = [-3, 0, 0, 0]
    yield SampleInput(
        make_arg(input_shape), delete_all_pad_width, make_number(dtype)
    ), RuntimeError, "Invalid resized domain extent"

    too_many_pad_width = [1, 1, 1, 1, 1, 1]
    yield SampleInput(
        make_arg(input_shape), too_many_pad_width, make_number(dtype)
    ), RuntimeError, "Number of pad widths must be at most twice the input dimension"

    uneven_pad_width = [1, 1, 0]
    yield SampleInput(
        make_arg(input_shape), uneven_pad_width, make_number(dtype)
    ), RuntimeError, "Invalid number of padding widths"


def permute_generator(
    op: OpInfo, dtype: torch.dtype, requires_grad: bool = False, **kwargs
):
    make_arg = partial(
        make_tensor, device="cuda", dtype=dtype, requires_grad=requires_grad
    )

    cases = (
        ((4, 3, 7, 8), (0, 1, 2, 3)),
        ((4, 3, 7, 8), (1, -2, 0, 3)),
        ((4, 3, 7, 8), (-2, 1, 0, -1)),
        ((4, 3, 7, 8), (0, 3, 1, 2)),
        ((4, 3, 7, 8), (0, -1, 1, 2)),
        ((4, 7), (1, 0)),
    )

    for shape, dims in cases:
        yield SampleInput(make_arg(shape), dims)


def permute_error_generator(
    op: OpInfo, dtype: torch.dtype, requires_grad: bool = False, **kwargs
):
    # torch.permute(input: torch.Tensor, dims: List[int])

    make_arg = partial(
        make_tensor, device="cuda", dtype=dtype, requires_grad=requires_grad
    )

    input_shape = (10, 3, 4, 4)
    # dims = dtype, duplicate, in-range

    # TODO Add dtype check.
    yield SampleInput(
        make_arg(input_shape), [0.0, 1.0, 2.0, 3.0]
    ), TypeError, "permute(): incompatible function arguments"

    # TODO Add duplicate axis check.
    yield SampleInput(
        make_arg(input_shape), [0, 1, 1, 3]
    ), RuntimeError, "duplicated dimension entries"

    # TODO Add in-range axis check.
    yield SampleInput(
        make_arg(input_shape), [0, 1, 2, 4]
    ), RuntimeError, "dims argument is out of range, expects"

    # TODO Add in-range axis check.
    yield SampleInput(
        make_arg(input_shape), [0, 1, 2, -5]
    ), RuntimeError, "dims argument is out of range, expects"

    # TODO Add missing axes check.
    # If dims list is empty, NvFuser ignores the permute operation.
    yield SampleInput(
        make_arg(input_shape), [0]
    ), RuntimeError, "argument to have the same length as input"

    # TODO Add out-of-bounds axes check.
    yield SampleInput(
        make_arg(input_shape), [0, 1, 2, 3, 4]
    ), RuntimeError, "argument to have the same length as input"


def random_dist_error_generator(
    op: OpInfo, dtype: torch.dtype, requires_grad: bool = False, **kwargs
):
    # Checking that non-supported dtypes fail
    yield SampleInput(
        make_number(torch.float),
        make_number(torch.float),
        [2, 2],
        dtype=torch_dtype_to_nvfuser_dtype(dtype),
    ), RuntimeError, "Random distributions only create floating point types"


def reduction_generator(
    op: OpInfo, dtype: torch.dtype, requires_grad: bool = False, **kwargs
):
    make_arg = partial(
        make_tensor,
        device="cuda",
        dtype=dtype,
        requires_grad=requires_grad,
        # We set low (inclusive) and high (exclusive) here to avoid values
        # whose products can otherwise become extremely large
        low=-2,
        high=3,
    )

    # shape, dim, keepdim, dtype
    cases = (
        ((4, 4), None, False, None),
        ((5,), None, True, None),
        ((5,), (0,), False, None),
        ((8, 1, 6), (1,), True, None),
        ((8, 7, 5, 1), (0, 1), True, None),
        ((8, 7, 5, 1), (1, 3), False, None),
    )

    for c in cases:
        shape, dim, keepdim, dtype = c
        yield (SampleInput(make_arg(shape), dim, keepdim, dtype=dtype))


def reduction_error_generator(
    op: OpInfo, dtype: torch.dtype, requires_grad: bool = False, **kwargs
):
    make_arg = partial(
        make_tensor,
        device="cuda",
        dtype=dtype,
        requires_grad=requires_grad,
        # We set low (inclusive) and high (exclusive) here to avoid values
        # whose products can otherwise become extremely large
        low=-2,
        high=3,
    )

    # shape
    cases = (
        (8, 1, 6),
        (8, 7, 5, 1),
    )

    # axes : List[int]
    # 1) all axis are int --- use float dtype
    # 2) all axes are unique --- duplicates
    # 3) after normalization, 0 <= axis[i] <= len(size)
    # 4) If empty tensor, then axis == 0

    int_dtype_axis = (
        lambda dims: float(dims),
        TypeError,
        "var_mean(): incompatible function arguments.",
    )
    duplicate_axis = (
        lambda dims: (0, 0, 0),
        RuntimeError,
        "Reduction axes are not unique",
    )
    lower_bound = (lambda dims: (-dims - 1,), RuntimeError, "Reduction on invalid axis")
    upper_bound = (lambda dims: (dims,), RuntimeError, "Reduction on invalid axis")
    # TODO Fix duplicate_axis, lower_bound, upper_bound
    error_cases = [int_dtype_axis]

    for shape, es in itertools.product(cases, error_cases):
        input_tensor = make_arg(shape)
        axis_fn, ex_type, ex_str = es
        yield SampleInput(input_tensor, axis_fn(len(shape))), ex_type, ex_str


def reshape_generator(
    op: OpInfo, dtype: torch.dtype, requires_grad: bool = False, **kwargs
):
    make_arg = partial(
        make_tensor, device="cuda", dtype=dtype, requires_grad=requires_grad
    )

    # TODO Add examples with negative index
    # TODO: Add zero-dim cases
    # TODO: Add strided tensor cases
    cases = (
        ((1, 19, 1, 12, 7, 1, 99), (1, 19, 1, 3, 2772)),
        ((3, 17, 80, 1), (51, 1, 2, 4, 10)),
        ((3, 17, 80, 1, 9), (51, 1, 2, 4, 10, 9)),
        ((2, 3, 4, 5), (1, 6, 1, 2, 2, 5)),
        ((22, 22, 2), (22, 11, 1, 1, 4)),
        ((37, 9, 7, 6, 10), (333, 2, 2, 3, 35)),
        ((8, 1, 1, 8, 1, 8), (8, 2, 4, 1, 8)),
        ((1, 333, 1), (1, 37, 9)),
        ((1, 333), (1, 1, 1, 111, 1, 3)),
        ((1, 27454, 1, 2), (1, 7844, 1, 7)),
        ((1, 7844, 1, 7), (1, 27454, 2)),
    )

    for input_shape, output_shape in cases:
        input_tensor = make_arg(input_shape)
        if op.name == "reshape_symbolic":
            reshaped_tensor = make_arg(output_shape)
            yield SampleInput(input_tensor, reshaped_tensor)
        else:
            yield SampleInput(input_tensor, output_shape)


def reshape_error_generator(
    op: OpInfo, dtype: torch.dtype, requires_grad: bool = False, **kwargs
):
    # torch.reshape(input: Tensor, shape: [int])

    make_arg = partial(
        make_tensor, device="cuda", dtype=dtype, requires_grad=requires_grad
    )

    input_shape = (3, 14)

    # Only a single inferred axis -1. Skip reshape_symbolic because
    # make_arg can't create a tensor with negative dimensions.
    if op.name == "reshape_constant":
        yield SampleInput(
            make_arg(input_shape), (3, -1, -1)
        ), RuntimeError, "A maximum of one value of -1"

    # Number of elements must be equal for input and output tensors
    output_shape = (3, 2, 8)
    yield SampleInput(
        make_arg(input_shape),
        (output_shape if op.name == "reshape_constant" else make_arg(output_shape)),
    ), RuntimeError, "Total element counts across view operation must match"


# TODO: add stride testing
def slice_generator(
    op: OpInfo, dtype: torch.dtype, requires_grad: bool = False, **kwargs
):
    make_arg = partial(
        make_tensor, device="cuda", dtype=dtype, requires_grad=requires_grad
    )

    # shape, start_indices, end_indices
    cases = (
        ((5, 7, 8), (1, 0, 3), (2, 6, 8)),
        ((3,), (1,), (2,)),
    )

    for shape, start_indices, end_indices in cases:
        a = make_arg(shape)
        yield SampleInput(a, start_indices=start_indices, end_indices=end_indices)


def slice_error_generator(
    op: OpInfo, dtype: torch.dtype, requires_grad: bool = False, **kwargs
):
    make_arg = partial(
        make_tensor, device="cuda", dtype=dtype, requires_grad=requires_grad
    )

    # shape
    cases = ((10, 10), (5, 5))

    check_start_indices = ErrorSample(
        {"start_indices": [-1, -2], "end_indices": [5, 5], "strides": [7, 7]},
        "Slice operation start_indices must be greater than or equal to 0.",
    )

    check_end_indices = ErrorSample(
        {"start_indices": [3, 4], "end_indices": [1, 2], "strides": [1, 1]},
        "Slice operation end_indices must be greater than or equal to start_indices.",
    )

    check_strides = ErrorSample(
        {"start_indices": [0, 0], "end_indices": [5, 5], "strides": [5, 5]},
        "nvFuser Limitation: All slice operation strides must be of const size 1.",
    )

    check_tensor_dims = ErrorSample(
        {"start_indices": [0, 0, 0], "end_indices": [4, 4, 4], "strides": [1, 1, 1]},
        "Number of tensor dimensions does not match slice dimensions!",
    )

    check_slice_dims_start = ErrorSample(
        {"start_indices": [0, 0, 0], "end_indices": [4, 4], "strides": [1, 1]},
        "Slice start_indices and strides don't match!",
    )

    check_slice_dims_end = ErrorSample(
        {"start_indices": [0, 0], "end_indices": [4, 4, 4], "strides": [1, 1]},
        "Slice indexing attribute dimensions don't match!",
    )

    check_slice_dims_stride = ErrorSample(
        {"start_indices": [0, 0], "end_indices": [4, 4], "strides": [1, 1, 1]},
        "Slice start_indices and strides don't match!",
    )

    error_cases = [
        check_start_indices,
        check_end_indices,
        check_strides,
        check_tensor_dims,
        check_slice_dims_start,
        check_slice_dims_end,
        check_slice_dims_stride,
    ]

    for shape, es in itertools.product(cases, error_cases):
        input_tensor = make_arg(shape)
        yield SampleInput(input_tensor, **es.kwargs), es.ex_type, es.ex_str


def squeeze_generator(
    op: OpInfo, dtype: torch.dtype, requires_grad: bool = False, **kwargs
):
    make_arg = partial(
        make_tensor, device="cuda", dtype=dtype, requires_grad=requires_grad
    )

    # shape, squeeze_dims
    cases = (
        ((5, 1, 1), (1, 2)),
        ((5, 1, 1), (-2, -1)),
        ((5, 1, 1), (2, 1)),
        ((5, 1, 1), (-1, -2)),
        ((1, 5, 1), (0, 2)),
        ((1, 5, 1), (-3, -1)),
        ((1, 1, 5), (0, 1)),
        ((1, 1, 5), (-3, -2)),
        ((5, 5, 5), ()),
        ((1, 1, 1), ()),
        ((1, 1, 1), (0, 1, 2)),
        ((1, 1, 1), (-3, -2, -1)),
        # No-op test cases
        # NOTE: These are skipped. We diverge from PyTorch behavior for squeeze
        # in nvFuser. Our squeeze op will throw an exception if we pass a
        # squeeze dimension that cannot be squeezed.
        # See https://github.com/NVIDIA/Fuser/pull/1717
        # ((5, 5, 5), (0, 1, 2)),
        # ((5, 5, 5), (-3, -2, -1)),
        ((), ()),
    )

    for shape, squeeze_dims in cases:
        a = make_arg(shape)
        yield SampleInput(a, squeeze_dims)


def squeeze_error_generator(
    op: OpInfo, dtype: torch.dtype, requires_grad: bool = False, **kwargs
):
    make_arg = partial(
        make_tensor, device="cuda", dtype=dtype, requires_grad=requires_grad
    )

    # shape, start_indices, end_indices
    out_of_range_cases = (
        ((5, 1, 1), (-4, -5)),  # Dims are completely outside of tensor dims
        ((5, 1, 1), (3, 4)),
        ((5, 1, 1), (-3, -4)),  # One dim in range, one dim out of range
        ((5, 1, 1), (2, 3)),
    )

    error_type = RuntimeError
    error_str = "Squeeze dim is outside of Tensor size!"
    for shape, squeeze_dims in out_of_range_cases:
        a = make_arg(shape)
        yield SampleInput(a, squeeze_dims), error_type, error_str

    # shape, start_indices, end_indices
    too_many_indices_cases = (
        ((5, 1, 1), (1, 2, 3, 4)),
        ((5, 1, 1), (-1, -2, -3, -4)),
        ((), (0,)),
        ((), (-1,)),
    )

    error_type = RuntimeError
    error_str = "The dims to squeeze must be <= the number of dims of the input tensor"
    for shape, squeeze_dims in too_many_indices_cases:
        a = make_arg(shape)
        yield SampleInput(a, squeeze_dims), error_type, error_str


def take_along_axis_generator(
    op: OpInfo, dtype: torch.dtype, requires_grad: bool = False, **kwargs
):
    make_arg = partial(
        make_tensor, device="cuda", dtype=dtype, requires_grad=requires_grad
    )
    make_index = partial(
        make_tensor, device="cuda", dtype=torch.long, requires_grad=False
    )

    # a.shape, dim, b.shape
    cases = (
        ((4, 2, 3), 0, (8, 2, 3)),
        ((4, 2, 3), 1, (4, 1, 3)),
        ((4, 2, 3), 2, (4, 2, 5)),
        ((4,), 0, (8)),
        ((4,), 0, (1)),
        ((4, 1), 0, (3, 1)),
        ((4, 1), 1, (4, 5)),
        # negative dim
        ((4, 2, 3), -3, (8, 2, 3)),
        ((4, 2, 3), -2, (4, 1, 3)),
        ((4, 2, 3), -1, (4, 2, 5)),
        ((4,), -1, (8)),
        ((4,), -1, (1)),
        ((4, 1), -2, (3, 1)),
        ((4, 1), -1, (4, 5)),
        # broadcast non-axis dimensions
        ((4, 2, 3), 0, (8, 2, 1)),
        ((4, 2, 3), 0, (8, 1, 3)),
        ((4, 2, 3), 0, (8, 2, 3)),
    )

    for shape_a, dim, shape_b in cases:
        a = make_arg(shape_a)
        b = make_index(shape_b, low=0, high=shape_a[dim])
        yield SampleInput(a, b, dim)


def take_along_axis_error_generator(
    op: OpInfo, dtype: torch.dtype, requires_grad: bool = False, **kwargs
):
    # numpy.take_along_axis(arr: Tensor, indices: LongTensor, axis: int)
    #
    # torch.take_along_dim(input: Tensor, indices: LongTensor, dim: int)
    # * If no dim argument, flatten tensors.

    make_arg = partial(
        make_tensor, device="cuda", dtype=dtype, requires_grad=requires_grad
    )
    make_index = partial(
        make_tensor, device="cuda", dtype=torch.long, requires_grad=False
    )

    input_shape = (4, 2)
    a = make_arg(input_shape)

    valid_index_shape = (3, 1)
    b = make_index(valid_index_shape, low=0, high=10, dtype=torch.long)

    # out-of-bounds axis error checks
    ex_type = RuntimeError
    ex_str = "Tensor arguments have dimension"
    positive_error_dim = 2
    negative_error_dim = -3
    yield SampleInput(a, b, positive_error_dim), ex_type, ex_str
    yield SampleInput(a, b, negative_error_dim), ex_type, ex_str

    # TODO Fix: index tensor integer dtype
    # b = make_index(valid_index_shape, low=0, high=input_shape[0], dtype=torch.float)
    # yield SampleInput(a, b, 0), RuntimeError, "index tensor can only be int or long dtype."

    # TODO Fix: out-of-bound index value
    # b = make_index(valid_index_shape, low=10, high=100, dtype=torch.long)
    # yield SampleInput(a, b, 0), RuntimeError, "out of bounds index value."

    # TODO Fix: index shape exceeds input tensor axis
    # larger_index_shape = (5, 3)
    # b = make_index(
    #    larger_index_shape, low=0, high=larger_index_shape[0], dtype=torch.long
    # )
    # yield (
    #    SampleInput(a, b, 0),
    #    RuntimeError,
    #    "Expected dimension of index tensor to be smaller than input tensor except for specified axis",
    # )

    # TODO Fix: too many dimensions in index tensor
    # dim argument must be specified. Otherwise, the tensors are flattened.
    # too_many_dims_index_shape = (3, 1, 2)
    # b = make_index(
    #    too_many_dims_index_shape,
    #    low=0,
    #    high=too_many_dims_index_shape[0],
    #    dtype=torch.long,
    # )
    # yield (
    #    SampleInput(a, b, 0),
    #    RuntimeError,
    #    "input and indices should have the same number of dimensions",
    # )


def var_mean_generator(
    op: OpInfo, dtype: torch.dtype, requires_grad: bool = False, **kwargs
):
    """torch.var_mean(input, dim=None, *, correction=1, keepdim=False)"""
    correction = (0, 1)
    samples = reduction_generator(op, dtype, requires_grad)
    for c, sample in itertools.product(correction, samples):
        a = sample.args[0]
        dim = (
            sample.args[1]
            if (len(sample.args) > 1 and sample.args[1])
            else tuple(range(a.ndim))
        )
        keepdim = sample.args[2] if len(sample.args) > 2 else False
        yield SampleInput(a, dim, correction=c, keepdim=keepdim)


def where_error_generator(
    op: OpInfo, dtype: torch.dtype, requires_grad: bool = False, **kwargs
):
    # torch.where(condition, input, other)

    make_arg = partial(
        make_tensor, device="cuda", dtype=dtype, requires_grad=requires_grad
    )

    input_shape = (2, 3, 4)
    yield SampleInput(
        make_tensor(input_shape, device="cuda", dtype=torch.float32),
        make_arg(input_shape),
        make_arg(input_shape),
    ), RuntimeError, "Condition should be of DataType Bool"


def tensor_size_error_generator(
    op: OpInfo, dtype: torch.dtype, requires_grad: bool = False, **kwargs
):
    make_arg = partial(
        make_tensor, device="cuda", dtype=dtype, requires_grad=requires_grad
    )

    check_index_beyond_num_dims = (
        {
            "tensor_shape": [2 for _ in range(0, MAX_TENSOR_DIMS)],
            "dim": MAX_TENSOR_DIMS,
        },
        RuntimeError,
        "Tried to access out of boundary index",
    )
    check_relative_index_beyond_num_dims = (
        {
            "tensor_shape": [2 for _ in range(0, MAX_TENSOR_DIMS)],
            "dim": -MAX_TENSOR_DIMS - 1,
        },
        RuntimeError,
        "Tried to access out of boundary index",
    )

    error_checks = [
        check_index_beyond_num_dims,
        check_relative_index_beyond_num_dims,
    ]

    for error_case, error_type, error_msg in error_checks:
        yield SampleInput(
            make_arg(error_case["tensor_shape"]), dim=error_case["dim"]
        ), error_type, error_msg


def vector_at_error_generator(
    op: OpInfo, dtype: torch.dtype, requires_grad: bool = False, **kwargs
):
    make_arg = partial(
        make_tensor, device="cuda", dtype=dtype, requires_grad=requires_grad
    )

    check_index_beyond_num_dims = (
        {
            "tensor_shape": [2 for _ in range(0, MAX_TENSOR_DIMS)],
            "index": MAX_TENSOR_DIMS,
        },
        RuntimeError,
        "Tried to access out of boundary index",
    )
    check_relative_index_beyond_num_dims = (
        {
            "tensor_shape": [2 for _ in range(0, MAX_TENSOR_DIMS)],
            "index": -MAX_TENSOR_DIMS - 1,
        },
        RuntimeError,
        "Tried to access out of boundary index",
    )

    error_checks = [
        check_index_beyond_num_dims,
        check_relative_index_beyond_num_dims,
    ]

    for error_case, error_type, error_msg in error_checks:
        yield SampleInput(
            make_arg(error_case["tensor_shape"]), index=error_case["index"]
        ), error_type, error_msg


def matmul_input_generator(
    op: OpInfo, dtype: torch.dtype, requires_grad: bool = False, **kwargs
):
    make_arg = partial(
        make_tensor,
        dtype=dtype,
        device="cuda",
        low=None,
        high=None,
        requires_grad=requires_grad,
    )

    B = 4
    M = 256
    N = 128
    K = 32

    shapes_a = ((K,), (M, K), (1, K), (B, M, K), (B, 1, M, K))
    shapes_b = ((K,), (K, N), (K, 1), (B, K, N))

    for shape_a, shape_b in itertools.product(shapes_a, shapes_b):
        yield SampleInput(make_arg(shape_a), make_arg(shape_b))


def linear_input_generator(
    op: OpInfo, dtype: torch.dtype, requires_grad: bool = False, **kwargs
):
    make_arg = partial(
        make_tensor,
        dtype=dtype,
        device="cuda",
        low=None,
        high=None,
        requires_grad=requires_grad,
    )

    B = 64
    M = 512
    N = 256
    K = 32

    # Cases without bias
    shapes_input = ((K), (M, K), (B, M, K), (B, 1, M, K))
    shapes_weight = ((N, K), (1, K))
    for shape_input, shape_weight in itertools.product(shapes_input, shapes_weight):
        yield SampleInput(make_arg(shape_input), make_arg(shape_weight))

    # Cases with bias
    shape_weight = (N, K)
    shapes_bias = ((N,),)
    for shape_input, shape_bias in itertools.product(shapes_input, shapes_bias):
        yield SampleInput(
            make_arg(shape_input), make_arg(shape_weight), make_arg(shape_bias)
        )


def linear_error_generator(
    op, dtype=torch.float32, requires_grad: bool = False, **kwargs
):
    make_arg = partial(
        make_tensor, device="cuda", dtype=dtype, requires_grad=requires_grad
    )
    # shapes, dim, exception type, exception string
    M = 512
    N = 256
    K = 32

    mismatched_bias_extent = (
        ((M, K), (1, K), (N)),
        RuntimeError,
        f"The expanded size of the tensor (1) must match the existing size ({N}) at non-singleton dimension 1.  Target sizes: [{M}, 1].  Tensor sizes: [{N}]",
    )

    error_cases = [mismatched_bias_extent]

    for input_shapes, ex_type, ex_str in error_cases:
        shape_input, shape_weight, shape_bias = input_shapes
        yield SampleInput(
            make_arg(shape_input), make_arg(shape_weight), make_arg(shape_bias)
        ), ex_type, ex_str


def div_input_generator(
    op: OpInfo,
    dtype: torch.dtype,
    requires_grad: bool = False,
):
    """Rescale to avoid very small denominators"""
    for sample in elementwise_binary_generator(
        op,
        dtype,
        requires_grad,
        supports_numbers=True,
        enable_small_value_testing=False,
        enable_extremal_value_testing=False,
        exclude_zero=True,
    ):
        if not is_floating_dtype(dtype):
            yield sample
            continue

        # rescale so that the denominator always has at least this modulus
        minabs = 1e-2
        numer, denom = sample.args
        denom = denom * 1e-4
        denom_abs = denom.abs()  # this is never zero because of exclude_zero=True
        denom_is_small = denom_abs < minabs
        denom_scaled_to_minabs = denom * (minabs / denom_abs)
        denom = torch.where(denom_is_small, denom_scaled_to_minabs, denom).detach()
        denom.requires_grad_(requires_grad)
        yield SampleInput(numer, denom)


def triu_input_generator(op: OpInfo, dtype: torch.dtype, requires_grad: bool = False):
    offsets = (0, 1, -1, 2, 3, -3, 1024, -1024)

    for element in elementwise_unary_generator(
        op,
        dtype,
        requires_grad,
        enable_extremal_value_testing=False,
        enable_large_value_testing=False,
        enable_small_value_testing=False,
    ):
        if element.args[0].ndim < 2:
            continue
        # to test cases where offset is not passed as an argument
        yield element
        # to test cases where offset is passed as an argument
        for offset in offsets:
            yield SampleInput(*element.args, offset)


def triu_error_generator(op: OpInfo, dtype: torch.dtype, requires_grad: bool = False):
    make_arg = partial(
        make_tensor, device="cuda", dtype=dtype, requires_grad=requires_grad
    )

    invalid_shapes = (
        (),
        (4,),
    )

    for shape in invalid_shapes:
        yield SampleInput(
            make_arg(shape),
        ), RuntimeError, f"input tensor for triu must have 2 or more dims, but got {len(shape)} dims"


def grouped_mm_input_generator(
    op: OpInfo, dtype: torch.dtype, requires_grad: bool = False, **kwargs
):
    """
    Generate valid test cases for grouped matrix multiplication.

    Args:
        op: OpInfo object for the bmm operation
        dtype: Data type for test tensors
        requires_grad: Whether tensors should require gradients
    """
    make_arg = partial(
        make_tensor,
        dtype=dtype,
        device="cuda",
        low=None,
        high=None,
        requires_grad=requires_grad,
    )

    def make_index(extent, num_groups):
        group_size = extent // num_groups
        return torch.arange(
            group_size,
            group_size * g + 1,
            group_size,
            device="cuda",
            dtype=torch.int32,
            requires_grad=False,
        )

    # TODO: expand the test when kernel restrictions are lifted
    # Test various group sizes and matrix dimensions
    configs = (
        (4, 128, 256, 64),
        (2, 32, 32, 32),
    )

    for config in configs:
        g, m, k, n = config

        # case 1: 2d x 2d
        mat1 = make_arg((m, k))
        mat2 = make_arg((k, n))
        offsets = make_index(k, g)
        yield SampleInput(mat1, mat2, offsets)
        # case 3: 2d x 3d
        mat1 = make_arg((m, k))
        mat2 = make_arg((g, k, n))
        offsets = make_index(m, g)
        yield SampleInput(mat1, mat2, offsets)
        # case 1: 3d x 2d
        mat1 = make_arg((g, m, k))
        mat2 = make_arg((k, n))
        offsets = make_index(n, g)
        yield SampleInput(mat1, mat2, offsets)


def scaled_grouped_mm_input_generator(
    op: OpInfo, dtype: torch.dtype, requires_grad: bool = False, **kwargs
):
    """
    Generate valid test cases for scaled grouped matrix multiplication.

    Args:
        op: OpInfo object for the bmm operation
        dtype: Data type for test tensors
        requires_grad: Whether tensors should require gradients
    """

    # TODO: enable mxfp8 test when backend supports it.
    make_arg = partial(
        make_tensor,
        dtype=dtype,
        device="cuda",
        low=None,
        high=None,
        requires_grad=requires_grad,
    )

    make_scale_factor = partial(
        make_tensor,
        dtype=torch.float32,
        device="cuda",
        low=None,
        high=None,
        requires_grad=False,
    )

    def make_index(extent, num_groups):
        group_size = extent // num_groups
        return torch.arange(
            group_size,
            group_size * g + 1,
            group_size,
            device="cuda",
            dtype=torch.int32,
            requires_grad=False,
        )

    # TODO: expand the test when fallback kernel restrictions are lifted
    #       currently only bf16 output is supported.
    #       there are also restrictions on the input/output shapes.
    # Test various group sizes and matrix dimensions
    # configs: list(g, m, k, n, output_dtype)
    configs = (
        (4, 128, 256, 64, torch.bfloat16),
        (2, 32, 32, 32, torch.bfloat16),
    )

    # TODO: Enable mxfp8 test when backend supports it.
    for config in configs:
        g, m, k, n, dtype = config
        # case 1: 2d x 2d
        mat1 = make_arg((m, k))
        mat2 = make_arg((k, n))
        scale1 = make_scale_factor((g, m, 1))
        scale2 = make_scale_factor((g, 1, n))
        offsets = make_index(k, g)
        yield SampleInput(mat1, mat2, offsets, scale1, scale2, None, None, None, dtype)
        # case 3: 2d x 3d
        mat1 = make_arg((m, k))
        mat2 = make_arg((g, k, n))
        scale1 = make_scale_factor((m, 1))
        scale2 = make_scale_factor((g, 1, n))
        offsets = make_index(m, g)
        yield SampleInput(mat1, mat2, offsets, scale1, scale2, None, None, None, dtype)
        # case 1: 3d x 2d
        mat1 = make_arg((g, m, k))
        mat2 = make_arg((k, n))
        offsets = make_index(n, g)
        scale1 = make_scale_factor((g, m, 1))
        scale2 = make_scale_factor((1, n))
        yield SampleInput(mat1, mat2, offsets, scale1, scale2, None, None, None, dtype)


###############################################################################
#
# NOTE: quantization taken from torch testing. Since this is using aten backend
#
###############################################################################
# largest power of 2 representable in `torch.float8_e4m3fn`
F8E4M3_LARGEST_POW2 = 8
# max value of `torch.float8_e4m3fn` (448)
F8E4M3_MAX_VAL = torch.finfo(torch.float8_e4m3fn).max
# exponent bias of `torch.float8_e8m0fnu`
F8E8M0_EXP_BIAS = 127


def data_to_mxfp8_scale(x, block_size):
    # simple implementation of https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
    # section 6.3, not all edge cases (such as NaN) are handled/tested
    orig_shape = x.shape
    x = x.reshape(-1, block_size)
    max_abs = torch.amax(torch.abs(x), 1)
    largest_p2_lt_max_abs = torch.floor(torch.log2(max_abs))
    scale_e8m0_unbiased = largest_p2_lt_max_abs - F8E4M3_LARGEST_POW2
    scale_e8m0_unbiased = torch.clamp(
        scale_e8m0_unbiased, -1 * F8E8M0_EXP_BIAS, F8E8M0_EXP_BIAS
    )
    scale_e8m0_biased = scale_e8m0_unbiased + F8E8M0_EXP_BIAS
    scale_e8m0_biased = scale_e8m0_biased.to(torch.uint8)
    scale_e8m0_biased = scale_e8m0_biased.view(torch.float8_e8m0fnu)
    return scale_e8m0_biased.reshape(orig_shape[0], -1)


def data_to_mxfp8(x, block_size):
    x_scale = data_to_mxfp8_scale(x, block_size)
    K = x.shape[1]
    max_val = F8E4M3_MAX_VAL
    min_val = -1 * max_val
    x = (x.reshape(-1, block_size) / x_scale.reshape(-1, 1).float()).reshape(-1, K)
    x = x.clamp(min=min_val, max=max_val).to(torch.float8_e4m3fn)
    return x, x_scale


def scaled_mm_input_generator(
    op: OpInfo, dtype: torch.dtype, requires_grad: bool = False, **kwargs
):
    """
    Generate valid test cases for scaled matrix multiplication.

    Args:
        op: OpInfo object for the bmm operation
        dtype: Data type for test tensors
        requires_grad: Whether tensors should require gradients
    """

    # TODO: enable mxfp8 test when backend supports it.
    make_arg = partial(
        make_tensor,
        dtype=torch.float32,
        device="cuda",
        low=None,
        high=None,
        requires_grad=requires_grad,
    )

    # TODO: expand the test when fallback kernel restrictions are lifted
    #       currently only bf16 output is supported.
    #       there are also restrictions on the input/output shapes.
    # Test various group sizes and matrix dimensions
    # configs: list(m, k, n, output_dtype)
    configs = (
        (128, 256, 512, torch.bfloat16),
        (128, 128, 128, torch.bfloat16),
    )

    # TODO: support nvfp4
    assert dtype == torch.float8_e4m3fn
    quantization = partial(data_to_mxfp8, block_size=32)

    for config in configs:
        m, k, n, dtype = config
        mat1_ref = make_arg((m, k))
        mat2_ref = make_arg((n, k))
        mat1, scale1 = quantization(mat1_ref)
        mat2, scale2 = quantization(mat2_ref)
        yield SampleInput(mat1, mat2.t(), scale1, scale2, None, None, None, dtype)
