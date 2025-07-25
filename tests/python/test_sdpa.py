# SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Owner(s): ["module: nvfuser"]

import itertools
import math
import pytest
import torch
import torch.nn.functional as F
from functools import partial
from nvfuser import FusionDefinition, DataType, FusionCache
from python.utils import NVFuserTest, is_pre_ampere, define_sdpa_rng_state


@pytest.mark.skipif(
    is_pre_ampere(),
    reason="Flash Attention is only supported on Ampere and newer devices.",
)
class TestSdpa(NVFuserTest):
    def test_softmax_logsumexp(self):
        def fusion_func(fd: FusionDefinition) -> None:
            q = fd.define_tensor(
                shape=[-1, -1, -1, -1],
                dtype=DataType.BFloat16,
            )
            k = fd.define_tensor(
                shape=[-1, -1, -1, -1],
                dtype=DataType.BFloat16,
            )
            v = fd.define_tensor(
                shape=[-1, -1, -1, -1],
                dtype=DataType.BFloat16,
            )
            (
                _,
                lse,
                *_,
            ) = fd.ops.sdpfa_fwd(q, k, v, dropout_p=None, is_causal=None, scale=None)
            fd.add_output(lse)

        n, h, l, s, e = 1, 1, 4, 4, 2
        inputs = [
            torch.ones((n, h, l, e), dtype=torch.bfloat16, device="cuda"),
            torch.ones((n, h, s, e), dtype=torch.bfloat16, device="cuda"),
            torch.ones((n, h, s, e), dtype=torch.bfloat16, device="cuda"),
        ]

        from torch.nn.attention import SDPBackend, sdpa_kernel

        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            nvf_out, _ = self.exec_nvfuser(
                fusion_func,
                inputs,
            )
        # Ignoring size-1 dimensions, `q @ k^T / sqrt(e)` generates a `l`x`s`
        # matrix full of `sqrt(e)`s.  Therefore, the logsumexp of each row is
        # expected to be log(exp(sqrt(e)) * s) = log(s) + sqrt(e).
        torch.testing.assert_close(
            nvf_out[0].cpu(), torch.full((n, h, l), math.log(s) + e**0.5)
        )

    def test_sdpa_fwd(self):
        def fusion_func(
            fd: FusionDefinition, has_dropout: bool, has_causal: bool, has_scale: bool
        ):
            q = fd.define_tensor(
                shape=[-1, -1, -1, -1],
                contiguity=[True, True, True, True],
                dtype=DataType.BFloat16,
                is_cpu=False,
            )
            k = fd.define_tensor(
                shape=[-1, -1, -1, -1],
                contiguity=[True, True, True, True],
                dtype=DataType.BFloat16,
                is_cpu=False,
            )
            v = fd.define_tensor(
                shape=[-1, -1, -1, -1],
                contiguity=[True, True, True, True],
                dtype=DataType.BFloat16,
                is_cpu=False,
            )
            dropout_p, is_causal, scale = None, None, None
            if has_dropout:
                dropout_p = fd.define_scalar(value=None, dtype=DataType.Double)
            if has_causal:
                is_causal = fd.define_scalar(value=None, dtype=DataType.Bool)
            if has_scale:
                scale = fd.define_scalar(value=None, dtype=DataType.Double)
            attn, *_ = fd.ops.sdpfa_fwd(q, k, v, dropout_p, is_causal, scale)
            fd.add_output(attn)

        N, H, L, S, E = 4, 8, 16, 16, 8
        qkv = [
            torch.randn((N, H, L, E), dtype=torch.bfloat16, device="cuda:0"),
            torch.randn((N, H, S, E), dtype=torch.bfloat16, device="cuda:0"),
            torch.randn((N, H, S, E), dtype=torch.bfloat16, device="cuda:0"),
        ]

        dropout_vals = [None, 0.0, 0.2]
        is_causal_vals = [None, True, False]
        scale_vals = [None, 1 / E**0.5, 1e-3]
        # TODO: Try to move this to pytest_ops.py. Currently, it does not work since the API between nvFuser and torch differs.
        for dropout_p, is_causal, scale in itertools.product(
            dropout_vals, is_causal_vals, scale_vals
        ):
            with self.subTest(dropout_p=dropout_p, is_causal=is_causal, scale=scale):
                from torch.nn.attention import SDPBackend, sdpa_kernel

                # Reset the FusionCache or the fusion would not recompile for all subtests, failing checks in exec_nvfuser.
                FusionCache.reset()
                has_dropout = True if dropout_p is not None else False
                has_causal = True if is_causal is not None else False
                has_scale = True if scale is not None else False
                inputs = [*qkv]
                for param in [dropout_p, is_causal, scale]:
                    if param is not None:
                        inputs.append(param)
                nvf_out, _ = self.exec_nvfuser(
                    partial(
                        fusion_func,
                        has_dropout=has_dropout,
                        has_causal=has_causal,
                        has_scale=has_scale,
                    ),
                    inputs,
                )

                # Torch does not accept NoneType dropout_p, is_causal.
                dropout_p = 0.0 if dropout_p is None else dropout_p
                is_causal = False if is_causal is None else is_causal

                with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                    torch.manual_seed(0)
                    ref_out = F.scaled_dot_product_attention(
                        *qkv, dropout_p=dropout_p, is_causal=is_causal, scale=scale
                    )
                torch.testing.assert_close(nvf_out[0], ref_out)

    def test_sdpa_bwd(self):
        N, H, L, S, E = 4, 8, 16, 16, 8

        grad_output = torch.randn((N, H, L, E), dtype=torch.bfloat16, device="cuda:0")
        q = torch.randn((N, H, L, E), dtype=torch.bfloat16, device="cuda:0")
        k = torch.randn((N, H, S, E), dtype=torch.bfloat16, device="cuda:0")
        v = torch.randn((N, H, S, E), dtype=torch.bfloat16, device="cuda:0")

        dropout_vals = [None, 0.0, 0.2]
        is_causal_vals = [None, True, False]
        scale_vals = [None, 1 / E**0.5, 1e-3]

        def fusion_func(
            fd: FusionDefinition, has_dropout: bool, has_causal: bool, has_scale: bool
        ):
            grad_output = fd.define_tensor(
                shape=[-1, -1, -1, -1],
                contiguity=[True, True, True, True],
                dtype=DataType.BFloat16,
                is_cpu=False,
            )
            q = fd.define_tensor(
                shape=[-1, -1, -1, -1],
                contiguity=[True, True, True, True],
                dtype=DataType.BFloat16,
                is_cpu=False,
            )
            k = fd.define_tensor(
                shape=[-1, -1, -1, -1],
                contiguity=[True, True, True, True],
                dtype=DataType.BFloat16,
                is_cpu=False,
            )
            v = fd.define_tensor(
                shape=[-1, -1, -1, -1],
                contiguity=[True, True, True, True],
                dtype=DataType.BFloat16,
                is_cpu=False,
            )
            output = fd.define_tensor(
                shape=[-1, -1, -1, -1],
                contiguity=[True, True, True, True],
                dtype=DataType.BFloat16,
                is_cpu=False,
            )
            log_sumexp = fd.define_tensor(
                shape=[-1, -1, -1],
                contiguity=[True, True, True],
                dtype=DataType.Float,
                is_cpu=False,
            )
            philox_seed, philox_offset = define_sdpa_rng_state(fd)

            dropout_p, is_causal, scale = None, None, None
            if has_dropout:
                dropout_p = fd.define_scalar(value=None, dtype=DataType.Double)
            if has_causal:
                is_causal = fd.define_scalar(value=None, dtype=DataType.Bool)
            if has_scale:
                scale = fd.define_scalar(value=None, dtype=DataType.Double)

            grad_query, grad_key, grad_value = fd.ops.sdpfa_bwd(
                grad_output,
                q,
                k,
                v,
                output,
                log_sumexp,
                dropout_p,
                is_causal,
                philox_seed,
                philox_offset,
                scale,
            )
            fd.add_output(grad_query)
            fd.add_output(grad_key)
            fd.add_output(grad_value)

        for dropout_p, is_causal, scale in itertools.product(
            dropout_vals, is_causal_vals, scale_vals
        ):
            with self.subTest(dropout_p=dropout_p, is_causal=is_causal, scale=scale):
                # Torch does not accept NoneType dropout_p, is_causal.
                at_dropout_p = 0.0 if dropout_p is None else dropout_p
                at_is_causal = False if is_causal is None else is_causal

                (
                    output,
                    log_sumexp,
                    cum_seq_q,
                    cum_seq_k,
                    query_seq_len,
                    key_seq_len,
                    philox_seed,
                    philox_offset,
                    _,
                ) = torch.ops.aten._scaled_dot_product_flash_attention(
                    q,
                    k,
                    v,
                    at_dropout_p,
                    at_is_causal,
                    return_debug_mask=False,
                    scale=scale,
                )
                ref_grad = torch.ops.aten._scaled_dot_product_flash_attention_backward(
                    grad_output,
                    q,
                    k,
                    v,
                    output,
                    log_sumexp,
                    cum_seq_q,
                    cum_seq_k,
                    query_seq_len,
                    key_seq_len,
                    at_dropout_p,
                    at_is_causal,
                    philox_seed,
                    philox_offset,
                    scale=scale,
                )

                # Reset the FusionCache or the fusion would not recompile for all subtests, failing checks in exec_nvfuser.
                FusionCache.reset()
                has_dropout = True if dropout_p is not None else False
                has_causal = True if is_causal is not None else False
                has_scale = True if scale is not None else False

                inputs = [
                    grad_output,
                    q,
                    k,
                    v,
                    output,
                    log_sumexp,
                    philox_seed,
                    philox_offset,
                ]
                for param in [dropout_p, is_causal, scale]:
                    if param is not None:
                        inputs.append(param)

                nvf_out, _ = self.exec_nvfuser(
                    partial(
                        fusion_func,
                        has_dropout=has_dropout,
                        has_causal=has_causal,
                        has_scale=has_scale,
                    ),
                    inputs,
                )
                torch.testing.assert_close(nvf_out[0], ref_grad[0])
                torch.testing.assert_close(nvf_out[1], ref_grad[1])
                torch.testing.assert_close(nvf_out[2], ref_grad[2])

    def test_sdpa_fwd_bwd(self):
        N, H, L, S, E = 4, 8, 16, 16, 8

        dropout_vals = [None, 0.0, 0.2]
        is_causal_vals = [None, True, False]
        scale_vals = [None, 1 / E**0.5, 1e-3]

        def fusion_func(
            fd: FusionDefinition, has_dropout: bool, has_causal: bool, has_scale: bool
        ):
            q = fd.define_tensor(
                shape=[-1, -1, -1, -1],
                contiguity=[True, True, True, True],
                dtype=DataType.BFloat16,
                is_cpu=False,
            )
            k = fd.define_tensor(
                shape=[-1, -1, -1, -1],
                contiguity=[True, True, True, True],
                dtype=DataType.BFloat16,
                is_cpu=False,
            )
            v = fd.define_tensor(
                shape=[-1, -1, -1, -1],
                contiguity=[True, True, True, True],
                dtype=DataType.BFloat16,
                is_cpu=False,
            )
            grad_out = fd.define_tensor(
                shape=[-1, -1, -1, -1],
                contiguity=[True, True, True, True],
                dtype=DataType.BFloat16,
                is_cpu=False,
            )

            dropout_p, is_causal, scale = None, None, None
            if has_dropout:
                dropout_p = fd.define_scalar(value=None, dtype=DataType.Double)
            if has_causal:
                is_causal = fd.define_scalar(value=None, dtype=DataType.Bool)
            if has_scale:
                scale = fd.define_scalar(value=None, dtype=DataType.Double)

            output, log_sumexp, philox_seed, philox_offset = fd.ops.sdpfa_fwd(
                q, k, v, dropout_p, is_causal, scale
            )
            grad_query, grad_key, grad_value = fd.ops.sdpfa_bwd(
                grad_out,
                q,
                k,
                v,
                output,
                log_sumexp,
                dropout_p,
                is_causal,
                philox_seed,
                philox_offset,
                scale,
            )

            fd.add_output(output)
            fd.add_output(grad_query)
            fd.add_output(grad_key)
            fd.add_output(grad_value)

        for dropout_p, is_causal, scale in itertools.product(
            dropout_vals, is_causal_vals, scale_vals
        ):
            with self.subTest(dropout_p=dropout_p, is_causal=is_causal, scale=scale):
                from torch.nn.attention import SDPBackend, sdpa_kernel

                q = torch.randn(
                    (N, H, L, E),
                    dtype=torch.bfloat16,
                    device="cuda:0",
                    requires_grad=True,
                )
                k = torch.randn(
                    (N, H, S, E),
                    dtype=torch.bfloat16,
                    device="cuda:0",
                    requires_grad=True,
                )
                v = torch.randn(
                    (N, H, S, E),
                    dtype=torch.bfloat16,
                    device="cuda:0",
                    requires_grad=True,
                )
                grad_output = torch.randn(
                    (N, H, L, E), dtype=torch.bfloat16, device="cuda:0"
                )

                # Reset the FusionCache or the fusion would not recompile for all subtests, failing checks in exec_nvfuser.
                FusionCache.reset()
                has_dropout = True if dropout_p is not None else False
                has_causal = True if is_causal is not None else False
                has_scale = True if scale is not None else False

                inputs = [q, k, v, grad_output]
                for param in [dropout_p, is_causal, scale]:
                    if param is not None:
                        inputs.append(param)

                # Torch does not accept NoneType dropout_p, is_causal.
                dropout_p = 0.0 if dropout_p is None else dropout_p
                is_causal = False if is_causal is None else is_causal

                with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                    torch.manual_seed(0)
                    ref_out = F.scaled_dot_product_attention(
                        q, k, v, dropout_p=dropout_p, is_causal=is_causal, scale=scale
                    )
                    ref_out.backward(grad_output)

                nvf_out, _ = self.exec_nvfuser(
                    partial(
                        fusion_func,
                        has_dropout=has_dropout,
                        has_causal=has_causal,
                        has_scale=has_scale,
                    ),
                    inputs,
                )
                torch.testing.assert_close(nvf_out[0], ref_out)
                torch.testing.assert_close(nvf_out[1], q.grad)
                torch.testing.assert_close(nvf_out[2], k.grad)
                torch.testing.assert_close(nvf_out[3], v.grad)
