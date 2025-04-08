# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import torch

import fixtures
import nvfuser
from nvfuser import DataType, FusionDefinition

multidevice_test = fixtures.multidevice_test


@pytest.mark.mpi
def test_linear(multidevice_test):
    class Model(FusionDefinition):
        def __init__(self, num_devices, batch, sequence, hidden):
            super().__init__()
            self._num_devices = num_devices
            self._batch = batch
            self._sequence = sequence
            self._hidden = hidden

        def definition(self):
            d, b, s, e = self._num_devices, self._batch, self._sequence, self._hidden
            self.inp = self.define_tensor([b, s, e])
            self.weight = self.define_tensor([d, e, e], contiguity=True)
            self.bias = self.define_tensor([d, e], contiguity=True)
            out = self.ops.linear(self.inp, self.weight, self.bias)
            self.add_output(out)

        def multidevice_schedule(self):
            mesh = nvfuser.DeviceMesh(range(self._num_devices))
            for t in [self.inp, self.weight, self.bias]:
                self.sched._set_device_mesh(t, mesh)
            for t in [self.weight, self.bias]:
                self.sched.parallelize(t, 0, nvfuser.ParallelType.mesh_x)

    d = multidevice_test.size
    rank = multidevice_test.rank

    torch.cuda.set_device(multidevice_test.local_rank)

    b, s, e = 2, 1024, 768
    inp_tensor = torch.randn(b, s, e, device="cuda")
    unsharded_weight_tensor = torch.randn(d * e, e, device="cuda")
    weight_tensor = unsharded_weight_tensor.view([d, e, e])[rank : rank + 1]
    unsharded_bias_tensor = torch.randn(d * e, device="cuda")
    bias_tensor = unsharded_bias_tensor.view([d, e])[rank : rank + 1]

    fd = Model(d, b, s, e)
    (out_tensor,), (out_sharding,) = fd.execute(
        [inp_tensor, weight_tensor, bias_tensor]
    )

    # [b, s, d*e]
    unsharded_out_tensor = torch.nn.functional.linear(
        inp_tensor, unsharded_weight_tensor, unsharded_bias_tensor
    )
    expected_out_tensor = unsharded_out_tensor.view([b, s, d, e]).permute(2, 0, 1, 3)[
        rank : rank + 1
    ]
    # rtol is the same as the default for fp32. atol is slightly increased.
    assert out_sharding.axis_sharded_on(nvfuser.ParallelType.mesh_x) == 0
    torch.testing.assert_close(out_tensor, expected_out_tensor, rtol=1.3e-6, atol=1e-3)


@pytest.mark.mpi
def test_linear_loop_split(multidevice_test):
    d = multidevice_test.size
    mesh = nvfuser.DeviceMesh(range(d))
    e = 768

    class Model(FusionDefinition):
        def definition(self):
            self.inp = self.define_tensor([-1, -1, e])
            self.weight = self.define_tensor([d * e, e])
            self.bias = self.define_tensor([d * e])
            self.out = self.ops.linear(self.inp, self.weight, self.bias)
            self.add_output(self.out)

        def multidevice_schedule(self):
            for t in [self.inp, self.weight, self.bias, self.out]:
                self.sched._set_device_mesh(t, mesh)

            # Shard N for weight (N, K) and bias (N)
            for t in [self.weight, self.bias]:
                self.sched.split(t, 0, d, False)
                self.sched.parallelize(t, 0, nvfuser.ParallelType.mesh_x)
                self.sched.set_allocation_as_loop(t)

            # Output of linear: {.., i{M}, i{N}, r{K}}
            # Shard N -> axis(-2)
            self.sched.split(self.out, -2, d, False)
            self.sched.parallelize(self.out, -3, nvfuser.ParallelType.mesh_x)
            self.sched.set_allocation_as_loop(self.out)

    torch.cuda.set_device(multidevice_test.local_rank)

    b, s = 2, 1024
    inp_tensor = torch.randn(b, s, e, device="cuda")
    unsharded_weight_tensor = torch.randn(d * e, e)
    sharded_weight_tensor = multidevice_test.shard_tensor(
        unsharded_weight_tensor, 0, mesh
    )
    unsharded_bias_tensor = torch.randn(d * e)
    sharded_bias_tensor = multidevice_test.shard_tensor(unsharded_bias_tensor, 0, mesh)

    fd = Model()
    (out_tensor,), _ = fd.execute(
        [inp_tensor, sharded_weight_tensor, sharded_bias_tensor]
    )

    # [b, s, d*e]
    unsharded_out_tensor = torch.nn.functional.linear(
        inp_tensor.cpu(), unsharded_weight_tensor, unsharded_bias_tensor
    )
    expected_out_tensor = multidevice_test.shard_tensor(unsharded_out_tensor, -1, mesh)
    # rtol is the same as the default for fp32. atol is slightly increased.
    torch.testing.assert_close(out_tensor, expected_out_tensor, rtol=1.3e-6, atol=1e-3)


@pytest.mark.mpi
def test_matmul_allreduce(multidevice_test):
    d, b, s, e = multidevice_test.size, 1, 4, 8

    class Model(FusionDefinition):
        def definition(self) -> None:
            # A pattern appeared in the backprop of the first linear layer in
            # Transformer's MLP.
            self.out_grad = self.define_tensor(
                [d, b * s, e], contiguity=True, dtype=DataType.Half
            )
            self.weight = self.define_tensor(
                [d, e, e], contiguity=True, dtype=DataType.Half
            )
            in_grad = self.ops.matmul(self.out_grad, self.weight)
            in_grad = self.ops.sum(in_grad, [0])
            in_grad = self.ops.reshape(in_grad, [b, s, e])
            in_grad = self.ops.cast(in_grad, dtype=DataType.Float)
            self.add_output(in_grad)

        def multidevice_schedule(self) -> None:
            mesh = nvfuser.DeviceMesh(range(d))
            for t in [self.out_grad, self.weight]:
                self.sched._set_device_mesh(t, mesh)
                self.sched.parallelize(t, 0, nvfuser.ParallelType.mesh_x)

    rank = multidevice_test.rank

    torch.cuda.set_device(multidevice_test.local_rank)

    unsharded_out_grad = torch.randn(b * s, d * e, dtype=torch.half, device="cpu")
    unsharded_weight = torch.randn(d * e, e, dtype=torch.half, device="cpu")
    expected_in_grad = (
        (unsharded_out_grad @ unsharded_weight).view([b, s, e]).to(torch.float32)
    )

    out_grad = (
        unsharded_out_grad.view([b * s, d, e])
        .permute([1, 0, 2])
        .contiguous()[rank : rank + 1]
    )
    weight = unsharded_weight.view([d, e, e])[rank : rank + 1]

    fd = Model()
    (in_grad,), _ = fd.execute([out_grad.cuda(), weight.cuda()])
    # Use the default rtol for half because the output, although being float32,
    # is a straight cast from half.
    torch.testing.assert_close(in_grad.cpu(), expected_in_grad, rtol=1e-3, atol=1e-2)


@pytest.mark.mpi
def test_matmul_loop_split(multidevice_test):
    d = multidevice_test.size
    mesh = nvfuser.DeviceMesh(range(d))
    e = 768

    class Model(FusionDefinition):
        def definition(self):
            self.inp = self.define_tensor([-1, -1, e])
            self.weight = self.define_tensor([e, d * e])
            self.out = self.ops.matmul(self.inp, self.weight)
            self.add_output(self.out)

        def multidevice_schedule(self):
            for t in [self.inp, self.weight, self.out]:
                self.sched._set_device_mesh(t, mesh)

            # Shard N for weight (K, N)
            self.sched.split(self.weight, -1, d, False)
            self.sched.parallelize(self.weight, -2, nvfuser.ParallelType.mesh_x)
            self.sched.set_allocation_as_loop(self.weight)

            # Output of linear: {.., i{M}, i{N}, r{K}}
            # Shard N -> axis(-2)
            self.sched.split(self.out, -2, d, False)
            self.sched.parallelize(self.out, -3, nvfuser.ParallelType.mesh_x)
            self.sched.set_allocation_as_loop(self.out)

    torch.cuda.set_device(multidevice_test.local_rank)

    b, s = 2, 1024
    inp_tensor = torch.randn(b, s, e, device="cuda")
    unsharded_weight_tensor = torch.randn(e, d * e)
    sharded_weight_tensor = multidevice_test.shard_tensor(
        unsharded_weight_tensor, -1, mesh
    )

    fd = Model()
    (out_tensor,), _ = fd.execute([inp_tensor, sharded_weight_tensor])

    # [b, s, d*e]
    unsharded_out_tensor = torch.matmul(inp_tensor.cpu(), unsharded_weight_tensor)
    expected_out_tensor = multidevice_test.shard_tensor(unsharded_out_tensor, -1, mesh)
    # rtol is the same as the default for fp32. atol is slightly increased.
    torch.testing.assert_close(
        out_tensor, expected_out_tensor.squeeze(0), rtol=1.3e-6, atol=1e-3
    )


@pytest.mark.mpi
def test_matmul_allreduce_loop_split(multidevice_test):
    d = multidevice_test.size
    mesh = nvfuser.DeviceMesh(range(d))
    e = 8

    class Model(FusionDefinition):
        def definition(self) -> None:
            self.inp = self.define_tensor(
                [-1, d * e], contiguity=True, dtype=DataType.Half
            )
            self.weight = self.define_tensor(
                [d * e, e], contiguity=True, dtype=DataType.Half
            )
            self.out = self.ops.matmul(self.inp, self.weight)
            self.add_output(self.out)

        def multidevice_schedule(self) -> None:
            for t in [self.inp, self.weight, self.out]:
                self.sched._set_device_mesh(t, mesh)

            # Shard K for inp (M, K)
            self.sched.split(self.inp, -1, d, False)
            self.sched.parallelize(self.inp, -2, nvfuser.ParallelType.mesh_x)
            self.sched.set_allocation_as_loop(self.inp)

            # Shard K for weight (K, N)
            self.sched.split(self.weight, 0, d, False)
            self.sched.parallelize(self.weight, 0, nvfuser.ParallelType.mesh_x)
            self.sched.set_allocation_as_loop(self.weight)

            # [i{M}, i{N}, r{K}]
            self.sched.split(self.out, -1, d, False)
            # [i{M}, i{N}, r{d}, r{K//d}]
            self.local_out = self.sched.rfactor(self.out, dims=[-1])
            # local_out = [i{M}, i{N}, i{d}, r{K//d}]
            # out = [i{M}, i{N}, r{d}]
            self.sched._set_device_mesh(self.local_out, mesh)
            self.sched.parallelize(self.local_out, -2, nvfuser.ParallelType.mesh_x)

    torch.cuda.set_device(multidevice_test.local_rank)

    b, s = 1, 4
    unsharded_inp = torch.randn(b * s, d * e, dtype=torch.half)
    unsharded_weight = torch.randn(d * e, e, dtype=torch.half)
    sharded_inp = multidevice_test.shard_tensor(unsharded_inp, -1, mesh)
    sharded_weight = multidevice_test.shard_tensor(unsharded_weight, 0, mesh)

    expected_out = torch.matmul(unsharded_inp, unsharded_weight)

    fd = Model()
    (out,), _ = fd.execute([sharded_inp, sharded_weight])

    torch.testing.assert_close(out.cpu(), expected_out, rtol=1e-3, atol=1e-2)
