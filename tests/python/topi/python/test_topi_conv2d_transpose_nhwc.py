# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Test code for transposed convolution."""
import numpy as np
import tvm
from tvm import te
from tvm import topi
import tvm.topi.testing
from tvm.contrib.pickle_memoize import memoize
from tvm.topi.utils import get_const_tuple

import tvm.testing


def verify_conv2d_transpose_nchw(
    batch, in_channel, in_size, num_filter, kernel, stride, padding, output_padding
):
    in_height, in_width = in_size
    kernel_height, kernel_width = kernel
    stride_height, stride_width = stride
    pad_top, pad_left, pad_bottom, pad_right = padding

    A = te.placeholder((batch, in_height, in_width, in_channel), name="A")
    W = te.placeholder((kernel_height, kernel_width, in_channel, num_filter), name="W")

    a_shape = get_const_tuple(A.shape)
    w_shape = get_const_tuple(W.shape)
    dtype = A.dtype

    @memoize("topi.tests.test_topi_conv2d_transpose.verify_conv2d_transpose_nhwc")
    def get_ref_data():
        a_np = np.random.uniform(size=a_shape).astype(dtype)
        w_np = np.random.uniform(size=w_shape).astype(dtype)

        a_np_trans = a_np.transpose([0, 3, 1, 2])
        w_np_trans = w_np.transpose([2, 3, 0, 1])
        b_np_trans = tvm.topi.testing.conv2d_transpose_nchw_python(
            a_np_trans, w_np_trans, stride, padding, output_padding
        )
        b_np = b_np_trans.transpose([0, 2, 3, 1])

        return a_np, w_np, b_np

    a_np, w_np, b_np = get_ref_data()

    def check(device, ctx):
        B = topi.nn.conv2d_transpose_nhwc(
            A,
            W,
            [stride_height, stride_width],
            [pad_top, pad_left, pad_bottom, pad_right],
            A.dtype,
            output_padding,
        )
        s = tvm.te.create_schedule([B.op])
        a = tvm.nd.array(a_np, ctx)
        w = tvm.nd.array(w_np, ctx)
        b = tvm.nd.array(np.zeros(get_const_tuple(B.shape), dtype=B.dtype), ctx)

        func = tvm.build(s, [A, W, B], device)
        func(a, w, b)
        tvm.testing.assert_allclose(b.asnumpy(), b_np, rtol=1e-5)

    check("llvm", tvm.cpu(0))


def test_conv2d_transpose_nhwc():
    verify_conv2d_transpose_nchw(1, 64,  (32, 32), 3,  (4, 4), (2, 2), (1, 1, 1, 1), (0, 0))
    verify_conv2d_transpose_nchw(1, 128, (16, 16), 64, (4, 4), (2, 2), (1, 1, 1, 1), (0, 0))
    verify_conv2d_transpose_nchw(1, 256, (8, 8), 128,  (4, 4), (2, 2), (1, 1, 1, 1), (0, 0))
    verify_conv2d_transpose_nchw(1, 512, (4, 4), 256,  (4, 4), (2, 2), (1, 1, 1, 1), (0, 0))


if __name__ == "__main__":
    test_conv2d_transpose_nhwc()

