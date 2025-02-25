import numpy as np
import math

import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
from neuronxcc.nki import baremetal
import neuronxcc.nki.compiler as compiler



"""
A convolution kernel that you need to implement.

Parameters:
    X: the input tensor
    W: the weights of the convolution filters.
    bias: the biases of the convolution filters.

expect: X.shape == [batch_size, in_channels, input_height, input_width]
expect: W.shape == [out_channels, in_channels, filter_height, filter_width]
expect: bias.shape == [out_channels]
expect: filter_height == filter_width
expect: input_channels % 128 == 0
expect: output_channels % 128 == 0

out_height = input_height - filter_height + 1
out_width = input_width - filter_width + 1

out_pool_height = out_height
out_pool_width = out_width

The shape of the output should be [batch_size, out_channels, out_pool_height, out_pool_width]

"""

@nki.jit
def conv2d(X, W, bias):

    batch_size, in_channels, input_height, input_width = X.shape
    out_channels, in_channels_, filter_height, filter_width = W.shape
    out_channels_ = bias.shape[0]

    assert (
        in_channels_ == in_channels and out_channels_ == out_channels
    ), f"Shape mismatch. {in_channels}, {in_channels_}, {out_channels}, {out_channels_}"

    out_height = input_height - filter_height + 1
    out_width = input_width - filter_width + 1

    out_pool_height = out_height
    out_pool_width = out_width

    # Can assume multiple of 128 to avoid using mask
    assert in_channels % 128 == 0

    # Can assume one PSUM bank can at least fit one row of the pixels
    assert nl.tile_size.gemm_moving_fmax >= out_width

    # Initialize output array
    X_out = nl.ndarray(
        shape=(batch_size, out_channels, out_pool_height, out_pool_width),
        dtype=X.dtype,
        buffer=nl.hbm,
    )

    c_in_pmax = nl.tile_size.pmax # 128, maximum partition dimension size
    n_tiles_c_in = in_channels // c_in_pmax
    c_out_pmax = c_in_pmax
    n_tiles_c_out = out_channels // c_out_pmax

    # two output rows at one time ('chunk')
    chunk_size = 2
    input_rows = chunk_size + filter_height - 1
    num_chunks = math.ceil(input_height/chunk_size)
    # print('input height:', input_height)
    # print('output height:', out_height)
    # print('number of chunks:', num_chunks)
    # print('input rows:', input_rows)

    #reshaping & transposing
    W = W.reshape((n_tiles_c_out, c_out_pmax, n_tiles_c_in, c_in_pmax, filter_height, filter_width))

    w = nl.ndarray((filter_height, filter_width, n_tiles_c_out, n_tiles_c_in, nl.par_dim(c_in_pmax), c_out_pmax),
                   dtype=W.dtype, buffer=nl.sbuf)

    bias_loaded = nl.ndarray(shape=(c_out_pmax, n_tiles_c_out), dtype=bias.dtype, buffer=nl.sbuf)
    for c_out in nl.affine_range(n_tiles_c_out):
        # allocate space in sbuf
        weight_sbuf = nl.ndarray(
            (n_tiles_c_out, nl.par_dim(c_out_pmax), n_tiles_c_in, c_in_pmax, filter_height, filter_width),
            dtype=W.dtype,
            buffer=nl.sbuf
        )
        # final transposed weight tiles ready for matrix multiplication.



        weight_sbuf[c_out] = nl.load(W[c_out])
        bias_loaded[:, c_out] = nl.load(bias[c_out * c_out_pmax:(c_out + 1) * c_out_pmax])
        for c_in in nl.affine_range(n_tiles_c_in):
            for i in nl.affine_range(filter_height):
                for j in nl.affine_range(filter_width):
                    # print(i, j)
                    w[i, j, c_out, c_in] = nisa.nc_transpose(nl.copy(weight_sbuf[c_out, :, c_in, :, i, j], dtype=W.dtype))
            # for idx in nl.affine_range(filter_width * filter_height):
            #     i = nl.floor(nl.divide(idx, filter_width))
            #     j = nl.subtract(idx, nl.multiply(i, filter_width))
            #     print(i, j)
            #     weight_copy[i, j, c_out, c_in, :, :] = nl.copy(weight_sbuf[c_out, :, c_in, :, i, j], dtype=W.dtype)
            #     w[i, j, c_out, c_in] = nisa.nc_transpose(weight_copy[i, j, c_out, c_in])

    # Process the image in batches
    for b in nl.affine_range(batch_size):

        for n in nl.affine_range(num_chunks): # spatial chunk
            x = nl.ndarray(
                shape=(num_chunks, n_tiles_c_in, nl.par_dim(c_in_pmax), input_rows, input_width),
                dtype=X.dtype,
                buffer=nl.sbuf
            )
            # allocate space in sbuf to store the whole image
            # x = nl.ndarray(
            #     shape=(n_tiles_c_in, nl.par_dim(c_in_pmax), input_rows, input_width),
            #     dtype=X.dtype,
            #     buffer=nl.sbuf
            # )

            i_p, i_r, i_c = nl.mgrid[0: c_in_pmax, 0: input_rows, 0: input_width]
            global_row = n * chunk_size + i_r
            mask = global_row < input_height
            for c_in in nl.affine_range(n_tiles_c_in):  # loop over tiles to load X into x
                # start_row = n * chunk_size
                # end_row = min(start_row + chunk_size + filter_height - 1, input_height)
                # print(end_row)
                x[n, c_in, i_p, i_r, i_c] = nl.load(
                    X[b, c_in * c_in_pmax + i_p, global_row, i_c], mask=mask)

            for c_out in nl.affine_range(n_tiles_c_out):
                output_val = nl.ndarray(
                    shape=(nl.par_dim(c_out_pmax), chunk_size, out_width),
                    dtype=X.dtype,
                    buffer=nl.sbuf
                )
                result = nl.zeros((nl.par_dim(c_out_pmax), chunk_size, out_width), nl.float32, buffer=nl.psum)
                for i in nl.affine_range(filter_height):
                    for j in nl.affine_range(filter_width):
                        for c_in in nl.affine_range(n_tiles_c_in):
                            result += nisa.nc_matmul(
                                w[i, j, c_out, c_in, :, :],
                                x[n, c_in, :, i : chunk_size + i, j : j + out_width]
                            )


                # for row_count in nl.affine_range(chunk_size):

                    # print(output_val.shape)
                    # add bias
                output_val[:, :, :] = nl.add(result, bias_loaded[:, c_out])

                i_p, i_r, i_c = nl.mgrid[0:c_out_pmax, 0:chunk_size, 0:out_width]
                nl.store(X_out[b, c_out * c_out_pmax + i_p, n * chunk_size + i_r, i_c], output_val[i_p, i_r, i_c],mask = n * chunk_size + i_r < out_height)

    return X_out

