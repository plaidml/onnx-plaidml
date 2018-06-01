# Copyright Vertex.AI.
"""ONNX-PlaidML ONNX operator set definitions."""

from __future__ import print_function, division
from enum import Enum
import functools
import operator
import six
import sys

from onnx import onnx_pb
import onnx_plaidml.opset_util as opset_util
from onnx_plaidml.opset_util import opset, opset_op
import plaidml
import plaidml.op as op
import plaidml.tile as tile


class Constant(tile.Operation):
    """
    Defines a constant tensor.
    """

    def __init__(self, value):
        """Initialize the Constant tensor operation.
        
        Args:
            value (onnx_pb.TensorProto): The tensor to construct.
        """
        self.value = value
        try:
            outshape = tile.Shape(opset_util.ONNX_DTYPE_TO_PLAIDML[value.data_type], value.dims)
        except KeyError:
            six.raise_from(
                NotImplementedError(
                    'ONNX data type {} is not yet implemented by the PlaidML ONNX backend'.format(
                        onnx_pb.TensorProto.DataType.Name(value.data_type))), None)
        super(Constant, self).__init__(None, [], [('O', outshape)])

    def bind(self, bindings):
        return {
            'O': opset_util.onnx_tensor_to_plaidml_tensor(bindings.ctx, bindings.dev, self.value)
        }


def _pad_compute(sym, input_size, filter_size, stride, padding, pads):
    """Computes info for an axis of a padded filter.

    Args:
        sym (str): The symbol for the input axis.
        input_size (tile.Value or int): The size of the input axis (possibly symbolic).
        filter_size (int): The size of the filter along this axis.
        stride (int): The stride of the filter along this axis.
        padding (op.AutoPadding): The padding style to use.
        pads ((int, int) or None): Explicit pre- and post-padding for this axis.

    Returns:
        tuple(A string representing the output size as TILE code,
              The pre-padding to use when building input accessor expressions,
              A tile.Value representing the computed output size)
    """
    if pads:
        num_out_size = (input_size + pads[0] + pads[1] - filter_size + stride) // stride
        sym_output_size = '({sym} + {pre} + {post} - {fs} + {s}) / {s}'.format(
            sym=sym, pre=pads[0], post=pads[1], fs=filter_size, s=stride)
        sym_padding_before = pads[0]
    elif padding == op.AutoPadding.VALID:
        num_out_size = (input_size - filter_size + stride) // stride
        sym_output_size = '({sym} - {fs} + {s}) / {s}'.format(sym=sym, fs=filter_size, s=stride)
        sym_padding_before = 0
    elif padding == op.AutoPadding.SAME_UPPER or padding == op.AutoPadding.SAME_LOWER:
        num_out_size = (input_size + stride - 1) // stride
        sym_output_size = '({sym} + {s} - 1) / {s}'.format(sym=sym, s=stride)

        if padding == op.AutoPadding.SAME_UPPER:
            expr = '(max(0, ({symout} - 1) * {s} + {fs} - {syminp})) / 2'
        else:
            expr = '((max(0, ({symout} - 1) * {s} + {fs} - {syminp})) + 1) / 2'
        sym_padding_before = expr.format(
            symout=sym_output_size, s=stride, fs=filter_size, syminp=sym)
    else:
        raise Exception('Invalid padding: ' + str(padding))
    if not isinstance(num_out_size, tile.Value) and num_out_size < 0:
        raise Exception(
            'Invalid output size computed for convolution: num_out_size={}'.format(num_out_size))
    return (sym_output_size, sym_padding_before, num_out_size)


def _format_conv_strings(rank, in_shape, kernel_shape, strides, padding, pads, dilation_rate,
                         group):
    sym_out_shape = list()
    pad_amount = list()
    num_out_shape = list()
    for i in range(rank):
        sym_out, sym_pad, num_out = _pad_compute('L{}'.format(i), in_shape[i + 2],
                                                 dilation_rate[i] * (kernel_shape[i + 2] - 1) + 1,
                                                 strides[i], padding, (pads[i], pads[i + rank])
                                                 if pads else None)
        sym_out_shape.append(sym_out)
        pad_amount.append(sym_pad)
        num_out_shape.append(num_out)

    input_idx_list = [
        '{s}*x{idx} + {d}*k{idx} - {p}'.format(
            s=strides[i], idx=i, d=dilation_rate[i], p=pad_amount[i]) for i in range(rank)
    ]

    input_dims = ['N']
    if group == 1:
        input_dims.append('C')
    else:
        input_dims.append('GC')
    input_dims.extend(['L{}'.format(i) for i in range(rank)])
    input_dims_str = ', '.join(input_dims)

    out_dims = ['N']
    out_idx = ['n']
    if group == 1:
        out_dims.append('M')
        out_idx.append('m')
    else:
        out_dims.append(str(group))
        out_dims.append('M/{}'.format(group))
        out_idx.append('g')
        out_idx.append('m')
    out_dims.extend(sym_out_shape)
    out_idx.extend(['x{}'.format(i) for i in range(rank)])

    out_dims_str = ', '.join(out_dims)
    out_idx_str = ', '.join(out_idx)

    input_idx = ['n']
    if group == 1:
        input_idx.append('c')
    else:
        input_idx.append('(g * (GC/{})) + c'.format(group))
    input_idx.extend(input_idx_list)
    input_idx_str = ', '.join(input_idx)

    ker_dims_str = 'M, C, ' + ', '.join(['LK{}'.format(i) for i in range(rank)])

    if group == 1:
        ker_idx = ['m']
    else:
        ker_idx = ['(g*(M/{}))+m'.format(group)]
    ker_idx.append('c')
    ker_idx.extend(['k{}'.format(i) for i in range(rank)])
    ker_idx_str = ', '.join(ker_idx)

    outshape = [in_shape[0]] + [kernel_shape[0]] + num_out_shape

    if group == 1:
        group_reshape = 'GO'
    else:
        group_reshape = 'reshape(GO, {})'.format(', '.join(str(o) for o in outshape))

    ret = {
        'input_dims_str': input_dims_str,
        'ker_dims_str': ker_dims_str,
        'out_idx_str': out_idx_str,
        'out_dims_str': out_dims_str,
        'input_idx_str': input_idx_str,
        'ker_idx_str': ker_idx_str,
        'outshape_tuple': outshape,
        'group_reshape': group_reshape
    }
    return ret


_CONV_AUTO_PAD = {
    'VALID': op.AutoPadding.VALID,
    'SAME_UPPER': op.AutoPadding.SAME_UPPER,
    'SAME_LOWER': op.AutoPadding.SAME_LOWER
}


def _convert_auto_pad(auto_pad, pads):
    """Converts an ONNX auto-padding string to an op.AutoPadding.
    
    Args:
        auto_pad (str or None): The string description of the auto-padding.
        pads ([int] or None): The explicit paddings provided with the operation.
    
    Raises:
        ValueError: The auto_pad string is unrecognized or incompatible with the explicit paddings.
    
    Returns:
        op.AutoPadding: The enum value for the convolution padding.
    """
    if not auto_pad:
        if pads:
            return op.AutoPadding.EXPLICIT
        return op.AutoPadding.VALID
    if pads:
        raise ValueError('Can\'t specify pads and auto_pad in the same convolution;' +
                         'pads={}, auto_pad={}'.format(pads, auto_pad))
    try:
        return _CONV_AUTO_PAD[auto_pad]
    except KeyError:
        six.raise_from(NotImplementedError('Unsupported auto pad: {}'.format(auto_pad)), None)


def _extend_pads(pads, rank):
    """Extends a padding list to match the necessary rank.
    
    Args:
        pads ([int] or None): The explicitly-provided padding list.
        rank (int): The rank of the operation.

    Returns:
        None: If pads is None
        [int]: The extended padding list.
    """
    if pads is None:
        return pads
    pads = list(pads)
    if len(pads) < rank:
        pads.extend([0] * (rank - len(pads)))
    if len(pads) < (2 * rank):
        pads.extend(pads[len(pads) - rank:rank])
    return pads


class Convolution(tile.Operation):
    """
    A standard ML convolution operator.
    """

    def __init__(self,
                 data,
                 kernel,
                 auto_pad=None,
                 dilations=None,
                 group=1,
                 kernel_shape=None,
                 pads=None,
                 strides=None):
        rank = data.shape.ndims - 2
        padding = _convert_auto_pad(auto_pad, pads)
        pads = _extend_pads(pads, rank)
        if not strides:
            strides = tuple(1 for _ in range(rank))
        if not dilations:
            dilations = tuple(1 for _ in range(rank))
        if not kernel_shape:
            kernel_shape = kernel.shape.dims
        else:
            kernel_shape = tuple([kernel.shape.dims[0], kernel.shape.dims[1]] + list(kernel_shape))

        for entry in dilations:
            if not isinstance(entry, six.integer_types) or entry <= 0:
                raise ValueError('Invalid dilation_rate: {}'.format(dilations))
        if kernel.shape.ndims != rank + 2:
            raise ValueError('Convolution kernel shape inconsistent with input shape: ' +
                             '{} (rank {}) v {} (rank {})'.format(
                                 kernel.shape, kernel.shape.ndims - 2, data.shape,
                                 data.shape.ndims - 2))
        if len(strides) != rank:
            raise ValueError('Convolution strides length inconsistent with input shape: ' +
                             '{} (rank {}) v {} (rank {})'.format(
                                 strides, len(strides), data.shape, data.shape.ndims - 2))
        if len(dilations) != rank:
            raise ValueError('Convolution dilations length inconsistent with input shape: ' +
                             '{} (rank {}) v {} (rank {})'.format(
                                 dilations, len(dilations), data.shape, data.shape.ndims - 2))

        conv_strs = _format_conv_strings(rank, data.shape.dims, kernel_shape, strides, padding,
                                         pads, dilations, group)

        outshape = tile.Shape(data.shape.dtype, conv_strs['outshape_tuple'])

        code = """
        function (I[{input_dims_str}], K[{ker_dims_str}]) -> (O) {{
            GO[{out_idx_str} : {out_dims_str}] = +(I[{input_idx_str}]*K[{ker_idx_str}]);
            O = {group_reshape};
        }}""".format(**conv_strs)

        super(Convolution, self).__init__(code, [('I', data), ('K', kernel)], [('O', outshape)])


class Gemm(tile.Operation):
    """
    Implements a general matrix multiplication.
    """

    def __init__(self, a, b, c, alpha=None, beta=None, broadcast=True, transA=False, transB=False):
        if not broadcast and c.shape.ndims != 2:
            raise NotImplementedError(
                'Gemm without multiplier broadcast requires a two-dimensional scalar multiplier; multiplier rank={}'.
                format(c.shape.ndims))

        def gemm_reshape(value):
            if value.shape.ndims < 2:
                raise tile.LogicError(
                    'Invalid Gemm input; two-dimensions required, got: {}'.format(value.shape))
            if value.shape.ndims == 2:
                return value
            newdims = (value.shape.dims[0], functools.reduce(lambda x, y: x * y,
                                                             value.shape.dims[1:]))
            return op.reshape(value, newdims)

        a = gemm_reshape(a)
        b = gemm_reshape(b)

        code = """
        function (A[{a_dims}], B[{b_dims}], C) -> (O) {{
          OM[row, col : ROW, COL] = +(A[{a_idxs}] * B[{b_idxs}]);
          OA = {alpha_expr};
          CB = {beta_expr};
          O = OA + CB;
        }}""".format(
            a_dims='MID, ROW' if transA else 'ROW, MID',
            b_dims='COL, MID' if transB else 'MID, COL',
            a_idxs='mid, row' if transA else 'row, mid',
            b_idxs='col, mid' if transB else 'mid, col',
            alpha_expr='OM * {}'.format(alpha) if alpha else 'OM',
            beta_expr='C * {}'.format(beta) if beta else 'C',
        )

        outshape = tile.Shape(
            tile.common_dtype(a.shape.dtype, b.shape.dtype, c.shape.dtype),
            tile.broadcast_dims((
                a.shape.dims[1] if transA else a.shape.dims[0],
                b.shape.dims[0] if transB else b.shape.dims[1],
            ), c.shape.dims))

        super(Gemm, self).__init__(code, [('A', a), ('B', b), ('C', c)], [('O', outshape)])


class LocalChannelSum(tile.Operation):
    """
    Implements a localized sum over channels, used for local response normalization.
    """

    def __init__(self, data, size):

        rank = data.shape.ndims - 2
        distance = size // 2

        code = """
        function (I[{dims}]) -> (O) {{
          O[{out_idxs} : {dims}] = +(I[{in_idxs}]){constraints};
        }}""".format(
            dims=', '.join(['N', 'C'] + ['D{}'.format(i) for i in range(rank)]),
            out_idxs=', '.join(['n', 'c'] + ['d{}'.format(i) for i in range(rank)]),
            in_idxs=', '.join(['n', 'c-{}+z'.format(distance)] +
                              ['d{}'.format(i) for i in range(rank)]),
            constraints=', z < {}'.format(size))

        super(LocalChannelSum, self).__init__(code, [('I', data)], [('O', data.shape)])


class Mean(tile.Operation):
    """
    Implements an elementwise mean over a list of tensors.
    """

    def __init__(self, tensors):
        tnames = ['I{}'.format(n) for n in range(len(tensors))]
        code = """
        function ({inputs}) -> (O) {{
            O = (({input_sum}) / {input_count});
        }}
        """.format(
            inputs=','.join(tnames), input_sum='+'.join(tnames), input_count=len(tnames))

        outshape = tile.Shape(
            tile.common_dtype(*[t.shape.dtype for t in tensors]),
            tile.broadcast_dims(*[t.shape.dims for t in tensors]))

        super(Mean, self).__init__(code, list(zip(tnames, tensors)), [('O', outshape)])


class Not(tile.Operation):
    """
    Computes the elementwise logical not of a value.
    """

    def __init__(self, x):
        if x.shape.dtype != plaidml.DType.BOOLEAN:
            raise tile.LogicError('Logical Not requires a boolean tensor input')
        super(Not, self).__init__("""
            function (I) -> (O) {
                O = cmp_eq(I, 0);
            } """, [('I', x)], [('O', x.shape)])


class Flatten(tile.Operation):
    """
    Flattens an input tensor into a two-dimensional matrix.
    """

    def __init__(self, data, axis):
        in_dim_list = ['N{}'.format(i) for i in range(data.shape.ndims)]
        out_l_dim_list = ['*'.join(['1'] + ['N{}'.format(i) for i in range(axis)])]
        out_r_dim_list = [
            '*'.join(['1'] + ['N{}'.format(i) for i in range(axis, data.shape.ndims)])
        ]
        in_dims = list(data.shape.dims)
        l_size = functools.reduce(lambda x, y: x * y, [1] + in_dims[:axis])
        r_size = functools.reduce(lambda x, y: x * y, [1] + in_dims[axis:])

        code = 'function (I[{idims}]) -> (O) {{ O = reshape(I, {o_l_dims}, {o_r_dims}); }}'.format(
            idims=', '.join(in_dim_list),
            o_l_dims=', '.join(out_l_dim_list),
            o_r_dims=','.join(out_r_dim_list))
        super(Flatten, self).__init__(code, [('I', data)],
                                      [('O', tile.Shape(data.shape.dtype, (l_size, r_size)))])


class PadConstant(tile.Operation):
    """
    Implements constant tensor padding.
    """

    def __init__(self, data, mode=None, pads=None, value=None):
        if value is None:
            value = 0.
        rank = data.shape.ndims
        in_dims = ['D{}'.format(d) for d in range(data.shape.ndims)]
        out_dims = list(in_dims)
        in_idxs = ['d{}'.format(d) for d in range(data.shape.ndims)]
        out_idxs = list(in_idxs)
        shape_dims = list(data.shape.dims)

        for idx in range(rank):
            start = pads[idx]
            end = pads[idx + rank]
            if start + end:
                out_dims[idx] = 'D{}+{}'.format(idx, start + end)
                shape_dims[idx] += start + end
            if start:
                out_idxs[idx] = 'd{}+{}'.format(idx, start)

        if value:
            # TODO: This is a somewhat inefficient way to write a padding operation.
            code = """
            function (I[{in_dims}], One[], V[]) -> (O) {{
                Ones[{out_idxs} : {in_dims}] = =(One[]);
                InMask[{out_idxs} : {out_dims}] = =(Ones[{in_idxs}]);
                ValMask = 1 - InMask;
                Vals = ValMask * V;
                Ins[{out_idxs} : {out_dims}] = =(I[{in_idxs}]);
                O = Ins+Vals;
            }}""".format(
                in_dims=', '.join(in_dims),
                out_dims=', '.join(out_dims),
                in_idxs=', '.join(in_idxs),
                out_idxs=', '.join(out_idxs))
            value_input = [('One', tile.Value.from_var(1, tuple())), ('V', tile.Value.from_var(
                value, tuple()))]
        else:
            code = """
            function (I[{in_dims}]) -> (O) {{
                O[{out_idxs} : {out_dims}] = =(I[{in_idxs}]);
            }}""".format(
                in_dims=', '.join(in_dims),
                out_dims=', '.join(out_dims),
                in_idxs=', '.join(in_idxs),
                out_idxs=', '.join(out_idxs))
            value_input = []

        outshape = tile.Shape(data.shape.dtype, shape_dims)

        super(PadConstant, self).__init__(code, [('I', data)] + value_input, [('O', outshape)])


class Selu(tile.Operation):
    """
    Implements a selu.
    """

    def __init__(self, data, alpha, gamma):
        code = """
        function (X) -> (Y) {{
            Y = (X <= 0.0 ? ({gamma}*({alpha}*exp(X)-{alpha})) : ({gamma}*X));
        }}
        """.format(
            alpha=alpha, gamma=gamma)

        super(Selu, self).__init__(code, [('X', data)], [('Y', data.shape)])


class Sqrt(tile.Operation):
    """
    Computes the elementwise square root of a value.
    """

    def __init__(self, x):
        super(Sqrt, self).__init__("""
            function (I) -> (O) {
                O = sqrt(I);
            } """, [('I', x)], [('O', x.shape)])


class ThresholdedRelu(tile.Operation):
    """
    Implements a thresholded relu.

    This is essentially a simple relu where the inflection point is X==alpha, not 0.
    """

    def __init__(self, data, alpha):
        code = """
        function (X) -> (Y) {{ Y = (X <= {} ? 0.0 : X); }}
        """.format(alpha)

        super(ThresholdedRelu, self).__init__(code, [('X', data)], [('Y', data.shape)])


_CONV_PADDING_MODE = {
    # TODO: Implement edge and reflection padding.
    six.b('constant'): PadConstant,
    # six.b('edge'): PadEdge,
    # six.b('reflect'): PadReflect
}


class Transpose(tile.Operation):
    """
    Transposes a tensor.
    """

    def __init__(self, data, perm=None):
        if not perm:
            perm = range(data.shape.ndims - 1, -1, -1)

        ndims = data.shape.ndims

        code = """
        function (I[{in_dims}]) -> (O) {{
            O[{out_idxs} : {out_dims}] = =(I[{in_idxs}]);
        }}""".format(
            in_dims=', '.join(['D{}'.format(d) for d in range(ndims)]),
            out_dims=', '.join(['D{}'.format(perm[d]) for d in range(ndims)]),
            in_idxs=', '.join(['d{}'.format(d) for d in range(ndims)]),
            out_idxs=', '.join(['d{}'.format(perm[d]) for d in range(ndims)]))

        outshape = tile.Shape(data.shape.dtype, [data.shape.dims[perm[d]] for d in range(ndims)])

        super(Transpose, self).__init__(code, [('I', data)], [('O', outshape)])


@opset('', 1)
class _V1(object):

    @staticmethod
    @opset_op('Abs')
    def abs(unused_ctx, data):
        return (abs(data),)

    @staticmethod
    @opset_op('Add')
    def add(unused_ctx, a, b, axis=None, broadcast=None):
        if not broadcast and a.shape.dims != b.shape.dims:
            raise tile.LogicError('Incompatible shapes in addition')
        if broadcast and (axis is not None):
            b = op.reshape(b, list(b.shape.dims) + ([1] * (a.shape.ndims - b.shape.ndims - axis)))
        return (a + b,)

    @staticmethod
    @opset_op('And')
    def and_op(unused_ctx, a, b, axis=None, broadcast=None):
        if not broadcast and a.shape.dims != b.shape.dims:
            raise tile.LogicError('Incompatible shapes in logical and')
        if broadcast and (axis is not None):
            b = op.reshape(b, list(b.shape.dims) + ([1] * (a.shape.ndims - b.shape.ndims - axis)))
        return (tile.binary_op(
            a, b, 'cmp_eq(L ? R : 0, 1)', dtype=plaidml.DType.BOOLEAN, name='And'),)

    @staticmethod
    @opset_op('ArgMax')
    def argmax(unused_ctx, data, axis=-1, keepdims=1):
        if not keepdims:
            raise NotImplementedError(
                'ArgMax with keepdims=0 is not yet implemented by the PlaidML ONNX backend')
        return (op.argmax(data, axis=axis),)

    @staticmethod
    @opset_op('ArgMin')
    def argmin(unused_ctx, data, axis=-1, keepdims=1):
        if not keepdims:
            raise NotImplementedError(
                'ArgMin with keepdims=0 is not yet implemented by the PlaidML ONNX backend')
        return (op.argmax(-data, axis=axis),)

    @staticmethod
    @opset_op('AveragePool')
    def average_pool(unused_ctx, data, auto_pad=None, kernel_shape=None, pads=None, strides=None):
        padding = _convert_auto_pad(auto_pad, pads)
        return (op.average_pool(
            data, padding=padding, kernel_shape=kernel_shape, pads=pads, strides=strides),)

    @staticmethod
    @opset_op('BatchNormalization')
    def batch_normalization(unused_ctx,
                            value,
                            scale,
                            bias,
                            mean,
                            variance,
                            epsilon=1e-5,
                            is_test=0,
                            momentum=.9,
                            spatial=1,
                            consumed_inputs=None):
        if not is_test:
            raise NotImplementedError()

        shape = [value.shape.dims[1]] + ([1] * (value.shape.ndims - 2))
        scale = op.reshape(scale, shape)
        bias = op.reshape(bias, shape)
        mean = op.reshape(mean, shape)
        variance = op.reshape(variance, shape)

        denom = op.sqrt(variance + epsilon)
        return (((value - mean) * scale / denom) + bias,)

    @staticmethod
    @opset_op('Cast')
    def cast(unused_ctx, x, to):
        dtype = opset_util.ONNX_DTYPE_TO_PLAIDML[onnx_pb.TensorProto.DataType.Value(
            to.decode('utf-8'))]
        return (op.cast(x, dtype),)

    @staticmethod
    @opset_op('Ceil')
    def ceil(unused_ctx, x):
        return (op.ceiling(x),)

    @staticmethod
    @opset_op('Clip')
    def clip(unused_ctx, x, min=None, max=None):
        return (op.clip(x, min_val=min, max_val=max),)

    @staticmethod
    @opset_op('Concat')
    def concat(unused_ctx, *inputs, **kwargs):
        try:
            axis = kwargs['axis']
        except KeyError:
            axis = 1
        return (op.concatenate(inputs, axis),)

    @staticmethod
    @opset_op('Constant')
    def constant(unused_ctx, value=None):
        return (Constant.function(value=value),)

    @staticmethod
    @opset_op('Conv')
    def convolution(unused_ctx,
                    data,
                    kernel,
                    bias=None,
                    auto_pad=None,
                    dilations=None,
                    group=1,
                    kernel_shape=None,
                    pads=None,
                    strides=None):
        result = Convolution.function(
            data,
            kernel,
            auto_pad=auto_pad,
            dilations=dilations,
            group=group,
            kernel_shape=kernel_shape,
            pads=pads,
            strides=strides)
        if bias:
            bias = op.reshape(bias, [result.shape.dims[1]] + ([1] * (result.shape.ndims - 2)))
            result += bias
        return (result,)

    @staticmethod
    @opset_op('Div')
    def div(unused_ctx, a, b, axis=None, broadcast=None):
        if not broadcast and a.shape.dims != b.shape.dims:
            raise tile.LogicError('Incompatible shapes in division')
        if broadcast and (axis is not None):
            b = op.reshape(b, list(b.shape.dims) + ([1] * (a.shape.ndims - b.shape.ndims - axis)))
        return (a / b,)

    @staticmethod
    @opset_op('Dropout')
    def dropout(unused_ctx, data, is_test=0, ratio=0.5):
        if is_test:
            return (data,)
        raise NotImplementedError('Dropout in training mode is not currently implemented')

    @staticmethod
    @opset_op('Elu')
    def elu(unused_ctx, data, alpha=1.0):
        return (op.elu(data, alpha),)

    @staticmethod
    @opset_op('Equal')
    def equal(unused_ctx, a, b, axis=None, broadcast=None):
        if not broadcast and a.shape.dims != b.shape.dims:
            raise tile.LogicError('Incompatible shapes in equal')
        if broadcast and (axis is not None):
            b = op.reshape(b, list(b.shape.dims) + ([1] * (a.shape.ndims - b.shape.ndims - axis)))
        return (op.equal(a, b),)

    @staticmethod
    @opset_op('Exp')
    def exp(unused_ctx, data):
        return (op.exp(data),)

    @staticmethod
    @opset_op('Flatten')
    def flatten(unused_ctx, x, axis=1):
        return (Flatten.function(x, axis),)

    @staticmethod
    @opset_op('Floor')
    def floor(unused_ctx, x):
        return (op.floor(x),)

    @staticmethod
    @opset_op('Gather')
    def gather(unused_ctx, data, indicies, axis=0):
        if axis != 0:
            raise NotImplementedError(
                'Gather with a non-zero axis is not yet implemented by the PlaidML ONNX backend')
        if indicies.shape.dtype == plaidml.DType.INT64:
            # TODO: Long-term, it'd be a fine thing to have PlaidML accept
            # an int64 indicies input to gather().
            indicies = op.cast(indicies, plaidml.DType.INT32)
        return (op.gather(data, indicies),)

    @staticmethod
    @opset_op('Gemm')
    def gemm(unused_ctx,
             a,
             b,
             c,
             alpha=None,
             beta=None,
             broadcast=True,
             transA=False,
             transB=False):
        return (Gemm.function(
            a, b, c, alpha=alpha, beta=beta, broadcast=broadcast, transA=transA, transB=transB),)

    @staticmethod
    @opset_op('GlobalAveragePool')
    def global_average_pool(unused_ctx, x):
        return (op.average_pool(
            x,
            padding=op.AutoPadding.VALID,
            kernel_shape=x.shape.dims[2:],
            pads=None,
            strides=None),)

    @staticmethod
    @opset_op('GlobalMaxPool')
    def global_max_pool(unused_ctx, x):
        return (op.max_pool(
            x,
            padding=op.AutoPadding.VALID,
            kernel_shape=x.shape.dims[2:],
            pads=None,
            strides=None),)

    @staticmethod
    @opset_op('Greater')
    def greater(unused_ctx, a, b, axis=None, broadcast=None):
        if not broadcast and a.shape.dims != b.shape.dims:
            raise tile.LogicError('Incompatible shapes in logical > comparison')
        if broadcast and (axis is not None):
            b = op.reshape(b, list(b.shape.dims) + ([1] * (a.shape.ndims - b.shape.ndims - axis)))
        return (a > b,)

    @staticmethod
    @opset_op('Hardmax')
    def hardmax(unused_ctx, data, axis=None):
        return (op.hardmax(data, axis=axis),)

    @staticmethod
    @opset_op('HardSigmoid')
    def hardsigmoid(unused_ctx, x, alpha=.2, beta=.5):
        return (tile.maximum(0., tile.minimum(1., alpha * x + beta)),)

    @staticmethod
    @opset_op('Identity')
    def identity(unused_ctx, x):
        return (x,)

    @staticmethod
    @opset_op('InstanceNormalization')
    def instance_normalization(unused_ctx, value, scale, bias, epsilon=1e-5):
        shape = [value.shape.dims[1]] + ([1] * (value.shape.ndims - 2))
        scale = op.reshape(scale, shape)
        bias = op.reshape(bias, shape)
        mean = op.mean(value, axes=list(range(2, value.shape.ndims)), keepdims=True)
        variance = op.variance(value, axes=list(range(2, value.shape.ndims)), keepdims=True)

        denom = op.sqrt(variance + epsilon)
        return (((value - mean) * scale / denom) + bias,)

    @staticmethod
    @opset_op('LeakyRelu')
    def leaky_relu(unused_ctx, x, alpha=0.01):
        return (op.relu(x, alpha),)

    @staticmethod
    @opset_op('Less')
    def less(unused_ctx, a, b, axis=None, broadcast=None):
        if not broadcast and a.shape.dims != b.shape.dims:
            raise tile.LogicError('Incompatible shapes in logical < comparison')
        if broadcast and (axis is not None):
            b = op.reshape(b, list(b.shape.dims) + ([1] * (a.shape.ndims - b.shape.ndims - axis)))
        return (a < b,)

    @staticmethod
    @opset_op('Log')
    def log(unused_ctx, x):
        return (op.log(x),)

    @staticmethod
    @opset_op('LogSoftmax')
    def log_softmax(unused_ctx, x, axis=None):
        return (op.log_softmax(x, axis),)

    @staticmethod
    @opset_op('LRN')
    def lrn(unused_ctx, data, alpha, beta, size, bias=1.):
        local_sums = LocalChannelSum.function(data * data, size)
        return (data / op.pow(bias + ((alpha / size) * local_sums), beta),)

    @staticmethod
    @opset_op('MatMul')
    def matmul(unused_ctx, lhs, rhs):
        return (op.matmul(lhs, rhs),)

    @staticmethod
    @opset_op('Max')
    def max(unused_ctx, *tensors):
        return (functools.reduce(lambda x, y: op.maximum(x, y), tensors),)

    @staticmethod
    @opset_op('MaxPool')
    def max_pool(unused_ctx, data, auto_pad=None, kernel_shape=None, pads=None, strides=None):
        padding = _convert_auto_pad(auto_pad, pads)
        return (op.max_pool(
            data, padding=padding, kernel_shape=kernel_shape, pads=pads, strides=strides),)

    @staticmethod
    @opset_op('Mean')
    def mean(unused_ctx, *tensors):
        if len(tensors) == 0:
            raise tile.LogicError('Must supply at least one tensor in a Mean operation')
        if len(tensors) == 1:
            return (tensors[0],)
        return (Mean.function(tensors),)

    @staticmethod
    @opset_op('Min')
    def min(unused_ctx, *tensors):
        return (functools.reduce(lambda x, y: op.minimum(x, y), tensors),)

    @staticmethod
    @opset_op('Mul')
    def mul(unused_ctx, a, b, axis=None, broadcast=None):
        if not broadcast and a.shape.dims != b.shape.dims:
            raise tile.LogicError('Incompatible shapes in multiplication')
        if broadcast and (axis is not None):
            b = op.reshape(b, list(b.shape.dims) + ([1] * (a.shape.ndims - b.shape.ndims - axis)))
        return (a * b,)

    @staticmethod
    @opset_op('Neg')
    def neg(unused_ctx, value):
        return (-value,)

    @staticmethod
    @opset_op('Not')
    def not_op(unused_ctx, value):
        return (Not.function(value),)

    @staticmethod
    @opset_op('Or')
    def or_op(unused_ctx, a, b, axis=None, broadcast=None):
        if not broadcast and a.shape.dims != b.shape.dims:
            raise tile.LogicError('Incompatible shapes in logical or')
        if broadcast and (axis is not None):
            b = op.reshape(b, list(b.shape.dims) + ([1] * (a.shape.ndims - b.shape.ndims - axis)))
        return (tile.binary_op(
            a, b, 'cmp_eq(L ? 1 : R, 1)', dtype=plaidml.DType.BOOLEAN, name='Or'),)

    @staticmethod
    @opset_op('PRelu')
    def prelu(unused_ctx, x, slope):
        if slope.shape.ndims == 1 and x.shape.ndims > 2:
            slope = op.reshape(slope, [slope.shape.dims[0]] + [1] * (x.shape.ndims - 2))
        return (op.relu(x, alpha=slope),)

    @staticmethod
    @opset_op('Pad')
    def pad(unused_ctx, data, mode=None, paddings=None, value=None):
        if not mode:
            mode = 'constant'
            padding_mode = PadConstant
        else:
            try:
                padding_mode = _CONV_PADDING_MODE[mode]
            except KeyError:
                six.raise_from(ValueError('Unsupported padding mode: {}'.format(mode)), None)
        if not paddings or len(paddings) != 2 * data.shape.ndims:
            raise tile.LogicError('Inconsistant padding request; rank={}, #paddings={}'.format(
                data.shape.ndims, len(paddings) if paddings else 0))

        return (padding_mode.function(data, pads=paddings, mode=mode, value=value),)

    @staticmethod
    @opset_op('Pow')
    def pow(unused_ctx, data, exponent, axis=None, broadcast=None):
        if not broadcast and data.shape.dims != exponent.shape.dims:
            raise tile.LogicError('Incompatible shapes in power')
        if broadcast and (axis is not None):
            exponent = op.reshape(exponent,
                                  list(exponent.shape.dims) +
                                  ([1] * (data.shape.ndims - exponent.shape.ndims - axis)))
        return (op.pow(data, exponent),)

    @staticmethod
    @opset_op('Reciprocal')
    def reciprocal(unused_ctx, data):
        return (1. / data,)

    @staticmethod
    @opset_op('Relu')
    def relu(unused_ctx, data):
        return (op.relu(data),)

    @staticmethod
    @opset_op('Reshape')
    def reshape(unused_ctx, data, shape=None):
        if not shape:
            raise tile.LogicError('Reshape requires a target shape')
        return (op.reshape(data, shape),)

    @staticmethod
    @opset_op('ReduceL1')
    def reduce_l1(unused_ctx, data, axes=None, keepdims=1):
        if axes is None:
            axes = range(data.shape.ndims)
        if not isinstance(axes, (list, tuple)):
            axes = tuple(axes)
        return (op.summation(abs(data), axes=axes, keepdims=keepdims),)

    @staticmethod
    @opset_op('ReduceL2')
    def reduce_l2(unused_ctx, data, axes=None, keepdims=1):
        if axes is None:
            axes = range(data.shape.ndims)
        if not isinstance(axes, (list, tuple)):
            axes = tuple(axes)
        return (op.sqrt(op.summation(data * data, axes=axes, keepdims=keepdims)),)

    @staticmethod
    @opset_op('ReduceLogSum')
    def reduce_log_sum(unused_ctx, data, axes=None, keepdims=1):
        if axes is None:
            axes = range(data.shape.ndims)
        if not isinstance(axes, (list, tuple)):
            axes = tuple(axes)
        return (op.log(op.summation(data, axes=axes, keepdims=keepdims)),)

    @staticmethod
    @opset_op('ReduceLogSumExp')
    def reduce_log_sum_exp(unused_ctx, data, axes=None, keepdims=1):
        if axes is None:
            axes = range(data.shape.ndims)
        if not isinstance(axes, (list, tuple)):
            axes = tuple(axes)
        return (op.log(op.summation(op.exp(data), axes=axes, keepdims=keepdims)),)

    @staticmethod
    @opset_op('ReduceMax')
    def reduce_max(unused_ctx, data, axes=None, keepdims=1):
        if axes is None:
            axes = range(data.shape.ndims)
        if not isinstance(axes, (list, tuple)):
            axes = tuple(axes)
        return (op.max_reduce(data, axes=axes, keepdims=keepdims),)

    @staticmethod
    @opset_op('ReduceMean')
    def reduce_mean(unused_ctx, data, axes=None, keepdims=1):
        if axes is None:
            axes = range(data.shape.ndims)
        if not isinstance(axes, (list, tuple)):
            axes = tuple(axes)
        return (op.mean(data, axes=axes, keepdims=keepdims),)

    @staticmethod
    @opset_op('ReduceMin')
    def reduce_min(unused_ctx, data, axes=None, keepdims=1):
        if axes is None:
            axes = range(data.shape.ndims)
        if not isinstance(axes, (list, tuple)):
            axes = tuple(axes)
        return (op.min_reduce(data, axes=axes, keepdims=keepdims),)

    @staticmethod
    @opset_op('ReduceProd')
    def reduce_prod(unused_ctx, data, axes=None, keepdims=1):
        if axes is None:
            axes = range(data.shape.ndims)
        if not isinstance(axes, (list, tuple)):
            axes = tuple(axes)
        return (op.prod(data, axes=axes, keepdims=keepdims),)

    @staticmethod
    @opset_op('ReduceSum')
    def reduce_sum(unused_ctx, data, axes=None, keepdims=1):
        if axes is None:
            axes = range(data.shape.ndims)
        if not isinstance(axes, (list, tuple)):
            axes = tuple(axes)
        return (op.summation(data, axes=axes, keepdims=keepdims),)

    @staticmethod
    @opset_op('ReduceSumSquare')
    def reduce_sum_square(unused_ctx, data, axes=None, keepdims=1):
        if axes is None:
            axes = range(data.shape.ndims)
        if not isinstance(axes, (list, tuple)):
            axes = tuple(axes)
        return (op.summation(data * data, axes=axes, keepdims=keepdims),)

    @staticmethod
    @opset_op('Selu')
    def selu(unused_ctx, data, alpha=1.6732, gamma=1.0507):
        return (Selu.function(data, alpha, gamma),)

    @staticmethod
    @opset_op('Shape')
    def shape(unused_ctx, data):
        return (op.cast(op.shape_of(data), plaidml.DType.INT64),)

    @staticmethod
    @opset_op('Sigmoid')
    def sigmoid(unused_ctx, value):
        return (op.sigmoid(value),)

    @staticmethod
    @opset_op('Size')
    def size(unused_ctx, value):
        return (op.cast(op.prod(op.shape_of(value)), plaidml.DType.INT64),)

    @staticmethod
    @opset_op('Slice')
    def slice(unused_ctx, data, axes=None, ends=None, starts=None):
        return (op.slice_tensor(data, axes=axes, ends=ends, starts=starts),)

    @staticmethod
    @opset_op('Softmax')
    def softmax(unused_ctx, data, axis=None):
        return (op.softmax(data, axis=axis),)

    @staticmethod
    @opset_op('Softplus')
    def softplus(unused_ctx, data):
        return (op.log(op.exp(data) + 1.),)

    @staticmethod
    @opset_op('Softsign')
    def softsign(unused_ctx, data):
        return (data / (1 + abs(data)),)

    @staticmethod
    @opset_op('Split')
    def split(ctx, value, axis=None, split=None):
        if axis is None:
            axis = 0
        if split is None:
            axis_len = value.shape.dims[axis]
            group_size = axis_len // len(ctx.node.output)
            split = [group_size] * len(ctx.node.output)
        axes = [slice(None, None) for n in range(value.shape.ndims)]
        prev = 0
        results = []
        for spl in split:
            axes[axis] = slice(prev, prev + spl)
            results.append(operator.getitem(value, tuple(axes)))
            prev += spl
        return tuple(results)

    @staticmethod
    @opset_op('Sqrt')
    def sqrt(unused_ctx, value):
        return (Sqrt.function(value),)

    @staticmethod
    @opset_op('Squeeze')
    def squeeze(unused_ctx, data, axes):
        return (op.squeeze(data, axes),)

    @staticmethod
    @opset_op('Sub')
    def sub(unused_ctx, a, b, axis=None, broadcast=None):
        if not broadcast and a.shape.dims != b.shape.dims:
            raise tile.LogicError('Incompatible shapes in subtraction')
        if broadcast and (axis is not None):
            b = op.reshape(b, list(b.shape.dims) + ([1] * (a.shape.ndims - b.shape.ndims - axis)))
        return (a - b,)

    @staticmethod
    @opset_op('Sum')
    def sum(unused_ctx, *args):
        return (functools.reduce(lambda x, y: x + y, args),)

    @staticmethod
    @opset_op('Tanh')
    def tanh(unused_ctx, value):
        return (op.tanh(value),)

    @staticmethod
    @opset_op('ThresholdedRelu')
    def thresholded_relu(unused_ctx, data, alpha=1.):
        return (ThresholdedRelu.function(data, alpha),)

    @staticmethod
    @opset_op('Transpose')
    def transpose(unused_ctx, data, perm=None):
        return (Transpose.function(data, perm=perm),)

    @staticmethod
    @opset_op('Unsqueeze')
    def unsqueeze(unused_ctx, data, axes):
        return (op.unsqueeze(data, axes),)

    @staticmethod
    @opset_op('Xor')
    def xor(unused_ctx, a, b, axis=None, broadcast=None):
        if not broadcast and a.shape.dims != b.shape.dims:
            raise tile.LogicError('Incompatible shapes in exclusive-or')
        if broadcast and (axis is not None):
            b = op.reshape(b, list(b.shape.dims) + ([1] * (a.shape.ndims - b.shape.ndims - axis)))
        return (a ^ b,)


@opset('', 2)
class _V2(_V1):

    @staticmethod
    @opset_op('Pad')
    def pad(unused_ctx, data, mode=None, pads=None, value=None):
        if not mode:
            mode = 'constant'
            padding_mode = PadConstant
        else:
            try:
                padding_mode = _CONV_PADDING_MODE[mode]
            except KeyError:
                six.raise_from(ValueError('Unsupported padding mode: {}'.format(mode)), None)
        if not pads or len(pads) != 2 * data.shape.ndims:
            raise tile.LogicError('Inconsistant padding request; rank={}, #pads={}'.format(
                data.shape.ndims, len(pads) if pads else 0))

        return (padding_mode.function(data, pads=pads, mode=mode, value=value),)


@opset('', 3)
class _V3(_V2):
    pass


@opset('', 4)
class _V4(_V3):

    @staticmethod
    @opset_op('Concat')
    def concat(unused_ctx, *inputs, **kwargs):
        try:
            axis = kwargs['axis']
        except KeyError:
            six.raise_from(ValueError('Concatenation axis must be explicitly specified'), None)
        return (op.concatenate(inputs, axis),)


@opset('', 5)
class _V5(_V4):

    @staticmethod
    @opset_op('Reshape')
    def reshape(unused_ctx, data, shape):
        # Reshape V5 takes its shape as a tensor.  This is tricky to implement -- there's no good
        # way to provide constant values to the reshape() operation until all inputs are actually
        # bound.  Once inputs have been bound, we could construct a program whose output is the
        # one-dimensional shape tensor, run it, read the result, and use that to build the
        # correct reshape() operation for the actual program we want to run.  But note that
        # changing the input tensor may require recompiling the program, which is somewhat against
        # PlaidML's model.  So this needs some further thought.
        raise NotImplementedError(
            'Version-5 reshape() is not yet implemented by the PlaidML ONNX backend')


@opset('', 6)
class _V6(_V5):

    @staticmethod
    @opset_op('Cast')
    def cast(unused_ctx, x, to):
        dtype = opset_util.ONNX_DTYPE_TO_PLAIDML[to]
        return (op.cast(x, dtype),)


@opset('', 7)
class _V7(_V6):

    @classmethod
    @opset_op('Add')
    def add(cls, ctx, a, b):
        return super(_V7, cls).add(ctx, a, b, broadcast=True)

    @classmethod
    @opset_op('And')
    def and_op(cls, ctx, a, b):
        return super(_V7, cls).and_op(ctx, a, b, broadcast=True)

    @staticmethod
    @opset_op('Cos')
    def cos(unused_ctx, x):
        return (op.cos(x),)

    @classmethod
    @opset_op('Div')
    def div(cls, ctx, a, b):
        return super(_V7, cls).div(ctx, a, b, broadcast=True)

    @classmethod
    @opset_op('Equal')
    def equal(cls, ctx, a, b):
        return super(_V7, cls).equal(ctx, a, b, broadcast=True)

    @classmethod
    @opset_op('Greater')
    def greater(cls, ctx, a, b):
        return super(_V7, cls).greater(ctx, a, b, broadcast=True)

    @classmethod
    @opset_op('Less')
    def less(cls, ctx, a, b):
        return super(_V7, cls).less(ctx, a, b, broadcast=True)

    @classmethod
    @opset_op('Mul')
    def mul(cls, ctx, a, b):
        return super(_V7, cls).mul(ctx, a, b, broadcast=True)

    @classmethod
    @opset_op('Or')
    def or_op(cls, ctx, a, b):
        return super(_V7, cls).or_op(ctx, a, b, broadcast=True)

    @classmethod
    @opset_op('Pow')
    def pow(cls, ctx, a, b):
        return super(_V7, cls).pow(ctx, a, b, broadcast=True)

    @staticmethod
    @opset_op('PRelu')
    def prelu(unused_ctx, x, slope):
        # N.B. According to the operator specification, this is the V6 behavior of PRelu.
        # The ONNX backend tests, however, expect that at V6, the V1 operator behavior will be used
        # (broadcasting a one-dimensional slope to the channels dimension of the input tensor);
        # at V7, they expect the unidirectional broadcast behavior (implemented by this method).
        return (op.relu(x, alpha=slope),)

    @classmethod
    @opset_op('Sub')
    def sub(cls, ctx, a, b):
        return super(_V7, cls).sub(ctx, a, b, broadcast=True)

    @classmethod
    @opset_op('Xor')
    def xor(cls, ctx, a, b):
        return super(_V7, cls).xor(ctx, a, b, broadcast=True)


OPSETS = [_V1, _V2, _V3, _V4, _V5, _V6, _V7]

# As a special case for the ONNX operator set, we define a default version:
# this is the version of the ONNX operator set that will be loaded when there
# is no operator set defined by the model.  It should typically be the highest
# version of the operator set defined by this module.
DEFAULT_VERSION = 7
