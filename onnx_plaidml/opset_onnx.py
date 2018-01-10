# Copyright Vertex.AI.
"""ONNX-PlaidML ONNX operator set definitions."""

from __future__ import print_function, division
from enum import Enum
import functools
import six

from onnx import onnx_pb2
import onnx_plaidml.opset_util as opset_util
from onnx_plaidml.opset_util import opset, operator
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
            value (onnx_pb2.TensorProto): The tensor to construct.
        """
        self.value = value
        try:
            outshape = tile.Shape(opset_util.ONNX_DTYPE_TO_PLAIDML[value.data_type], value.dims)
        except KeyError:
            six.raise_from(
                NotImplementedError(
                    'ONNX data type {} is not yet implemented by the PlaidML ONNX backend'.format(
                        onnx_pb2.TensorProto.DataType.Name(value.data_type))), None)
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
            raise ValueError(
                'Convolution kernel shape inconsistent with input shape: ' +
                '{} (rank {}) v {} (rank {})'.format(kernel.shape, kernel.shape.ndims - 2,
                                                     data.shape, data.shape.ndims - 2))
        if len(strides) != rank:
            raise ValueError('Convolution strides length inconsistent with input shape: ' +
                             '{} (rank {}) v {} (rank {})'.format(strides, len(
                                 strides), data.shape, data.shape.ndims - 2))
        if len(dilations) != rank:
            raise ValueError('Convolution dilations length inconsistent with input shape: ' +
                             '{} (rank {}) v {} (rank {})'.format(dilations, len(
                                 dilations), data.shape, data.shape.ndims - 2))

        conv_strs = _format_conv_strings(rank, data.shape.dims, kernel_shape, strides, padding,
                                         pads, dilations, group)

        outshape = tile.Shape(data.shape.dtype, conv_strs['outshape_tuple'])

        code = """
        function (I[{input_dims_str}], K[{ker_dims_str}]) -> (O) {{
            GO[{out_idx_str} : {out_dims_str}] = +(I[{input_idx_str}]*K[{ker_idx_str}]);
            O = {group_reshape};
        }}""".format(**conv_strs)

        super(Convolution, self).__init__(code, [('I', data), ('K', kernel)], [('O', outshape)])


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
            in_idxs=', '.join(
                ['n', 'c-{}+z'.format(distance)] + ['d{}'.format(i) for i in range(rank)]),
            constraints=', z < {}'.format(size))

        super(LocalChannelSum, self).__init__(code, [('I', data)], [('O', data.shape)])


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
            value_input = [('One', tile.Value.from_var(1, tuple())), ('V',
                                                                      tile.Value.from_var(
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
    @operator('Abs')
    def abs(data):
        return (abs(data),)

    @staticmethod
    @operator('Add')
    def add(a, b, axis=None, broadcast=None):
        if not broadcast and a.shape.dims != b.shape.dims:
            raise tile.LogicError('Incompatible shapes in addition')
        if broadcast and axis:
            b = op.reshape(b, list(b.shape.dims) + ([1] * (a.shape.ndims - b.shape.ndims - axis)))
        return (a + b,)

    @staticmethod
    @operator('And')
    def and_op(a, b, axis=None, broadcast=None):
        if not broadcast and a.shape.dims != b.shape.dims:
            raise tile.LogicError('Incompatible shapes in logical and')
        if broadcast and axis:
            b = op.reshape(b, list(b.shape.dims) + ([1] * (a.shape.ndims - b.shape.ndims - axis)))
        return (tile.binary_op(a, b, 'L && R', dtype=plaidml.DType.BOOLEAN, name='And'),)

    @staticmethod
    @operator('ArgMax')
    def argmax(data, axis=-1, keepdims=1):
        if not keepdims:
            raise NotImplementedError(
                'ArgMax with keepdims=0 is not yet implemented by the PlaidML ONNX backend')
        return (op.argmax(data, axis=axis),)

    @staticmethod
    @operator('ArgMin')
    def argmin(data, axis=-1, keepdims=1):
        if not keepdims:
            raise NotImplementedError(
                'ArgMin with keepdims=0 is not yet implemented by the PlaidML ONNX backend')
        return (op.argmax(-data, axis=axis),)

    @staticmethod
    @operator('AveragePool')
    def average_pool(data, auto_pad=None, kernel_shape=None, pads=None, strides=None):
        padding = _convert_auto_pad(auto_pad, pads)
        return (op.average_pool(
            data, padding=padding, kernel_shape=kernel_shape, pads=pads, strides=strides),)

    @staticmethod
    @operator('BatchNormalization')
    def batch_normalization(value,
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
    @operator('Cast')
    def cast(x, to):
        dtype = opset_util.ONNX_DTYPE_TO_PLAIDML[onnx_pb2.TensorProto.DataType.Value(to)]
        return (op.cast(x, dtype),)

    @staticmethod
    @operator('Ceil')
    def ceil(x):
        return (op.ceiling(x),)

    @staticmethod
    @operator('Clip')
    def clip(x, min, max):
        return (op.clip(x, min_val=min, max_val=max),)

    @staticmethod
    @operator('Concat')
    def concat(*inputs, **kwargs):
        try:
            axis = kwargs['axis']
        except KeyError:
            axis = 1
        return (op.concatenate(inputs, axis),)

    @staticmethod
    @operator('Constant')
    def constant(value=None):
        return (Constant.function(value=value),)

    @staticmethod
    @operator('Conv')
    def convolution(data,
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
    @operator('Div')
    def div(a, b, axis=None, broadcast=None):
        if not broadcast and a.shape.dims != b.shape.dims:
            raise tile.LogicError('Incompatible shapes in division')
        if broadcast and axis:
            b = op.reshape(b, list(b.shape.dims) + ([1] * (a.shape.ndims - b.shape.ndims - axis)))
        return (a / b,)

    @staticmethod
    @operator('Dropout')
    def dropout(data, is_test=0, ratio=0.5):
        if is_test:
            return (data,)
        raise NotImplementedError('Dropout in training mode is not currently implemented')

    @staticmethod
    @operator('Elu')
    def elu(data, alpha=1.0):
        return (op.elu(data, alpha),)

    @staticmethod
    @operator('Equal')
    def equal(a, b, axis=None, broadcast=None):
        if not broadcast and a.shape.dims != b.shape.dims:
            raise tile.LogicError('Incompatible shapes in equal')
        if broadcast and axis:
            b = op.reshape(b, list(b.shape.dims) + ([1] * (a.shape.ndims - b.shape.ndims - axis)))
        return (op.equal(a, b),)

    @staticmethod
    @operator('Exp')
    def exp(data):
        return (op.exp(data),)

    @staticmethod
    @operator('Floor')
    def floor(x):
        return (op.floor(x),)

    @staticmethod
    @operator('Gather')
    def gather(data, indicies):
        if indicies.shape.dtype == plaidml.DType.INT64:
            # TODO: Long-term, it'd be a fine thing to have PlaidML accept
            # an int64 indicies input to gather().
            indicies = op.cast(indicies, plaidml.DType.INT32)
        return (op.gather(data, indicies),)

    @staticmethod
    @operator('Gemm')
    def gemm(a, b, c, alpha=None, beta=None, broadcast=True, transA=False, transB=False):
        return (op.gemm(
            a, b, c, alpha=alpha, beta=beta, broadcast=broadcast, transA=transA, transB=transB),)

    @staticmethod
    @operator('GlobalAveragePool')
    def global_average_pool(x):
        return (op.average_pool(
            x,
            padding=op.AutoPadding.VALID,
            kernel_shape=x.shape.dims[2:],
            pads=None,
            strides=None),)

    @staticmethod
    @operator('GlobalMaxPool')
    def global_max_pool(x):
        return (op.max_pool(
            x,
            padding=op.AutoPadding.VALID,
            kernel_shape=x.shape.dims[2:],
            pads=None,
            strides=None),)

    @staticmethod
    @operator('Greater')
    def greater(a, b, axis=None, broadcast=None):
        if not broadcast and a.shape.dims != b.shape.dims:
            raise tile.LogicError('Incompatible shapes in logical > comparison')
        if broadcast and axis:
            b = op.reshape(b, list(b.shape.dims) + ([1] * (a.shape.ndims - b.shape.ndims - axis)))
        return (a > b,)

    @staticmethod
    @operator('LeakyRelu')
    def leaky_relu(x, alpha=0.01):
        return (op.relu(x, alpha),)

    @staticmethod
    @operator('Less')
    def less(a, b, axis=None, broadcast=None):
        if not broadcast and a.shape.dims != b.shape.dims:
            raise tile.LogicError('Incompatible shapes in logical < comparison')
        if broadcast and axis:
            b = op.reshape(b, list(b.shape.dims) + ([1] * (a.shape.ndims - b.shape.ndims - axis)))
        return (a < b,)

    @staticmethod
    @operator('Log')
    def log(x):
        return (op.log(x),)

    @staticmethod
    @operator('LRN')
    def lrn(data, alpha, beta, size, bias=1.):
        local_sums = LocalChannelSum.function(data * data, size)
        return (data / op.pow(bias + ((alpha / size) * local_sums), beta),)

    @staticmethod
    @operator('MatMul')
    def matmul(lhs, rhs):
        return (op.matmul(lhs, rhs),)

    @staticmethod
    @operator('Max')
    def max(*tensors):
        return (functools.reduce(lambda x, y: op.maximum(x, y), tensors),)

    @staticmethod
    @operator('MaxPool')
    def max_pool(data, auto_pad=None, kernel_shape=None, pads=None, strides=None):
        padding = _convert_auto_pad(auto_pad, pads)
        return (op.max_pool(
            data, padding=padding, kernel_shape=kernel_shape, pads=pads, strides=strides),)

    @staticmethod
    @operator('Min')
    def min(*tensors):
        return (functools.reduce(lambda x, y: op.minimum(x, y), tensors),)

    @staticmethod
    @operator('Mul')
    def mul(a, b, axis=None, broadcast=None):
        if not broadcast and a.shape.dims != b.shape.dims:
            raise tile.LogicError('Incompatible shapes in multiplication')
        if broadcast and axis:
            b = op.reshape(b, list(b.shape.dims) + ([1] * (a.shape.ndims - b.shape.ndims - axis)))
        return (a * b,)

    @staticmethod
    @operator('Neg')
    def neg(value):
        return (-value,)

    @staticmethod
    @operator('Not')
    def not_op(value):
        return (~value,)

    @staticmethod
    @operator('Or')
    def or_op(a, b, axis=None, broadcast=None):
        if not broadcast and a.shape.dims != b.shape.dims:
            raise tile.LogicError('Incompatible shapes in logical or')
        if broadcast and axis:
            b = op.reshape(b, list(b.shape.dims) + ([1] * (a.shape.ndims - b.shape.ndims - axis)))
        return (tile.binary_op(a, b, 'L || R', dtype=plaidml.DType.BOOLEAN, name='Or'),)

    @staticmethod
    @operator('PRelu')
    def prelu(x, slope):
        if slope.shape.ndims == 1 and x.shape.ndims > 2:
            slope = op.reshape(slope, [slope.shape.dims[0]] + [1] * (x.shape.ndims - 2))
        return (op.relu(x, alpha=slope),)

    @staticmethod
    @operator('Pad')
    def pad(data, mode=None, paddings=None, value=None):
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
                data.shape.ndims,
                len(paddings) if paddings else 0))

        return (padding_mode.function(data, pads=paddings, mode=mode, value=value),)

    @staticmethod
    @operator('Pow')
    def pow(data, exponent, axis=None, broadcast=None):
        if not broadcast and data.shape.dims != exponent.shape.dims:
            raise tile.LogicError('Incompatible shapes in power')
        if broadcast and axis:
            exponent = op.reshape(exponent,
                                  list(exponent.shape.dims) +
                                  ([1] * (data.shape.ndims - exponent.shape.ndims - axis)))
        return (op.pow(data, exponent),)

    @staticmethod
    @operator('Reciprocal')
    def reciprocal(data):
        return (1. / data,)

    @staticmethod
    @operator('Relu')
    def relu(data):
        return (op.relu(data),)

    @staticmethod
    @operator('Reshape')
    def reshape(data, shape=None):
        if not shape:
            raise tile.LogicError('Reshape requires a target shape')
        return (op.reshape(data, shape),)

    @staticmethod
    @operator('Selu')
    def selu(data, alpha=1.6732, gamma=1.0507):
        return (gamma * (alpha * op.exp(data) - alpha),)

    @staticmethod
    @operator('Shape')
    def shape(data):
        return (op.shape_of(data),)

    @staticmethod
    @operator('Sigmoid')
    def sigmoid(value):
        return (op.sigmoid(value),)

    @staticmethod
    @operator('Size')
    def size(value):
        return (functools.reduce(lambda x, y: x * y, value.shape.dims),)

    @staticmethod
    @operator('Slice')
    def slice(data, axes=None, ends=None, starts=None):
        return (op.slice_tensor(data, axes=axes, ends=ends, starts=starts),)

    @staticmethod
    @operator('Softmax')
    def softmax(data, axis=None):
        return (op.softmax(data, axis=axis),)

    @staticmethod
    @operator('Softplus')
    def softplus(data):
        return (op.log(op.exp(data) + 1.),)

    @staticmethod
    @operator('Softsign')
    def softsign(data):
        return (data / (1 + abs(data)),)

    @staticmethod
    @operator('Sqrt')
    def sqrt(value):
        return (op.sqrt(value),)

    @staticmethod
    @operator('Sub')
    def sub(a, b, axis=None, broadcast=None):
        if not broadcast and a.shape.dims != b.shape.dims:
            raise tile.LogicError('Incompatible shapes in subtraction')
        if broadcast and axis:
            b = op.reshape(b, list(b.shape.dims) + ([1] * (a.shape.ndims - b.shape.ndims - axis)))
        return (a - b,)

    @staticmethod
    @operator('Sum')
    def sum(*args):
        return (functools.reduce(lambda x, y: x + y, args),)

    @staticmethod
    @operator('Tanh')
    def tanh(value):
        return (op.tanh(value),)

    @staticmethod
    @operator('Transpose')
    def transpose(data, perm=None):
        return (Transpose.function(data, perm=perm),)

    @staticmethod
    @operator('Xor')
    def xor(a, b, axis=None, broadcast=None):
        if not broadcast and a.shape.dims != b.shape.dims:
            raise tile.LogicError('Incompatible shapes in subtraction')
        if broadcast and axis:
            b = op.reshape(b, list(b.shape.dims) + ([1] * (a.shape.ndims - b.shape.ndims - axis)))
        return (a ^ b,)


@opset('', 2)
class _V2(_V1):

    @staticmethod
    @operator('Pad')
    def pad(data, mode=None, pads=None, value=None):
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
                data.shape.ndims,
                len(pads) if pads else 0))

        return (padding_mode.function(data, pads=pads, mode=mode, value=value),)


@opset('', 3)
class _V3(_V2):
    pass


@opset('', 4)
class _V4(_V3):

    @staticmethod
    @operator('Concat')
    def concat(*inputs, **kwargs):
        try:
            axis = kwargs['axis']
        except KeyError:
            six.raise_from(ValueError('Concatenation axis must be explicitly specified'), None)
        return (op.concatenate(inputs, axis),)


OPSETS = [_V1, _V2, _V3, _V4]

# As a special case for the ONNX operator set, we define a default version:
# this is the version of the ONNX operator set that will be loaded when there
# is no operator set defined by the model.  It should typically be the highest
# version of the operator set defined by this module.
DEFAULT_VERSION = 4
