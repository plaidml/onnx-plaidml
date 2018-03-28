# Copyright Vertex.AI.
"""ONNX-PlaidML ONNX operator utilities."""

import operator
import struct

from onnx import onnx_pb2
import plaidml
import plaidml.tile as tile
import six

# Translation from ONNX dtypes to PlaidML dtypes.
ONNX_DTYPE_TO_PLAIDML = {
    onnx_pb2.TensorProto.FLOAT: plaidml.DType.FLOAT32,
    onnx_pb2.TensorProto.UINT8: plaidml.DType.UINT8,
    onnx_pb2.TensorProto.INT8: plaidml.DType.INT8,
    onnx_pb2.TensorProto.UINT16: plaidml.DType.UINT16,
    onnx_pb2.TensorProto.INT16: plaidml.DType.INT16,
    onnx_pb2.TensorProto.INT32: plaidml.DType.INT32,
    onnx_pb2.TensorProto.INT64: plaidml.DType.INT64,
    onnx_pb2.TensorProto.BOOL: plaidml.DType.BOOLEAN,
    onnx_pb2.TensorProto.FLOAT16: plaidml.DType.FLOAT16,
    onnx_pb2.TensorProto.DOUBLE: plaidml.DType.FLOAT64,
    onnx_pb2.TensorProto.UINT32: plaidml.DType.UINT32,
    onnx_pb2.TensorProto.UINT64: plaidml.DType.UINT64,
}

_ONNX_TENSOR_DATATYPE_TO_GETTER = {
    onnx_pb2.TensorProto.FLOAT: operator.attrgetter('float_data'),
    onnx_pb2.TensorProto.DOUBLE: operator.attrgetter('float_data'),
    onnx_pb2.TensorProto.UINT8: operator.attrgetter('int32_data'),
    onnx_pb2.TensorProto.INT8: operator.attrgetter('int32_data'),
    onnx_pb2.TensorProto.UINT16: operator.attrgetter('int32_data'),
    onnx_pb2.TensorProto.INT16: operator.attrgetter('int32_data'),
    onnx_pb2.TensorProto.UINT32: operator.attrgetter('uint64_data'),
    onnx_pb2.TensorProto.INT32: operator.attrgetter('int32_data'),
    onnx_pb2.TensorProto.UINT64: operator.attrgetter('uint64_data'),
    onnx_pb2.TensorProto.INT64: operator.attrgetter('int64_data'),
    onnx_pb2.TensorProto.STRING: operator.attrgetter('string_data'),
    onnx_pb2.TensorProto.BOOL: operator.attrgetter('int32_data'),
}

_ONNX_TENSOR_DATATYPE_TO_UNPACK_TEMPLATE = {
    onnx_pb2.TensorProto.FLOAT: '<{}f',
    onnx_pb2.TensorProto.DOUBLE: '<{}d',
    onnx_pb2.TensorProto.UINT8: '<{}B',
    onnx_pb2.TensorProto.INT8: '<{}b',
    onnx_pb2.TensorProto.UINT16: '<{}H',
    onnx_pb2.TensorProto.INT16: '<{}h',
    onnx_pb2.TensorProto.UINT32: '<{}I',
    onnx_pb2.TensorProto.INT32: '<{}i',
    onnx_pb2.TensorProto.UINT64: '<{}Q',
    onnx_pb2.TensorProto.INT64: '<{}q',
    onnx_pb2.TensorProto.BOOL: '<{}?',
}


def opset(domain, version):
    """Annotates a class as implementing an ONNX operator set.

    Args:
        domain (str): The domain of the operator set.
        version (int): The version of the operator set.
    """

    def _wrap(klass):
        klass.onnx_plaidml_opset_id = (domain, version)
        return klass

    return _wrap


def opset_op(name):
    """Annotates a method as implementing an ONNX operator.

    N.B. If @opset_op is used with @staticmethod or @classmethod, make
    sure that the @staticmethod or @classmethod wraps @opset_op, instead
    of the other way around.  The reason is that the ONNX-PlaidML backend
    applies the descriptor machinery when inspecting object attributes
    for @opset_op annotations, so the @opset_op annotation must be applied
    to the inner (actual function) object, not the descriptor.

    Args:
        name (str): The name of the operator.
    """

    def _wrap(func):
        func.onnx_plaidml_operator_name = name
        return func

    return _wrap


def onnx_tensor_to_plaidml_tensor(ctx, dev, tensor):
    """
    Converts an ONNX tensor proto to a TILE tensor value.
    
    Args:
        ctx (plaidml.Context): The context for the value creation.
        dev (plaidml.Device): The PlaidML device on which the value will be used.
        tensor (onnx_pb2.TensorProto): The tensor data.
    
    Returns:
        plaidml.Var: A variable describing the tensor value.
    """
    if not tensor.data_type in ONNX_DTYPE_TO_PLAIDML:
        six.raise_from(
            NotImplementedError(
                'ONNX data type {} is not yet implemented by the PlaidML ONNX backend'.format(
                    onnx_pb2.TensorProto.DataType.Name(tensor.data_type))), None)
    dtype = ONNX_DTYPE_TO_PLAIDML[tensor.data_type]
    var = plaidml.Tensor(dev, plaidml.Shape(ctx, dtype, *tensor.dims))
    with var.mmap_discard(ctx) as view:
        # TODO: Map ONNX datatypes to strings to use for conversion.
        # ALso, consider precompiling the structs.
        if tensor.raw_data:
            view[:len(view)] = struct.unpack_from(
                _ONNX_TENSOR_DATATYPE_TO_UNPACK_TEMPLATE[tensor.data_type].format(len(view)),
                tensor.raw_data)
        else:
            view[:len(view)] = _ONNX_TENSOR_DATATYPE_TO_GETTER[tensor.data_type](tensor)
        view.writeback()
    return var


def onnx_type_to_placeholder_value(type_proto):
    """
    Converts an ONNX type to a TILE placeholder Value.

    Args:
        type_proto (onnx_pb2.TypeProto): The ONNX type proto.

    Returns:
        tile.Value: The corresponding TILE Value.
    """
    dtype = ONNX_DTYPE_TO_PLAIDML[type_proto.tensor_type.elem_type]
    dims = tuple([dim.dim_value for dim in type_proto.tensor_type.shape.dim])
    return tile.Value.from_dimensions(dims, dtype=dtype)
