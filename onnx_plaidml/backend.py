# Copyright Vertex.AI.
"""The PlaidML ONNX Backend.

To use this, you'll need to have PlaidML installed:

    pip install plaidml

"""

from collections import namedtuple, defaultdict
import functools
import json
import operator
import struct

import numpy as np
import onnx.backend.base
from onnx import onnx_pb2
import onnx_plaidml
from onnx_plaidml import opset_onnx
from onnx_plaidml import opset_util
import plaidml
import plaidml.settings
import plaidml.tile as tile
import six


def _as_input_id(name):
    """Returns an identifier for an input name in a graph."""
    return 'I' + name


def _as_output_id(name):
    """Returns an identifier for an output name in a graph."""
    return 'O' + name


# The opsets known to the backend.
#
# Each opset is an object that's been decorated by onnx_plaidml.opset.opset_id;
# this provides the opset's domain and version.
#
# Some of the object's attributes will have been decorated by onnx_plaidml.opset.operator;
# these are the actual operation method bindings.
#
# TODO: Support registered entrypoints for loading opsets.
_KNOWN_OPSETS = dict([(o.onnx_plaidml_opset_id, o) for o in opset_onnx.OPSETS])


def _load_ops(opset_ids=None):
    """Builds the operator ID dictionary to use for a model.
    
    Args:
        opset_ids ([onnx_pb2.OperatorSetIdProto]): The operator sets imported by the model.
    
    Returns:
        (domain, operator_name) -> function: The map from (domain, operator name) to operator
                                             implementations.
    """
    versions = defaultdict(int)
    if not opset_ids:
        opset_id = onnx_pb2.OperatorSetIdProto()
        opset_id.domain = ''
        opset_id.version = opset_onnx.DEFAULT_VERSION
        opset_ids = [opset_id]
    for opset_id in opset_ids:
        if versions[opset_id.domain] < opset_id.version:
            versions[opset_id.domain] = opset_id.version
    ops = {}
    for domain, version in versions.items():
        try:
            opset = _KNOWN_OPSETS[(domain, version)]
        except KeyError:
            six.raise_from(
                NotImplementedError(
                    '"{}"/version={}" is not implemented by the PlaidML ONNX backend'.format(
                        domain, version)), None)
        for name in dir(opset):
            attr = getattr(opset, name)
            try:
                opname = attr.onnx_plaidml_operator_name
            except AttributeError:
                continue
            ops[(domain, opname)] = attr
    return ops


# Translations from ONNX attributes to protobuf field getters.
_ONNX_ATTRTYPE_TO_GETTER = {
    onnx_pb2.AttributeProto.FLOAT: operator.attrgetter('f'),
    onnx_pb2.AttributeProto.INT: operator.attrgetter('i'),
    onnx_pb2.AttributeProto.STRING: operator.attrgetter('s'),
    onnx_pb2.AttributeProto.TENSOR: operator.attrgetter('t'),
    onnx_pb2.AttributeProto.GRAPH: operator.attrgetter('g'),
    onnx_pb2.AttributeProto.FLOATS: operator.attrgetter('floats'),
    onnx_pb2.AttributeProto.INTS: operator.attrgetter('ints'),
    onnx_pb2.AttributeProto.STRINGS: operator.attrgetter('strings'),
    onnx_pb2.AttributeProto.TENSORS: operator.attrgetter('tensors'),
    onnx_pb2.AttributeProto.GRAPHS: operator.attrgetter('graphs'),
}


def _get_device_configs(ctx):
    configs = {}
    type_indicies = {}
    for config in plaidml.devices(ctx):
        configs[config.id.decode()] = config
    return configs


class PlaidMLBackendRep(onnx.backend.base.BackendRep):
    """Implements onnx.backend.base.BackendRep for the PlaidML backend."""

    def __init__(self, model, ctx, dev, func, input_valinfos):
        self._model = model
        self._ctx = ctx
        self._dev = dev
        self._func = func
        self._invoker = None
        self._input_valinfos = input_valinfos

    def save(self, filename):
        """Saves the function to a TILE-format file.
        
        Args:
            filename (str): The name of the file to write.
        """
        self._func.save(filename)

    def run(self, inputs, **kwargs):
        if not self._invoker:
            self._invoker = plaidml.Invoker(self._ctx, self._func)

        # TODO: Use the datatype from the model.
        for inp, valinfo in zip(inputs, self._input_valinfos):
            val = tile.Value.from_python_value(inp, ctx=self._ctx, dev=self._dev).var
            self._invoker.set_input(_as_input_id(valinfo.name), val)
        outputs = []
        for valinfo in self._model.graph.output:
            shape = self._invoker.get_output_shape(_as_output_id(valinfo.name))
            output = plaidml.Tensor(self._dev, shape)
            outputs.append(output)
            self._invoker.set_output(_as_output_id(valinfo.name), output)

        self._invoker.invoke()

        return [output.as_ndarray(self._ctx) for output in outputs]


class PlaidMLBackend(onnx.backend.base.Backend):
    ctx = plaidml.Context()
    device_configs = _get_device_configs(ctx)
    ops = opset_onnx.OPSETS

    @classmethod
    def prepare(cls, model, device=None, **kwargs):
        if device is None:
            device = cls._get_default_device()
        return super(PlaidMLBackend, cls).prepare(model, device=device, **kwargs)

    @classmethod
    def run_model(cls, model, inputs, device=None, **kwargs):
        if device is None:
            device = cls._get_default_device()
        return super(PlaidMLBackend, cls).run_model(model, device=device, **kwargs)

    @classmethod
    def run_node(cls, node, inputs, device=None, outputs_info=None, **kwargs):
        if device is None:
            device = cls._get_default_device()
        return super(PlaidMLBackend, cls).run_node(
            node, inputs, device=device, output_info=outputs_info, **kwargs)

    @classmethod
    def _get_default_device(cls):
        device_ids = plaidml.settings.device_ids
        if not device_ids:
            six.raise_from(
                onnx_plaidml.DeviceNotFoundError(None, list(cls.device_configs.keys())), None)
        if len(device_ids) != 1:
            six.raise_from(onnx_plaidml.TooManyDefaultDevicesError(device_ids), None)
        if not cls.supports_device(device_ids[0]):
            six.raise_from(
                onnx_plaidml.DeviceNotFoundError(device_id, list(cls.device_configs.keys())), None)
        return device_ids[0]

    @classmethod
    def _apply_node(cls, ops, node, bindings):
        """
        Applies an operation given the supplied bindings.

        Args:
            ops: The operation definitions.
            node: The ONNX NodeProto describing the operation node.
            bindings: The binding map.

        Returns:
            None.  Updates from the operation are added to the binding map.
        """
        attrs = {}
        for attr in node.attribute:
            if attr.type:
                attrs[attr.name] = _ONNX_ATTRTYPE_TO_GETTER[attr.type](attr)
            elif attr.HasField('f'):
                attrs[attr.name] = attr.f
            elif attr.HasField('i'):
                attrs[attr.name] = attr.i
            elif attr.HasField('s'):
                attrs[attr.name] = attr.s
            elif attr.HasField('t'):
                attrs[attr.name] = attr.t
            elif attr.HasField('g'):
                attrs[attr.name] = attr.g
            elif attr.floats:
                attrs[attr.name] = attr.floats
            elif attr.ints:
                attrs[attr.name] = attr.ints
            elif attr.strings:
                attrs[attr.name] = attr.strings
            elif attr.tensors:
                attrs[attr.name] = attr.tensors
            elif attr.graphs:
                attrs[attr.name] = attr.graphs
            else:
                attrs[attr.name] = 0

        input_vars = [bindings[name] for name in node.input]
        try:
            operation = ops[(node.domain, node.op_type)]
        except KeyError:
            six.raise_from(
                NotImplementedError(
                    '"{}"/"{}" is not implemented by the PlaidML ONNX backend'.format(
                        node.domain, node.op_type)), None)

        output_vars = operation(*input_vars, **attrs)

        for (name, var) in zip(node.output, output_vars):
            bindings[name] = var

    @classmethod
    def prepare(cls, model, device=None, **kwargs):
        if not device:
            device = cls._get_default_device()
        super(PlaidMLBackend, cls).prepare(model, device, **kwargs)
        ops = _load_ops(model.opset_import)
        try:
            config = cls.device_configs[device]
        except KeyError:
            six.raise_from(
                onnx_plaidml.DeviceNotFoundError(device, list(cls.device_configs.keys())), None)
        dev = plaidml.Device(cls.ctx, config)

        bindings = {}
        graph = model.graph

        initializers = set()
        for initializer in graph.initializer:
            initializers.add(initializer.name)
            bindings[initializer.name] = tile.Value.from_var(
                opset_util.onnx_tensor_to_plaidml_tensor(cls.ctx, dev, initializer),
                initializer.dims, opset_util.ONNX_DTYPE_TO_PLAIDML[initializer.data_type])

        input_valinfos = []
        for valinfo in graph.input:
            if valinfo.name not in initializers:
                bindings[valinfo.name] = opset_util.onnx_type_to_placeholder_value(valinfo.type)
                input_valinfos.append(valinfo)

        for node in graph.node:
            cls._apply_node(ops, node, bindings)

        func = tile.compose(
            cls.ctx,
            dev,
            inputs=[(_as_input_id(inp.name), bindings[inp.name]) for inp in graph.input
                    if inp.name not in initializers],
            outputs=[(_as_output_id(outp.name), bindings[outp.name]) for outp in graph.output])

        return PlaidMLBackendRep(model, cls.ctx, dev, func, input_valinfos)

    @classmethod
    def run_node(cls, node, inputs, device=None):
        if not device:
            device = cls._get_default_device()
        super(PlaidMLBackend, cls).run_node(node, inputs, device)
        dev = plaidml.Device(cls.ctx, cls.device_configs[device])
        try:
            bindings = {}

            for (name, py_input) in zip(node.input, inputs):
                bindings[name] = tile.Value.from_python_value(py_input, ctx=cls.ctx, dev=dev)

            cls._apply_node(_load_ops(), node, bindings)

            func = tile.compose(
                cls.ctx,
                dev,
                inputs=[],
                outputs=[(_as_output_id(name), bindings[name]) for name in node.output])

            invoker = plaidml.Invoker(cls.ctx, func)

            tensors = [
                plaidml.Tensor(dev, invoker.get_output_shape(_as_output_id(name)))
                for name in node.output
            ]
            for (name, tensor) in zip(node.output, tensors):
                invoker.set_output(_as_output_id(name), tensor)

            invoker.invoke()

            return [tensor.as_ndarray(cls.ctx) for tensor in tensors]

        finally:
            dev.close()

    @classmethod
    def supports_device(cls, device_id):
        return device_id in cls.device_configs


# Bind backend class methods.
# pylint: disable=invalid-name

prepare = PlaidMLBackend.prepare
run_node = PlaidMLBackend.run_node
run_model = PlaidMLBackend.run_model
supports_device = PlaidMLBackend.supports_device
