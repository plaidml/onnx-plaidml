# ONNX PlaidML

[PlaidML](https://github.com/plaidml/plaidml) is an efficient computation engine for deep neural networks; [ONNX](https://onnx.ai/) is an interchange format for neural networks, and provides a straightforward API for integration with a variety of neural network frameworks.

This package bridges between the two.  It consists of:

* A PlaidML ONNX backend, allowing every neural network framework that uses ONNX as a high-level backend to use PlaidML as its low-level computation layer.

* An ONNX-to-PlaidML file translator, making it easy for applications to run models directly via the PlaidML API -- improving performance and reducing dependencies (e.g. Python).

## Current Status

[![Build Status](https://travis-ci.org/earhart/onnx-plaidml.svg?branch=master)](https://travis-ci.org/plaidml/onnx-plaidml)

As of this release, we've implemented the operations required for the networks we focus on, operations whose definitions are relatively straightforward and unambiguous, and most of the operations exercised by the ONNX backend tests.  A number of operations aren't well-covered by the ONNX backend tests, and we haven't implemented our own tests for them (we think it'd be more useful to add them to ONNX); there may be bugs.

We think we've implemented enough operations to be useful, but we know we're missing a few -- in particular, edge and reflection padding modes for convolutions.  If you run into any operations you need, please let us know -- or if you'd like to implement them, please see the section on [Development](#development).

## Installation

`pip install onnx-plaidml`

If you're installing PlaidML for the first time, you'll want to run `plaidml-setup` to set your default computation device and verify that everything's working as it should.

Note that ONNX itself has a few prerequisites; you'll want to read the [ONNX Installation Instructions](https://github.com/onnx/onnx).

## Usage

### Loading and running a model

Here's some sample Python code to load and run resnet50.  Note that we explicitly request the GPU device -- unless your default compute device is a CPU, the default device setting ("CPU") won't work.

```python
import onnx
import onnx_plaidml.backend
import numpy as np

# Assumption: test_images.npz contains a tensor, "inputs", whose
# most-major dimension strides through images to be supplied to the model.
with np.load('test_images.npz') as img:
    model = onnx.load('resnet50_model.onnx')
    for batch in range(img['inputs'].shape[0]):
        data = img['inputs'][batch, :]
        output = onnx_plaidml.backend.run_model(model, [data], device='GPU')
        print(output)
```

### Caching the model construction

If you're going to be using the model multiple times (for instance, if you're going to be doing a series of inferences),
you'll probably want to cache the model construction.  (This is exactly how ONNX implements `run_model`.)

```python
import onnx
import onnx_plaidml.backend
import numpy as np

# Assumption: test_images.npz contains a tensor, "inputs", whose
# most-major dimension strides through images to be supplied to the model.
with np.load('test_images.npz') as img:
    model = onnx.load('resnet50_model.onnx')
    rep = onnx_plaidml.backend.prepare(model, device='GPU')
    for batch in range(img['inputs'].shape[0]):
        data = img['inputs'][batch, :]
        output = rep.run([data])
        print(output)
```

### Exporting an ONNX model to TILE

You can also convert an ONNX model to TILE, making it easy to use the model directly from the PlaidML C or C++ APIs:

    convert-onnx-to-tile <onnx_model.pb> <tile_model.tile>

## Development

We welcome contributions!  Please read the [PlaidML Contribution Guidelines](https://github.com/plaidml/plaidml/blob/master/CONTRIBUTING.md) for details on the process.

The ONNX PlaidML backend is essentially just a simple wrapper around PlaidML itself.  If you're implementing new operations, you'll definitely want to read the [Tile Tutorial](https://github.com/plaidml/plaidml/wiki/Tile-Tutorial) before diving in.  You'll also probably want to remove some of the exclusions in `tests/test_onnx_backend.py`.

The base ONNX operations are defined in `onnx_plaidml/opset_onnx.py`.  New versions of the base ONNX operations may be defined there; additional operation sets should go into their own files, and included in `onnx_plaidml/backend.py` for discovery during graph processing.

To run tests, use `python setup.py test`.  Note that you'll need to have already installed the ONNX prerequisites; see the [ONNX Installation Instructions](https://github.com/onnx/onnx).  Also, if you're using python3, you'll need to explicitly `pip3 install pytest-runner` first.

Explicitly excluding tests is fine, but we expect all enabled tests to pass for every checkin.
