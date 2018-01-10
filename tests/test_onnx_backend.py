# Copyright Vertex.AI.

import json
import os
import unittest

import onnx
import onnx.backend.test
import onnx_plaidml.backend as opb

import plaidml

# Register plugins
pytest_plugins = 'onnx.backend.test.report',


class BackendTest(onnx.backend.test.BackendTest):

    def __init__(self, backend, name):
        super(BackendTest, self).__init__(backend, name)
        self.exclude('GLU')  # Requires Split
        self.exclude('ReflectionPad2d_[gc]pu')  # Requires Pad(reflect)
        self.exclude('ReplicationPad2d_[gc]pu')  # Requires Pad(edge)
        self.exclude('edge_pad_[gc]pu')  # Requires Pad(edge)
        self.exclude('reflect_pad_[gc]pu')  # Requires Pad(reflect)

        if onnx.version.version == '1.0.0' or onnx.version.version == '1.0.1':
            # ONNX changed the filename of the protobuf model description within their sample model
            # packages, and uploaded the updated sample model packages, without updating the
            # published wheels that used the old filename.  TL;DR: If you're using onnx <= 1.0.1,
            # the following tests don't work, unless you go into your copy of
            # onnx/backend/test/runner/__init__.py, look for the place where _add_model_test() uses
            # the string 'model.pb', and change it to 'model.onnx'.  (That then causes other tests
            # to break, so you don't want to comment these out; change them from 'exclude' to
            # 'include', so that the other tests aren't run.)
            self.exclude('bvlc_alexnet_[gc]pu')  # Requires compatible model
            self.exclude('densenet121_[gc]pu')  # Requires compatible model
            self.exclude('inception_v1_[gc]pu')  # Requires compatible model
            self.exclude('inception_v2_[gc]pu')  # Requires compatible model
            self.exclude('resnet50_[gc]pu')  # Requires compatible model
            self.exclude('shufflenet_[gc]pu')  # Requires compatible model
            self.exclude('squeezenet_[gc]pu')  # Requires compatible mode
            self.exclude('vgg16_[gc]pu')  # Requires compatible model
            self.exclude('vgg19_[gc]pu')  # Requires compatible model

    def _add_test(self, category, test_name, test_func, report_item, devices=None):
        if not devices:
            devices = tuple(opb.PlaidMLBackend.device_configs.keys())
        return super(BackendTest, self)._add_test(category, test_name, test_func, report_item,
                                                  devices)


# Import test cases for unittest
globals().update(BackendTest(opb, __name__).enable_report().test_cases)

if __name__ == '__main__':
    unittest.main()
