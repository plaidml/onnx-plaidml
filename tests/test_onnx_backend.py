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

        # Unimplemented functionality
        self.exclude('test_ConvTranspose2d_')  # Requires ConvTranspose
        self.exclude('test_ReflectionPad2d_')  # Requires Pad(reflect)
        self.exclude('test_ReplicationPad2d_')  # Requires Pad(edge)
        self.exclude('test_acos_')  # Requires ACos
        self.exclude('test_asin_')  # Requires ASin
        self.exclude('test_atan_')  # Requires ATan
        self.exclude('test_averagepool_2d_precomputed_same_upper_')  # Requires Pad(same_upper)
        self.exclude('test_averagepool_2d_same_lower_')  # Requires Pad(same_lower)
        self.exclude('test_averagepool_2d_same_upper_')  # Requires Pad(same_upper)
        self.exclude('test_depthtospace_')  # Requires DepthToSpace
        self.exclude('test_edge_pad_')  # Requires Pad(edge)
        self.exclude('test_gru_')  # Requires GRU
        self.exclude('test_lstm_')  # Requires LSTM
        self.exclude('test_maxpool_2d_precomputed_same_lower_')  # Requires Pad(same_lower)
        self.exclude('test_maxpool_2d_precomputed_same_upper_')  # Requires Pad(same_upper)
        self.exclude('test_maxpool_2d_same_lower_')  # Requires Pad(same_lower)
        self.exclude('test_maxpool_2d_same_upper_')  # Requires Pad(same_upper)
        self.exclude('test_gather_1_')  # Requires Gather on non-outermost axis
        self.exclude('test_hardmax_one_hot_')  # Requires filtered Hardmax
        self.exclude('test_reflect_pad_')  # Requires Pad(reflect)
        self.exclude('test_rnn_')  # Requires RNN
        self.exclude('test_simple_rnn_')  # Requires RNN
        self.exclude('test_sin_')  # Requires Sin
        self.exclude('test_tan_')  # Requires Tan
        self.exclude('test_tile_')  # Requires Tile
        self.exclude('test_top_k_')  # Requires TopK
        self.exclude('test_upsample_nearest_')  # Requires Upsample
        self.exclude('test_Upsample_nearest_2d_')  # Requires Upsample
        self.exclude('test_Upsample_nearest_scale_2d_')  # Requires Upsample
        self.exclude('test_Upsample_nearest_tuple_2d_')  # Requires Upsample
        self.exclude('test_operator_convtranspose_')  # Requires ConvTranspose
        self.exclude('test_operator_lstm_')  # Requires ConstantFill
        self.exclude('test_operator_rnn_')  # Requires ConstantFill
        self.exclude('test_operator_repeat_')  # Requires Tile
        self.exclude('test_operator_pad_')  # Requires additional padding modes
        self.exclude('test_reshape_extended_dims_')  # Requires V5 reshape semantics
        self.exclude('test_reshape_negative_dim_')  # Requires V5 reshape semantics
        self.exclude('test_reshape_one_dim_')  # Requires V5 reshape semantics
        self.exclude('test_reshape_reduced_dims_')  # Requires V5 reshape semantics
        self.exclude('test_reshape_reordered_dims_')  # Requires V5 reshape semantics
        self.exclude('test_PixelShuffle_')  # Requires V5 reshape semantics

        # Tests that are correct, but take too long on CI.
        self.exclude('test_bvlc_alexnet_')
        self.exclude('test_densenet121_')
        self.exclude('test_inception_v1_')
        self.exclude('test_inception_v2_')
        self.exclude('test_resnet50_')
        self.exclude('test_shufflenet_')
        self.exclude('test_vgg19_')
        self.exclude('test_zfnet512_')

    def _add_test(self, category, test_name, test_func, report_item, devices=None):
        if not devices:
            devices = tuple(opb.PlaidMLBackend.device_configs.keys())
        return super(BackendTest, self)._add_test(category, test_name, test_func, report_item,
                                                  devices)


# Import test cases for unittest
globals().update(BackendTest(opb, __name__).enable_report().test_cases)

if __name__ == '__main__':
    unittest.main()
