import unittest, submitted, os, random
from gradescope_utils.autograder_utils.decorators import weight
import numpy as np
import torch
import torch.nn as nn

class TestStep(unittest.TestCase):
    @weight(2.5)
    def test_linear_init(self):
        input_dimension = random.randint(2,25)
        output_dimension = random.randint(2,25)
        layer = submitted.LineEar(input_dimension, output_dimension)
        weight_shape = layer.weight.shape
        expected_weight_shape = torch.Size([output_dimension, input_dimension])
        self.assertEqual(weight_shape,
                         expected_weight_shape,
                         msg='Expected weight shape [%d,%d]'%expected_weight_shape
                           + 'got weight shape [%d,%d]'%weight_shape)
        bias_shape = layer.bias.shape
        expected_bias_shape = torch.Size([output_dimension])
        self.assertEqual(bias_shape,
                         expected_bias_shape,
                         msg='Expected bias shape [%d,]'%expected_bias_shape
                           + 'got bias shape [%d,]'%bias_shape)

    @weight(2.5)
    def test_linear_forward(self):
        data_dict = torch.load('visible_linear.pkl')
        input_tensor = data_dict['input']
        expected_output_tensor = data_dict['output']
        layer = submitted.LineEar(input_tensor.shape[-1], expected_output_tensor.shape[-1])
        layer.load_state_dict(data_dict['params'])
        output_tensor = layer(input_tensor)
        output_tensor_differences = torch.sum(torch.abs(output_tensor-expected_output_tensor))/torch.sum(torch.abs(expected_output_tensor))
        self.assertLess(output_tensor_differences.item(),0.04,
                        msg="Output off by more than 4%")

    @weight(2.5)
    def test_lstm_init(self):
        input_dimension = random.randint(2,25)
        hidden_dimension = random.randint(2,25)
        num_layers = random.randint(2,5)
        layer = submitted.EllEssTeeEmm(input_dimension, hidden_dimension, num_layers, True)
        self.lstm_common_check(input_dimension, hidden_dimension, num_layers, True, layer)

    def lstm_common_check(self, input_dimension, hidden_dimension, layer_count, bidi, actual_lstm):
        direction_list = ["forward_layers", "reverse_layers"] if bidi else ["forward_layers"]
        for direction in direction_list:
            layer_stack = getattr(actual_lstm, direction)
            for index in range(layer_count):
                if index == 0:
                    target_input_dim = input_dimension
                elif bidi:
                    target_input_dim = 2*hidden_dimension
                else:
                    target_input_dim = hidden_dimension
                current_layer = layer_stack[index]
                self.assertEqual(current_layer.weight_ih.shape,
                                 torch.Size([4*hidden_dimension, target_input_dim]),
                                 msg="weight_ih shape is incorrect for cell at layer %d in %s"%(index, direction))
                self.assertEqual(current_layer.weight_hh.shape,
                                 torch.Size([4*hidden_dimension, hidden_dimension]),
                                 msg="weight_hh shape is incorrect for cell at layer %d in %s"%(index, direction))
                for variable_name in ["bias_ih", "bias_hh"]:
                    self.assertEqual(getattr(current_layer,variable_name).shape,
                                     torch.Size([4*hidden_dimension,]),
                                     msg="%s shape is incorrect for cell at layer %d in %s"%(variable_name, index, direction))

    @weight(10.)
    def test_lstm_forward(self):
        data_dict = torch.load('visible_lstm.pkl')
        input_tensor = data_dict['input']
        expected_output_tensor = data_dict['output']
        layer = submitted.EllEssTeeEmm(input_tensor.shape[-1], expected_output_tensor.shape[-1] // 2, data_dict['layer_count'], True)
        layer.load_state_dict(data_dict['params'])
        output_tensor = layer(input_tensor)
        output_tensor_differences = torch.sum(torch.abs(output_tensor-expected_output_tensor))/torch.sum(torch.abs(expected_output_tensor))
        self.assertLess(output_tensor_differences.item(),0.04,
                        msg="Output off by more than 4%")

    def gru_common_check(self, input_dimension, hidden_dimension, layer_count, actual_gru):
        layer_stack = actual_gru.forward_layers
        for index in range(layer_count):
            if index == 0:
                target_input_dim = input_dimension
            else:
                target_input_dim = hidden_dimension
            current_layer = layer_stack[index]
            self.assertEqual(current_layer.weight_ih.shape,
                             torch.Size([3*hidden_dimension, target_input_dim]),
                             msg="weight_ih shape is incorrect for cell at layer %d"%(index,))
            self.assertEqual(current_layer.weight_hh.shape,
                             torch.Size([3*hidden_dimension, hidden_dimension]),
                             msg="weight_hh shape is incorrect for cell at layer %d"%(index,))
            for variable_name in ["bias_ih", "bias_hh"]:
                self.assertEqual(getattr(current_layer,variable_name).shape,
                                 torch.Size([3*hidden_dimension,]),
                                 msg="%s shape is incorrect for cell at layer %d"%(variable_name, index,))

    @weight(2.5)
    def test_gru_init(self):
        input_dimension = random.randint(2,25)
        hidden_dimension = random.randint(2,25)
        num_layers = random.randint(2,5)
        layer = submitted.GeeArrYou(input_dimension, hidden_dimension, num_layers, 0.)
        self.gru_common_check(input_dimension, hidden_dimension, num_layers, layer)

    @weight(7.5)
    def test_gru_forward(self):
        data_dict = torch.load('visible_gru.pkl')
        input_tensor = data_dict['input']
        expected_output_tensor = data_dict['output']
        layer = submitted.GeeArrYou(input_tensor.shape[-1], expected_output_tensor.shape[-1], data_dict['layer_count'], 0.)
        layer.load_state_dict(data_dict['params'])
        output_tensor = layer(input_tensor)
        output_tensor_differences = torch.sum(torch.abs(output_tensor-expected_output_tensor))/torch.sum(torch.abs(expected_output_tensor))
        self.assertLess(output_tensor_differences.item(),0.04,
                        msg="Output off by more than 4%")

    @weight(2.5)
    def test_encoder_init(self):
        input_dim = random.randint(25,45)
        input_length = random.randint(25,45)
        output_dim = random.randint(20,50)
        layer = submitted.Encoder(output_dim, input_dim, 20)
        self.lstm_common_check(512, output_dim, 2, True, layer.recurrent)
        for index, module in enumerate(layer.convolutions):
            if isinstance(module, nn.Conv1d):
                if index == 0:
                    expected_in_channels = 80 + input_dim
                else:
                    expected_in_channels = 512
                self.assertEqual(module.in_channels, expected_in_channels, "Encoder Conv1d with incorrect input channels")
                self.assertEqual(module.out_channels, 512, "Encoder Conv1d with incorrect output channels")
                self.assertIn(module.kernel_size, [(5,), (5, 5)], msg="Encoder Conv1d with incorrect kernel size")
                self.assertIn(module.stride, [(1,), (1, 1)], msg="Encoder Conv1d with incorrect stride size")
                self.assertIn(module.padding, [(2,), (2, 2)], msg="Encoder Conv1d with incorrect padding size")
                self.assertIn(module.dilation, [(1,), (1, 1)], msg="Encoder Conv1d with incorrect dilation size")
                self.assertEqual(index % 3, 0, msg="Encoder Conv1d is not in the right place")
            elif isinstance(module, nn.BatchNorm1d):
                self.assertEqual(module.num_features, 512, "Encoder BatchNorm1d with incorrect feature size")
                self.assertEqual(index % 3, 1, msg="Encoder BatchNorm1d is not in the right place")
            elif isinstance(module, nn.ReLU):
                self.assertEqual(index % 3, 2, msg="Encoder ReLU is not in the right place")

    @weight(5.)
    def test_encoder_forward(self):
        data_dict = torch.load('visible_encoder.pkl')
        input_tensor = data_dict['input']
        expected_output_tensor = data_dict['output']
        layer = submitted.Encoder(data_dict['output_dim'], data_dict['input_dim'], 20)
        layer.load_state_dict(data_dict['params'])
        output_tensor = layer(input_tensor)
        output_tensor_differences_0 = torch.sum(torch.abs(output_tensor[0]-expected_output_tensor[0]))/torch.sum(torch.abs(expected_output_tensor[0]))
        self.assertLess(output_tensor_differences_0,0.04,
                        msg="Forward output off by more than 4%")
        output_tensor_differences_1 = torch.sum(torch.abs(output_tensor[1]-expected_output_tensor[1]))/torch.sum(torch.abs(expected_output_tensor[1]))
        self.assertLess(output_tensor_differences_1,0.04,
                        msg="Reverse output off by more than 4%")

    @weight(2.5)
    def test_decoder_init(self):
        input_dim = random.randint(25,45)
        input_length = random.randint(25,45)
        dim_neck = random.randint(20,40)
        pre_dim = random.randint(12,32)
        layer = submitted.Decoder(dim_neck, input_dim, pre_dim)
        self.lstm_common_check(dim_neck*2+input_dim, pre_dim, 1, False, layer.recurrent1)
        for index, module in enumerate(layer.convolutions):
            if isinstance(module, nn.Conv1d):
                expected_in_channels = 512
                self.assertEqual(module.in_channels, pre_dim, "Decoder Conv1d with incorrect input channels")
                self.assertEqual(module.out_channels, pre_dim, "Decoder Conv1d with incorrect output channels")
                self.assertIn(module.kernel_size, [(5,), (5, 5)], msg="Decoder Conv1d with incorrect kernel size")
                self.assertIn(module.stride, [(1,), (1, 1)], msg="Decoder Conv1d with incorrect stride size")
                self.assertIn(module.padding, [(2,), (2, 2)], msg="Decoder Conv1d with incorrect padding size")
                self.assertIn(module.dilation, [(1,), (1, 1)], msg="Decoder Conv1d with incorrect dilation size")
                self.assertEqual(index % 3, 0, msg="Decoder Conv1d is not in the right place")
            elif isinstance(module, nn.BatchNorm1d):
                self.assertEqual(module.num_features, pre_dim, "Decoder BatchNorm1d with incorrect feature size")
                self.assertEqual(index % 3, 1, msg="Decoder BatchNorm1d is not in the right place")
            elif isinstance(module, nn.ReLU):
                self.assertEqual(index % 3, 2, msg="Decoder ReLU is not in the right place")
        self.lstm_common_check(pre_dim, 1024, 2, False, layer.recurrent2)
        self.assertEqual(layer.fc_projection.weight.shape, torch.Size([80, 1024]),
            msg="Expected Decoder fc_projection weight vector of size [80, 1024], got one of size [%d,%d]"%layer.fc_projection.weight.shape)
        self.assertEqual(layer.fc_projection.bias.shape, torch.Size([80]),
            msg="Expected Decoder fc_projection bias vector of size 80, got one of size %d"%layer.fc_projection.bias.shape)

    @weight(2.5)
    def test_decoder_forward(self):
        data_dict = torch.load('visible_decoder.pkl')
        input_tensor = data_dict['input']
        expected_output_tensor = data_dict['output']
        layer = submitted.Decoder(data_dict['neck_dim'], data_dict['input_dim'], data_dict['pre_dim'])
        layer.load_state_dict(data_dict['params'])
        output_tensor = layer(input_tensor)
        output_tensor_differences = torch.sum(torch.abs(output_tensor-expected_output_tensor))/torch.sum(torch.abs(expected_output_tensor))
        self.assertLess(output_tensor_differences.item(),0.04,
                        msg="Output off by more than 4%")

    @weight(2.5)
    def test_postnet_init(self):
        layer = submitted.Postnet()
        for index, module in enumerate(layer.convolutions):
            if isinstance(module, nn.Conv1d):
                expected_in_channels = 80 if index == 0 else 512
                expected_out_channels = 80 if index == 12 else 512
                self.assertEqual(module.in_channels, expected_in_channels, "Postnet Conv1d with incorrect input channels")
                self.assertEqual(module.out_channels, expected_out_channels, "Postnet Conv1d with incorrect output channels")
                self.assertIn(module.kernel_size, [(5,), (5, 5)], msg="Postnet Conv1d with incorrect kernel size")
                self.assertIn(module.stride, [(1,), (1, 1)], msg="Postnet Conv1d with incorrect stride size")
                self.assertIn(module.padding, [(2,), (2, 2)], msg="Postnet Conv1d with incorrect padding size")
                self.assertIn(module.dilation, [(1,), (1, 1)], msg="Postnet Conv1d with incorrect dilation size")
                self.assertEqual(index % 3, 0, msg="Postnet Conv1d is not in the right place")
            elif isinstance(module, nn.BatchNorm1d):
                self.assertEqual(module.num_features, expected_out_channels, "Postnet BatchNorm1d with incorrect feature size")
                self.assertEqual(index % 3, 1, msg="Postnet BatchNorm1d is not in the right place")
            elif isinstance(module, nn.ReLU):
                self.assertEqual(index % 3, 2, msg="Postnet ReLU is not in the right place")

    @weight(2.5)
    def test_postnet_forward(self):
        data_dict = torch.load('visible_postnet.pkl')
        input_tensor = data_dict['input']
        expected_output_tensor = data_dict['output']
        layer = submitted.Postnet()
        layer.load_state_dict(data_dict['params'])
        output_tensor = layer(input_tensor)
        output_tensor_differences = torch.sum(torch.abs(output_tensor-expected_output_tensor))/torch.sum(torch.abs(expected_output_tensor))
        self.assertLess(output_tensor_differences.item(),0.04,
                        msg="Output off by more than 4%")

    @weight(2.5)
    def test_speakerembedder_init(self):
        n_mels = random.randint(15,35)
        n_hid = random.randint(10,30)
        n_layers = random.randint(2,4)
        fc_dim = random.randint(20,40)
        layer = submitted.SpeakerEmbedderGeeArrYou(n_hid, n_mels, n_layers, fc_dim, 0.)
        self.gru_common_check(n_mels, n_hid, n_layers, layer.rnn_stack)
        self.assertEqual(layer.projection.weight.shape, torch.Size([fc_dim, n_hid]),
            msg="Expected SpeakerEmbedderGeeArrYou projection weight vector of size [%d,%d], got one of size [%d,%d]"%(fc_dim, n_hid, layer.projection.weight.shape[0], layer.projection.weight.shape[1]))
        self.assertEqual(layer.projection.bias.shape, torch.Size([fc_dim]),
            msg="Expected SpeakerEmbedderGeeArrYou projection bias vector of size %d, got one of size %d"%(fc_dim, layer.projection.bias.shape[0]))

    @weight(2.5)
    def test_speakerembedder_forward(self):
        data_dict = torch.load('visible_speakerembedder.pkl')
        input_tensor = data_dict['input']
        expected_output_tensor = data_dict['output']
        n_hid = input_tensor.shape[1]
        n_mels = input_tensor.shape[2]
        layer = submitted.SpeakerEmbedderGeeArrYou(n_hid, n_mels, data_dict['layer_count'], data_dict['fc_dim'], 0.)
        layer.load_state_dict(data_dict['params'])
        output_tensor = layer(input_tensor)
        output_tensor_differences = torch.sum(torch.abs(output_tensor-expected_output_tensor))/torch.sum(torch.abs(expected_output_tensor))
        self.assertLess(output_tensor_differences.item(),0.04,
                        msg="Output off by more than 4%")
