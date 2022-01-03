from ctypes import c_char_p
from posixpath import relpath
from typing import Any, Tuple

from torch import Tensor, matmul, norm, randn, zeros, zeros_like, cat, stack, split, reshape
from torch._C import has_openmp
from torch.jit import _hide_source_ranges
from torch.nn import BatchNorm1d, Conv1d, Module, ModuleList, Parameter, ReLU, Sequential, Sigmoid, Tanh
import torch.nn.functional as F
from torch.nn.modules.rnn import LSTM, GRUCell, LSTMCell

from torchtyping import TensorType

################################################################################

class LineEar(Module):
    def __init__(self, input_size: int, output_size: int) -> None:
        """
        Sets up the following Parameters:
            self.weight - A Parameter holding the weights of the layer,
                of size (output_size, input_size).
            self.bias - A Parameter holding the biases of the layer,
                of size (output_size,).
        You may also set other instance variables at this point, but these are not strictly necessary.
        """
        super(LineEar, self).__init__()
        self.weight = Parameter(zeros((output_size, input_size)))
        self.bias = Parameter(zeros(output_size))
        

    def forward(self,
                inputs: TensorType["batch", "input_size"]) -> TensorType["batch", "output_size"]:
        """
        Performs forward propagation of the inputs.
        Input:
            inputs - the inputs to the cell.
        Output:
            outputs - the outputs from the cell.
        Note that all dimensions besides the last are preserved
            between inputs and outputs.
        """
        return inputs @ self.weight.T + self.bias

class EllEssTeeEmm(Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, bidirectional: bool=False) -> None:
        """
        Sets up the following:
            self.forward_layers - A ModuleList of num_layers EllEssTeeEmmCell layers.
                The first layer should have an input size of input_size
                    and an output size of hidden_size,
                while all other layers should have input and output both of size hidden_size.
        
        If bidirectional is True, then the following apply:
          - self.reverse_layers - A ModuleList of num_layers EllEssTeeEmmCell layers,
                of the exact same size and structure as self.forward_layers.
          - In both self.forward_layers and self.reverse_layers,
                all layers other than the first should have an input size of two times hidden_size.
        """
        super(EllEssTeeEmm, self).__init__()
        #initialize
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.judge = bidirectional
        #set up layers
        self.forward_layers = ModuleList()
        for n in range(num_layers):
            if n == 0:
                self.forward_layers.append(LSTMCell(input_size, hidden_size))
            else:
                self.forward_layers.append(LSTMCell(hidden_size, hidden_size))
        
        if bidirectional == True:
            self.forward_layers = ModuleList()
            self.reverse_layers = ModuleList()
            for n in range(num_layers):
                if n == 0:
                    self.forward_layers.append(LSTMCell(input_size, hidden_size))
                    self.reverse_layers.append(LSTMCell(input_size, hidden_size))
                else:
                    self.forward_layers.append(LSTMCell(hidden_size*2, hidden_size))
                    self.reverse_layers.append(LSTMCell(hidden_size*2, hidden_size)) 


    def forward(self,
                x: TensorType["batch", "length", "input_size"]) -> TensorType["batch", "length", "output_size"]:
        """
        Performs the forward propagation of an EllEssTeeEmm layer.
        Inputs:
           x - The inputs to the cell.
        Outputs:
           output - The resulting (hidden state) output h.
               If bidirectional was True when initializing the EllEssTeeEmm layer, then the "output_size"
               of the output should be twice the hidden_size.
               Otherwise, this "output_size" should be exactly the hidden size.
        """
        #initializing
        batch, length, input_size = x.shape

        #bidirectional condition
        if self.bidirectional:
            c = zeros(batch, length, self.hidden_size*2)
            h = zeros(batch, length, self.hidden_size*2)

            # first layer
            h[:,-1,self.hidden_size:], c[:,-1,self.hidden_size:] = self.reverse_layers[0](x[:,-1,:])
            h[:,0,:self.hidden_size], c[:,0,:self.hidden_size] = self.forward_layers[0](x[:,0,:])
            for l in range(1, length):
                h[:,l,:self.hidden_size], c[:,l,:self.hidden_size] = self.forward_layers[0](x[:,l,:], (h[:,l-1,:self.hidden_size], c[:,l-1,:self.hidden_size]))
                h[:,-l-1,self.hidden_size:], c[:,-l-1,self.hidden_size:] = self.reverse_layers[0](x[:,-l-1,:], (h[:,-l,self.hidden_size:], c[:,-l,self.hidden_size:]))

            # other layers
            for n in range(1, self.num_layers):
                h_ = zeros(batch, length, self.hidden_size*2)
                h_[:,0,:self.hidden_size], c[:,0,:self.hidden_size] = self.forward_layers[n](h[:,0,:])
                h_[:,-1,self.hidden_size:], c[:,-1,self.hidden_size:] = self.reverse_layers[n](h[:,-1,:])
                for l in range(1, length):
                    h_[:,l,:self.hidden_size], c[:,l,:self.hidden_size] = self.forward_layers[n](h[:,l,:], (h_[:,l-1,:self.hidden_size], c[:,l-1,:self.hidden_size]))
                    h_[:,-l-1,self.hidden_size:], c[:,-l-1,self.hidden_size:] = self.reverse_layers[n](h[:,-l-1,:], (h_[:,-l,self.hidden_size:], c[:,-l,self.hidden_size:]))
                h = h_

            return h

        #unidirectional condition
        else:
            c = zeros(batch, length, self.hidden_size)
            h = zeros(batch, length, self.hidden_size)
            
            # first layer
            h[:,0,:], c[:,0,:] = self.forward_layers[0](x[:,0,:])
            for l in range(1, length):
                h[:,l,:], c[:,l,:] = self.forward_layers[0](x[:,l,:], (h[:,l-1,:], c[:,l-1,:]))

            # other layers
            for n in range(1, self.num_layers):
                h[:,0,:], c[:,0,:] = self.forward_layers[n](h[:,0,:])
                for l in range(1, length):
                    h[:,l,:], c[:,l,:] = self.forward_layers[n](h[:,l,:], (h[:,l-1,:], c[:,l-1,:]))

            return h

class GeeArrYou(Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float=0) -> None:
        """
        Sets up the following:
            self.forward_layers - A ModuleList of num_layers GeeArrYouCell layers.
                The first layer should have an input size of input_size
                    and an output size of hidden_size,
                while all other layers should have input and output both of size hidden_size.
            self.dropout - A dropout probability, usable as the "p" value of F.dropout.
        """
        super(GeeArrYou, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.dropout = dropout

        # set up layers
        self.forward_layers = ModuleList()
        for n in range(num_layers):
            if n == 0:
                self.forward_layers.append(GRUCell(input_size, hidden_size))
            else:
                self.forward_layers.append(GRUCell(hidden_size, hidden_size))

    def forward(self, x: TensorType["batch", "length", "input_size"]) -> TensorType["batch", "length", "hidden_size"]:
        """
        Performs the forward propagation of a GeeArrYou layer.
        Inputs:
           x - The inputs to the cell.
        Outputs:
           output - The resulting (hidden state) output h.
               Note that the input to each GeeArrYouCell (except the first) should be
                   passed through F.dropout with the dropout probability provided when
                   initializing the GeeArrYou layer.
        """
        # initializing
        batch, length, input_size = x.shape
        h = zeros(batch, length, self.hidden_size)

        #forward propagation
        for n in range(self.num_layers):
            if n == 0:
                h[:,0,:] = self.forward_layers[0](x[:,0,:])
                for l in range(1, length):
                    h[:,l,:] = self.forward_layers[0](x[:,l,:],h[:,l-1,:])
            else:
                h[:,0,:] = self.forward_layers[n](F.dropout(h[:,0,:], self.dropout))
                for l in range(1, length):
                    h[:,l,:] = self.forward_layers[n](F.dropout(h[:,l,:], self.dropout), F.dropout(h[:,l-1,:], self.dropout))

        return h

class Encoder(Module):
    """Encoder module (Figure 3 (a) in the AutoVC paper).
    """
    def __init__(self, dim_neck: int, dim_emb: int, freq: int):
        """
        Sets up the following:
            self.convolutions - the 1-D convolution layers.
                The first should have 80 + dim_emb input channels and 512 output channels,
                    while each following convolution layer should have 512 input and 512 output channels.
                    All such layers should have a 5x5 kernel, with a stride of 1,
                    a dilation of 1, and a padding of 2.
                The output of each convolution layer should be fed into a BatchNorm1d layer of 512 input features,
                and the output of each BatchNorm1d should be fed into a ReLU layer.
            self.recurrents - a bidirectional EllEssTeeEmm with two layers, an input size of 512,
                and an output size of dim_neck.
        """
        super(Encoder, self).__init__()
        #set up the convolution layer
        convolution_list = []
        convolution_list.append(Conv1d(80+dim_emb, 512, 5, 1, 2, 1)),
        convolution_list.append(BatchNorm1d(512))
        convolution_list.append(ReLU())

        for n in range(2):
            convolution_list.append(Conv1d(512, 512, 5, 1, 2, 1)),
            convolution_list.append(BatchNorm1d(512))
            convolution_list.append(ReLU())
        self.convolutions = Sequential(*convolution_list)
        self.recurrent = EllEssTeeEmm(512, dim_neck, 2, True)

    def forward(self, x: TensorType["batch", "input_dim", "length"]) -> Tuple[
        TensorType["batch", "length", "dim_neck"],
        TensorType["batch", "length", "dim_neck"]
    ]:
        """
        Performs the forward propagation of the AutoVC encoder.
        After passing the input through the convolution layers, the last two dimensions
            should be transposed before passing those layers' output through the EllEssTeeEmm.
            The output from the EllEssTeeEmm should then be split *along the last dimension* into two chunks,
            one for the forward direction (the first self.recurrent_hidden_size columns)
            and one for the backward direction (the last self.recurrent_hidden_size columns).
        """
        temp = self.convolutions(x)
        output = self.recurrent(temp.transpose(1,2))
        recurrent_hidden_size = int(output.shape[2] / 2)
        return output[:,:,:recurrent_hidden_size], output[:,:,recurrent_hidden_size:]
      
class Decoder(Module):
    """Decoder module (Figure 3 (c) in the AutoVC paper, up to the "1x1 Conv").
    
    
    """
    def __init__(self, dim_neck: int, dim_emb: int, dim_pre: int) -> None:
        """
        Sets up the following:
            self.recurrent1 - a unidirectional EllEssTeeEmm with one layer, an input size of 2*dim_neck + dim_emb
                and an output size of dim_pre.
            self.convolutions - the 1-D convolution layers.
                Each convolution layer should have dim_pre input and dim_pre output channels.
                All such layers should have a 5x5 kernel, with a stride of 1,
                a dilation of 1, and a padding of 2.
                The output of each convolution layer should be fed into a BatchNorm1d layer of dim_pre input features,
                and the output of that BatchNorm1d should be fed into a ReLU.
            self.recurrent2 - a unidirectional EllEssTeeEmm with two layers, an input size of dim_pre
                and an output size of 1024.
            self.fc_projection = a LineEar layer with an input size of 1024 and an output size of 80.
        """
        super(Decoder, self).__init__()
        #initializing
        self.recurrent1 = EllEssTeeEmm(2*dim_neck+dim_emb, dim_pre, 1, False)
        self.recurrent2 = EllEssTeeEmm(dim_pre, 1024, 2, False)
        self.fc_projection = LineEar(1024, 80)

        #set up convolution layer
        convolution_list = []
        for n in range(3):
            convolution_list.append(Conv1d(dim_pre, dim_pre, 5, 1, 2, 1)),
            convolution_list.append(BatchNorm1d(dim_pre))
            convolution_list.append(ReLU())
        self.convolutions = Sequential(*convolution_list)

    def forward(self, x: TensorType["batch", "input_length", "input_dim"]) -> TensorType["batch", "input_length", "output_dim"]:
        """
        Performs the forward propagation of the AutoVC decoder.
            It should be enough to pass the input through the first EllEssTeeEmm,
            the convolution layers, the second EllEssTeeEmm, and the final LineEar
            layer in that order--except that the "input_length" and "input_dim" dimensions
            should be transposed before input to the convolution layers, and this transposition
            should be undone before input to the second EllEssTeeEmm.
        """
        temp = self.recurrent1(x)
        output = self.recurrent2(self.convolutions(temp.transpose(1,2)).transpose(1,2))
        output1 = self.fc_projection(output)
        return output1
    
class Postnet(Module):
    """Post-network module (in Figure 3 (c) in the AutoVC paper,
           the two layers "5x1 ConvNorm x 4" and "5x1 ConvNorm".).
    """
    def __init__(self) -> None:
        """
        Sets up the following:
            self.convolutions - a Sequential object with five Conv1d layers, each with 5x5 kernels,
            a stride of 1, a padding of 2, and a dilation of 1:
                The first should take an 80-channel input and yield a 512-channel output.
                The next three should take 512-channel inputs and yield 512-channel outputs.
                The last should take a 512-channel input and yield an 80-channel output.
                Each layer's output should be passed into a BatchNorm1d,
                and (except for the last layer) from there through a Tanh,
                before being sent to the next layer.
        """
        super(Postnet, self).__init__()

        #set up convolution layer
        convolution_list = []
        convolution_list.append(Conv1d(80, 512, 5, 1, 2, 1)),
        convolution_list.append(BatchNorm1d(512))
        convolution_list.append(Tanh())
        for n in range(3):
            convolution_list.append(Conv1d(512, 512, 5, 1, 2, 1)),
            convolution_list.append(BatchNorm1d(512))
            convolution_list.append(Tanh())        
        convolution_list.append(Conv1d(512, 80, 5, 1, 2, 1)),
        convolution_list.append(BatchNorm1d(80))
        self.convolutions = Sequential(*convolution_list)

    def forward(self, x: TensorType["batch", "input_channels", "n_mels"]) -> TensorType["batch", "input_channels", "n_mels"]:
        """
        Performs the forward propagation of the AutoVC decoder.
        If you initialized this module properly, passing the input through self.convolutions here should suffice.
        """
        return self.convolutions(x)

class SpeakerEmbedderGeeArrYou(Module):
    """
    """
    def __init__(self, n_hid: int, n_mels: int, n_layers: int, fc_dim: int, hidden_p: float) -> None:
        """
        Sets up the following:
            self.rnn_stack - an n_layers-layer GeeArrYou with n_mels input features,
                n_hid hidden features, and a dropout of hidden_p.
            self.projection - a LineEar layer with an input size of n_hid
                and an output size of fc_dim.
        """
        super(SpeakerEmbedderGeeArrYou, self).__init__()
        self.rnn_stack = GeeArrYou(n_mels, n_hid, n_layers, hidden_p)
        self.projection = LineEar(n_hid, fc_dim)
        
    def forward(self, x: TensorType["batch", "frames", "n_mels"]) -> TensorType["batch", "fc_dim"]:
        """
        Performs the forward propagation of the SpeakerEmbedderGeeArrYou.
            After passing the input through the RNN, the last frame of the output
            should be taken and passed through the fully connected layer.
            Each of the frames should then be normalized so that its Euclidean norm is 1.
        """
        temp = self.rnn_stack(x)
        last_frame = temp.shape[1]-1
        output = self.projection(temp[:,last_frame,:])
        for i in range(output.shape[0]):
            output[i,:] /= norm(output[i,:])
        return output
