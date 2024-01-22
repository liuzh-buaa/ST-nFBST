import numbers
import warnings
from typing import Tuple, Optional, Union

import torch
from torch import nn, Tensor
from torch.nn.utils.rnn import PackedSequence

from libcity.model.gaussian_layers.rnn_cells import LSTMCell, GRUCell, RNNCell


def apply_permutation(tensor: Tensor, permutation: Tensor, dim: int = 1) -> Tensor:
    return tensor.index_select(dim, permutation)


class RNNLayer(nn.Module):
    def __init__(self, cell, *cell_args):
        super(RNNLayer, self).__init__()
        self.cell = cell(*cell_args)

    def forward(self, input, state, **kwargs):
        if 'batch_sizes' in kwargs:
            batch_sizes = kwargs['batch_sizes']
            outputs, ret_state = [], None
            i = 0
            for batch_size in batch_sizes:
                real_input = input.data[i: i + batch_size]
                # real_input = torch.cat([real_input, torch.zeros((batch_sizes[0] - batch_size, real_input.shape[1]))])
                # out, state = self.cell(real_input, ret_state)
                out, state = self.cell(real_input, (state[0][:batch_size], state[1][:batch_size]))
                outputs += [out.clone()]
                if ret_state is None:
                    ret_state = state
                else:
                    ret_state = torch.cat([state[0], ret_state[0][batch_size:, ]]), \
                                torch.cat([state[1], ret_state[1][batch_size:, ]])
                    # ret_state[0][:batch_size], ret_state[1][:batch_size] = state  # inplace
                    # ret_state[0][:batch_size], ret_state[1][:batch_size] = state[0][:batch_size], state[1][:batch_size]
                i += batch_size
            outputs = torch.cat(outputs, dim=0)
        else:
            batch_first = kwargs['batch_first']
            inputs = input.unbind(1) if batch_first else input.unbind(0)
            outputs = []
            for i in range(len(inputs)):  # according to seq_len
                out, state = self.cell(inputs[i], state)
                outputs += [out]
            outputs = torch.stack(outputs, dim=1) if batch_first else torch.stack(outputs, dim=0)
            ret_state = state
        return outputs, ret_state

    def reset_parameters(self):
        self.cell.reset_parameters()


class RNNBase(nn.Module):
    """
    Note: this is only to test whether the implementation is correct, the performance is slower than official version
        which is implemented with C++ _VF.
    Warning: This version doesn't support setting bidirectional or proj_size or dropout... #TODO
    Examples::
        nn.RNNBase
    """

    def __init__(self, mode: str, input_size: int, hidden_size: int,
                 num_layers: int = 1, bias: bool = True, batch_first: bool = False,
                 dropout: float = 0., bidirectional: bool = False, proj_size: int = 0) -> None:
        super(RNNBase, self).__init__()
        self.mode = mode
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = float(dropout)
        self.bidirectional = bidirectional
        self.proj_size = proj_size
        num_directions = 2 if bidirectional else 1

        assert dropout == 0, "this version doesn't support setting dropout."
        assert not bidirectional, "this version doesn't support setting bidirectional."
        assert proj_size == 0, "this version doesn't support setting proj_size."

        if not isinstance(dropout, numbers.Number) or not 0 <= dropout <= 1 or \
                isinstance(dropout, bool):
            raise ValueError("dropout should be a number in range [0, 1] "
                             "representing the probability of an element being "
                             "zeroed")
        if dropout > 0 and num_layers == 1:
            warnings.warn("dropout option adds dropout after all but last "
                          "recurrent layer, so non-zero dropout expects "
                          "num_layers greater than 1, but got dropout={} and "
                          "num_layers={}".format(dropout, num_layers))
        if proj_size < 0:
            raise ValueError("proj_size should be a positive integer or zero to disable projections")
        if proj_size >= hidden_size:
            raise ValueError("proj_size has to be smaller than hidden_size")

        # if mode == 'LSTM':
        #     gate_size = 4 * hidden_size
        # elif mode == 'GRU':
        #     gate_size = 3 * hidden_size
        # elif mode == 'RNN_TANH':
        #     gate_size = hidden_size
        # elif mode == 'RNN_RELU':
        #     gate_size = hidden_size
        # else:
        #     raise ValueError("Unrecognized RNN mode: " + mode)
        #
        # self._all_weights = []
        #
        # for layer in range(num_layers):
        #     for direction in range(num_directions):
        #         real_hidden_size = proj_size if proj_size > 0 else hidden_size
        #         layer_hidden_size = input_size if layer == 0 else real_hidden_size * num_directions
        #
        #         w_ih = Parameter(torch.Tensor(gate_size, layer_hidden_size))
        #         w_hh = Parameter(torch.Tensor(gate_size, real_hidden_size))
        #         b_ih = Parameter(torch.Tensor(gate_size))
        #         # Second bias vector included for CuDNN compatibility. Only one
        #         # bias vector is needed in standard definition
        #         b_hh = Parameter(torch.Tensor(gate_size))
        #         layer_params: Tuple[Tensor, ...] = ()
        #         if self.proj_size == 0:
        #             if bias:
        #                 layer_params = (w_ih, w_hh, b_ih, b_hh)
        #             else:
        #                 layer_params = (w_ih, w_hh)
        #         else:
        #             w_hr = Parameter(torch.Tensor(proj_size, hidden_size))
        #             if bias:
        #                 layer_params = (w_ih, w_hh, b_ih, b_hh, w_hr)
        #             else:
        #                 layer_params = (w_ih, w_hh, w_hr)
        #
        #         suffix = '_reverse' if direction == 1 else ''
        #         param_names = ['weight_ih_l{}{}', 'weight_hh_l{}{}']
        #         if bias:
        #             param_names += ['bias_ih_l{}{}', 'bias_hh_l{}{}']
        #         if self.proj_size > 0:
        #             param_names += ['weight_hr_l{}{}']
        #         param_names = [x.format(layer, suffix) for x in param_names]
        #
        #         for name, param in zip(param_names, layer_params):
        #             setattr(self, name, param)
        #         self._all_weights.append(param_names)

        def init_stacked_layers(num_layers, layer, first_layer_args, other_layer_args):
            layers = [layer(*first_layer_args)] + [layer(*other_layer_args)
                                                   for _ in range(num_layers - 1)]
            return nn.ModuleList(layers)

        if mode == 'LSTM':
            self.layers = init_stacked_layers(self.num_layers, RNNLayer,
                                              [LSTMCell, input_size, hidden_size, bias],
                                              [LSTMCell, hidden_size, hidden_size, bias])
        elif mode == 'GRU':
            self.layers = init_stacked_layers(self.num_layers, RNNLayer,
                                              [GRUCell, input_size, hidden_size, bias],
                                              [GRUCell, hidden_size, hidden_size, bias])
        elif mode == 'RNN_TANH':
            self.layers = init_stacked_layers(self.num_layers, RNNLayer,
                                              [RNNCell, input_size, hidden_size, bias, 'tanh'],
                                              [RNNCell, hidden_size, hidden_size, bias, 'tanh'])
        elif mode == 'RNN_RELU':
            self.layers = init_stacked_layers(self.num_layers, RNNLayer,
                                              [RNNCell, input_size, hidden_size, bias, 'relu'],
                                              [RNNCell, hidden_size, hidden_size, bias, 'relu'])
        else:
            raise ValueError("Unrecognized RNN mode: " + mode)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # stdv = 1.0 / math.sqrt(self.hidden_size)
        # for weight in self.parameters():
        #     init.uniform_(weight, -stdv, stdv)

        for layer in self.layers:
            layer.reset_parameters()

    def check_input(self, input: Tensor, batch_sizes: Optional[Tensor]) -> None:
        expected_input_dim = 2 if batch_sizes is not None else 3
        if input.dim() != expected_input_dim:
            raise RuntimeError(
                'input must have {} dimensions, got {}'.format(
                    expected_input_dim, input.dim()))
        if self.input_size != input.size(-1):
            raise RuntimeError(
                'input.size(-1) must be equal to input_size. Expected {}, got {}'.format(
                    self.input_size, input.size(-1)))

    def get_expected_hidden_size(self, input: Tensor, batch_sizes: Optional[Tensor]) -> Tuple[int, int, int]:
        if batch_sizes is not None:
            mini_batch = int(batch_sizes[0])
        else:
            mini_batch = input.size(0) if self.batch_first else input.size(1)
        num_directions = 2 if self.bidirectional else 1
        if self.proj_size > 0:
            expected_hidden_size = (self.num_layers * num_directions,
                                    mini_batch, self.proj_size)
        else:
            expected_hidden_size = (self.num_layers * num_directions,
                                    mini_batch, self.hidden_size)
        return expected_hidden_size

    def check_hidden_size(self, hx: Tensor, expected_hidden_size: Tuple[int, int, int],
                          msg: str = 'Expected hidden size {}, got {}') -> None:
        if hx.size() != expected_hidden_size:
            raise RuntimeError(msg.format(expected_hidden_size, list(hx.size())))

    def check_forward_args(self, input: Tensor, hidden: Tensor, batch_sizes: Optional[Tensor]):
        self.check_input(input, batch_sizes)
        expected_hidden_size = self.get_expected_hidden_size(input, batch_sizes)

        self.check_hidden_size(hidden, expected_hidden_size)

    def permute_hidden(self, hx: Tensor, permutation: Optional[Tensor]):
        if permutation is None:
            return hx
        return apply_permutation(hx, permutation)

    def extra_repr(self) -> str:
        s = '{input_size}, {hidden_size}'
        if self.proj_size != 0:
            s += ', proj_size={proj_size}'
        if self.num_layers != 1:
            s += ', num_layers={num_layers}'
        if self.bias is not True:
            s += ', bias={bias}'
        if self.batch_first is not False:
            s += ', batch_first={batch_first}'
        if self.dropout != 0:
            s += ', dropout={dropout}'
        if self.bidirectional is not False:
            s += ', bidirectional={bidirectional}'
        return s.format(**self.__dict__)


class RNN(RNNBase):
    """
    Examples::
        nn.RNN
    """

    def __init__(self, *args, **kwargs):
        if 'proj_size' in kwargs:
            raise ValueError("proj_size argument is only supported for LSTM, not RNN or GRU")
        self.nonlinearity = kwargs.pop('nonlinearity', 'tanh')
        if self.nonlinearity == 'tanh':
            mode = 'RNN_TANH'
        elif self.nonlinearity == 'relu':
            mode = 'RNN_RELU'
        else:
            raise ValueError("Unknown nonlinearity '{}'".format(self.nonlinearity))
        super(RNN, self).__init__(mode, *args, **kwargs)


class LSTM(RNNBase):
    """
    Examples::
        nn.LSTM
    """

    def __init__(self, *args, **kwargs):
        super(LSTM, self).__init__('LSTM', *args, **kwargs)

    def get_expected_cell_size(self, input: Tensor, batch_sizes: Optional[Tensor]) -> Tuple[int, int, int]:
        if batch_sizes is not None:
            mini_batch = int(batch_sizes[0])
        else:
            mini_batch = input.size(0) if self.batch_first else input.size(1)
        num_directions = 2 if self.bidirectional else 1
        expected_hidden_size = (self.num_layers * num_directions,
                                mini_batch, self.hidden_size)
        return expected_hidden_size

    # In the future, we should prevent mypy from applying contravariance rules here.
    # See torch/nn/modules/module.py::_forward_unimplemented
    def check_forward_args(self, input: Tensor, hidden: Tuple[Tensor, Tensor],
                           batch_sizes: Optional[Tensor]):  # type: ignore
        self.check_input(input, batch_sizes)
        self.check_hidden_size(hidden[0], self.get_expected_hidden_size(input, batch_sizes),
                               'Expected hidden[0] size {}, got {}')
        self.check_hidden_size(hidden[1], self.get_expected_cell_size(input, batch_sizes),
                               'Expected hidden[1] size {}, got {}')

    # Same as above, see torch/nn/modules/module.py::_forward_unimplemented
    def permute_hidden(self, hx: Tuple[Tensor, Tensor], permutation: Optional[Tensor]) -> Tuple[
        Tensor, Tensor]:  # type: ignore
        if permutation is None:
            return hx
        return apply_permutation(hx[0], permutation), apply_permutation(hx[1], permutation)

    def forward(self, input: Union[Tensor, PackedSequence],
                hx: Optional[Tensor] = None) -> Tuple[Union[Tensor, PackedSequence], Tensor]:
        orig_input = input
        # xxx: isinstance check needs to be in conditional for TorchScript to compile
        if isinstance(orig_input, PackedSequence):
            input, batch_sizes, sorted_indices, unsorted_indices = input
            max_batch_size = batch_sizes[0]
            max_batch_size = int(max_batch_size)
        else:
            batch_sizes = None
            max_batch_size = input.size(0) if self.batch_first else input.size(1)
            sorted_indices = None
            unsorted_indices = None

        if hx is None:
            num_directions = 2 if self.bidirectional else 1
            real_hidden_size = self.proj_size if self.proj_size > 0 else self.hidden_size
            h_zeros = torch.zeros(self.num_layers * num_directions,
                                  max_batch_size, real_hidden_size,
                                  dtype=input.dtype, device=input.device)
            c_zeros = torch.zeros(self.num_layers * num_directions,
                                  max_batch_size, self.hidden_size,
                                  dtype=input.dtype, device=input.device)
            hx = (h_zeros, c_zeros)
        else:
            # Each batch of the hidden state should match the input sequence that
            # the user believes he/she is passing in.
            hx = self.permute_hidden(hx, sorted_indices)

        self.check_forward_args(input, hx, batch_sizes)
        # if batch_sizes is None:
        #     result = _VF.lstm(input, hx, self._flat_weights, self.bias, self.num_layers,
        #                       self.dropout, self.training, self.bidirectional, self.batch_first)
        # else:
        #     result = _VF.lstm(input, batch_sizes, hx, self._flat_weights, self.bias,
        #                       self.num_layers, self.dropout, self.training, self.bidirectional)
        if batch_sizes is None:
            result = self.lstm(input, hx, self.batch_first)
        else:
            result = self.lstm_packed(input, batch_sizes, hx)
        output = result[0]
        hidden = result[1:]
        # xxx: isinstance check needs to be in conditional for TorchScript to compile
        if isinstance(orig_input, PackedSequence):
            output_packed = PackedSequence(output, batch_sizes, sorted_indices, unsorted_indices)
            return output_packed, self.permute_hidden(hidden, unsorted_indices)
        else:
            return output, self.permute_hidden(hidden, unsorted_indices)

    def lstm(self, input, states, batch_first):
        # if batch_first: (batch, seq_len, input_size); if not batch_first: (seq_len, batch, input_size)
        output_states1, output_states2 = [], []
        output = input
        # XXX: enumerate https://github.com/pytorch/pytorch/issues/14471
        i = 0
        for rnn_layer in self.layers:
            # if batch_first: (batch, seq_len, hidden_size), (batch, hidden_size);
            output, out_state = rnn_layer(output, (states[0][i], states[0][i]), batch_first=batch_first)
            output_states1 += [out_state[0]]
            output_states2 += [out_state[1]]
            i += 1
        return output, torch.stack(output_states1, dim=0), torch.stack(output_states2, dim=0)

    def lstm_packed(self, input, batch_sizes, states):
        output_states1, output_states2 = [], []
        output = input
        # XXX: enumerate https://github.com/pytorch/pytorch/issues/14471
        i = 0
        for rnn_layer in self.layers:
            # if batch_first: (batch, seq_len, hidden_size), (batch, hidden_size);
            output, out_state = rnn_layer(output, (states[0][i], states[0][i]), batch_sizes=batch_sizes)
            output_states1 += [out_state[0]]
            output_states2 += [out_state[1]]
            i += 1
        return output, torch.stack(output_states1, dim=0), torch.stack(output_states2, dim=0)