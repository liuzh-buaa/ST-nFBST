import math
from typing import Tuple, Optional

import torch
from torch import Tensor, nn
from torch.nn import Parameter, init


class RNNCellBase(nn.Module):
    """
    the same as torch\nn\modules\rnn.py
    Examples::
    >>> nn.RNNCellBase
    """
    __constants__ = ['input_size', 'hidden_size', 'bias']

    input_size: int
    hidden_size: int
    bias: bool
    weight_ih: Tensor
    weight_hh: Tensor

    # WARNING: bias_ih and bias_hh purposely not defined here.
    # See https://github.com/pytorch/pytorch/issues/39670

    def __init__(self, input_size: int, hidden_size: int, bias: bool, num_chunks: int) -> None:
        super(RNNCellBase, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.weight_ih = Parameter(torch.Tensor(num_chunks * hidden_size, input_size))
        self.weight_hh = Parameter(torch.Tensor(num_chunks * hidden_size, hidden_size))
        if bias:
            self.bias_ih = Parameter(torch.Tensor(num_chunks * hidden_size))
            self.bias_hh = Parameter(torch.Tensor(num_chunks * hidden_size))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)
        self.reset_parameters()

    def extra_repr(self) -> str:
        s = '{input_size}, {hidden_size}'
        if 'bias' in self.__dict__ and self.bias is not True:
            s += ', bias={bias}'
        if 'nonlinearity' in self.__dict__ and self.nonlinearity != "tanh":
            s += ', nonlinearity={nonlinearity}'
        return s.format(**self.__dict__)

    def check_forward_input(self, input: Tensor) -> None:
        if input.size(1) != self.input_size:
            raise RuntimeError(
                "input has inconsistent input_size: got {}, expected {}".format(
                    input.size(1), self.input_size))

    def check_forward_hidden(self, input: Tensor, hx: Tensor, hidden_label: str = '') -> None:
        if input.size(0) != hx.size(0):
            raise RuntimeError(
                "Input batch size {} doesn't match hidden{} batch size {}".format(
                    input.size(0), hidden_label, hx.size(0)))

        if hx.size(1) != self.hidden_size:
            raise RuntimeError(
                "hidden{} has inconsistent hidden_size: got {}, expected {}".format(
                    hidden_label, hx.size(1), self.hidden_size))

    def reset_parameters(self) -> None:
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            init.uniform_(weight, -stdv, stdv)


class RNNCell(RNNCellBase):
    """
    replace _VF with specific implementation
    Examples::
    >>> nn.RNNCell
    """
    __constants__ = ['input_size', 'hidden_size', 'bias', 'nonlinearity']
    nonlinearity: str

    def __init__(self, input_size: int, hidden_size: int, bias: bool = True, nonlinearity: str = "tanh") -> None:
        super(RNNCell, self).__init__(input_size, hidden_size, bias, num_chunks=1)
        self.nonlinearity = nonlinearity

    def forward(self, input: Tensor, hx: Optional[Tensor] = None) -> Tensor:
        self.check_forward_input(input)
        if hx is None:
            hx = torch.zeros(input.size(0), self.hidden_size, dtype=input.dtype, device=input.device)
        self.check_forward_hidden(input, hx, '')
        if self.nonlinearity == "tanh":
            # ret = _VF.rnn_tanh_cell(
            #     input, hx,
            #     self.weight_ih, self.weight_hh,
            #     self.bias_ih, self.bias_hh,
            # )
            igates = torch.mm(input, self.weight_ih.t()) + self.bias_ih
            hgates = torch.mm(hx, self.weight_hh.t()) + self.bias_hh
            ret = torch.tanh(igates + hgates)
        elif self.nonlinearity == "relu":
            # ret = _VF.rnn_relu_cell(
            #     input, hx,
            #     self.weight_ih, self.weight_hh,
            #     self.bias_ih, self.bias_hh,
            # )
            igates = torch.mm(input, self.weight_ih.t()) + self.bias_ih
            hgates = torch.mm(hx, self.weight_hh.t()) + self.bias_hh
            ret = torch.relu(igates + hgates)
        else:
            ret = input  # TODO: remove when jit supports exception flow
            raise RuntimeError(
                "Unknown nonlinearity: {}".format(self.nonlinearity))
        return ret


class LSTMCell(RNNCellBase):
    """
    replace _VF with specific implementation
    Examples::
    >>> nn.LSTMCell
    """

    def __init__(self, input_size: int, hidden_size: int, bias: bool = True) -> None:
        super(LSTMCell, self).__init__(input_size, hidden_size, bias, num_chunks=4)

    def forward(self, input: Tensor, hx: Optional[Tuple[Tensor, Tensor]] = None) -> Tuple[Tensor, Tensor]:
        self.check_forward_input(input)  # (batch_size, input_size)
        if hx is None:
            zeros = torch.zeros(input.size(0), self.hidden_size, dtype=input.dtype, device=input.device)
            hx = (zeros, zeros)
        self.check_forward_hidden(input, hx[0], '[0]')  # (batch_size, hidden_size)
        self.check_forward_hidden(input, hx[1], '[1]')  # (batch_size, hidden_size)
        # return _VF.lstm_cell(
        #     input, hx,
        #     self.weight_ih, self.weight_hh,
        #     self.bias_ih, self.bias_hh,
        # )
        hx, cx = hx
        gates = torch.mm(input, self.weight_ih.t()) + torch.mm(hx, self.weight_hh.t()) + self.bias_ih + self.bias_hh

        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * torch.tanh(cy)

        return hy, (hy, cy)


class GRUCell(RNNCellBase):
    """
    replace _VF with specific implementation
    Examples::
    >>> nn.GRUCell
    """

    def __init__(self, input_size: int, hidden_size: int, bias: bool = True) -> None:
        super(GRUCell, self).__init__(input_size, hidden_size, bias, num_chunks=3)

    def forward(self, input: Tensor, hx: Optional[Tensor] = None) -> Tensor:
        self.check_forward_input(input)
        if hx is None:
            hx = torch.zeros(input.size(0), self.hidden_size, dtype=input.dtype, device=input.device)
        self.check_forward_hidden(input, hx, '')
        # return _VF.gru_cell(
        #     input, hx,
        #     self.weight_ih, self.weight_hh,
        #     self.bias_ih, self.bias_hh,
        # )
        gi = torch.mm(input, self.weight_ih.t()) + self.bias_ih
        gh = torch.mm(hx, self.weight_hh.t()) + self.bias_hh
        i_r, i_i, i_n = gi.chunk(3, 1)
        h_r, h_i, h_n = gh.chunk(3, 1)

        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + resetgate * h_n)
        hy = newgate + inputgate * (hx - newgate)

        return hy