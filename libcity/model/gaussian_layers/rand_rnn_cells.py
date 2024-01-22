import math
from typing import Tuple, Optional

import torch
from torch import Tensor, nn
from torch.nn import Parameter, init


class RandRNNCellBase(nn.Module):
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

    def __init__(self, input_size: int, hidden_size: int, bias: bool, num_chunks: int, random_bias: bool, kl_bias: bool,
                 sigma_pi: float, sigma_start: float) -> None:
        super(RandRNNCellBase, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.random_bias = self.bias and random_bias
        self.kl_bias = self.random_bias and kl_bias
        self.sigma_pi = sigma_pi
        self.mu_weight_ih = Parameter(torch.Tensor(num_chunks * hidden_size, input_size))
        self.log_sigma_weight_ih = Parameter(torch.Tensor(num_chunks * hidden_size, input_size))
        self.register_buffer('buffer_eps_weight_ih', torch.Tensor(num_chunks * hidden_size, input_size))
        self.mu_weight_hh = Parameter(torch.Tensor(num_chunks * hidden_size, hidden_size))
        self.log_sigma_weight_hh = Parameter(torch.Tensor(num_chunks * hidden_size, hidden_size))
        self.register_buffer('buffer_eps_weight_hh', torch.Tensor(num_chunks * hidden_size, hidden_size))
        if bias:
            self.mu_bias_ih = Parameter(torch.Tensor(num_chunks * hidden_size))
            self.mu_bias_hh = Parameter(torch.Tensor(num_chunks * hidden_size))
        else:
            self.register_parameter('mu_bias_ih', None)
            self.register_parameter('mu_bias_hh', None)
        if self.random_bias:
            self.log_sigma_bias_ih = Parameter(torch.Tensor(num_chunks * hidden_size))
            self.log_sigma_bias_hh = Parameter(torch.Tensor(num_chunks * hidden_size))
            self.register_buffer('buffer_eps_bias_ih', torch.Tensor(num_chunks * hidden_size))
            self.register_buffer('buffer_eps_bias_hh', torch.Tensor(num_chunks * hidden_size))
        else:
            self.register_parameter('log_sigma_bias_ih', None)
            self.register_parameter('log_sigma_bias_hh', None)
            self.register_buffer('buffer_eps_bias_ih', None)
            self.register_buffer('buffer_eps_bias_hh', None)
        # self.reset_parameters()
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for mu in [self.mu_weight_ih, self.mu_weight_hh, self.mu_bias_ih, self.mu_bias_hh]:
            if mu is not None:
                init.uniform_(mu, -stdv, stdv)
        for log_sigma in [self.log_sigma_weight_ih, self.log_sigma_bias_hh, self.log_sigma_bias_ih,
                          self.log_sigma_bias_hh]:
            if log_sigma is not None:
                init.constant_(log_sigma, math.log(sigma_start))
        for buffer_eps in [self.buffer_eps_weight_ih, self.buffer_eps_weight_hh, self.buffer_eps_bias_ih,
                           self.buffer_eps_bias_hh]:
            if buffer_eps is not None:
                init.constant_(buffer_eps, 0)
        self.shared_eps = False

    def extra_repr(self) -> str:
        s = '{input_size}, {hidden_size}'
        if 'bias' in self.__dict__ and self.bias is not True:
            s += ', bias={bias}'
        if 'nonlinearity' in self.__dict__ and self.nonlinearity != "tanh":
            s += ', nonlinearity={nonlinearity}'
        s += ', random_bias={}, kl_bias={}, sigma_pi={}'.format(self.random_bias, self.kl_bias, self.sigma_pi)
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

    def get_kl_sum(self):
        kl_weight_ih = - self.log_sigma_weight_ih + 0.5 * (  # +math.log(self.sigma_pi)
                torch.exp(self.log_sigma_weight_ih) ** 2 + self.mu_weight_ih ** 2) / (self.sigma_pi ** 2)
        kl_weight_hh = - self.log_sigma_weight_hh + 0.5 * (  # +math.log(self.sigma_pi)
                torch.exp(self.log_sigma_weight_hh) ** 2 + self.mu_weight_hh ** 2) / (self.sigma_pi ** 2)
        if self.kl_bias:
            kl_bias_ih = -self.log_sigma_bias_ih + 0.5 * (  # +math.log(self.sigma_pi)
                    torch.exp(self.log_sigma_bias_ih) ** 2 + self.mu_bias_ih ** 2) / (self.sigma_pi ** 2)
            kl_bias_hh = -self.log_sigma_bias_hh + 0.5 * (  # +math.log(self.sigma_pi)
                    torch.exp(self.log_sigma_bias_hh) ** 2 + self.mu_bias_hh ** 2) / (self.sigma_pi ** 2)
        else:
            kl_bias_ih, kl_bias_hh = 0, 0

        return kl_weight_ih.sum() + kl_weight_hh.sum() + kl_bias_ih.sum() + kl_bias_hh.sum()

    def set_shared_eps(self):
        self.shared_eps = True
        torch.nn.init.normal_(self.buffer_eps_weight_ih)
        torch.nn.init.normal_(self.buffer_eps_weight_hh)
        if self.buffer_eps_bias_ih is not None:
            torch.nn.init.normal_(self.buffer_eps_bias_ih)
        if self.buffer_eps_bias_hh is not None:
            torch.nn.init.normal_(self.buffer_eps_bias_hh)

    def clear_shared_eps(self):
        self.shared_eps = False


class RandRNNCell(RandRNNCellBase):
    """
    replace _VF with specific implementation
    Examples::
    >>> nn.RNNCell
    """
    __constants__ = ['input_size', 'hidden_size', 'bias', 'nonlinearity']
    nonlinearity: str

    def __init__(self, input_size: int, hidden_size: int, bias: bool = True, nonlinearity: str = "tanh",
                 random_bias: bool = True, kl_bias: bool = True, sigma_pi: float = 1.0,
                 sigma_start: float = 1.0) -> None:
        super(RandRNNCell, self).__init__(input_size, hidden_size, bias, num_chunks=1, random_bias=random_bias,
                                          kl_bias=kl_bias, sigma_pi=sigma_pi, sigma_start=sigma_start)
        self.nonlinearity = nonlinearity

    def forward(self, input: Tensor, hx: Optional[Tensor] = None) -> Tensor:
        self.check_forward_input(input)
        if hx is None:
            hx = torch.zeros(input.size(0), self.hidden_size, dtype=input.dtype, device=input.device)
        self.check_forward_hidden(input, hx, '')

        sigma_weight_ih = torch.exp(self.log_sigma_weight_ih)
        sigma_weight_hh = torch.exp(self.log_sigma_weight_hh)
        if self.shared_eps:
            weight_ih = self.mu_weight_ih + sigma_weight_ih * self.buffer_eps_weight_ih
            weight_hh = self.mu_weight_hh + sigma_weight_hh * self.buffer_eps_weight_hh
        else:
            weight_ih = self.mu_weight_ih + sigma_weight_ih * torch.randn(self.mu_weight_ih.shape,
                                                                          device=self.mu_weight_ih.device)
            weight_hh = self.mu_weight_hh + sigma_weight_hh * torch.randn(self.mu_weight_hh.shape,
                                                                          device=self.mu_weight_hh.device)
        if self.mu_bias_ih is not None:
            if self.random_bias:
                sigma_bias_ih = torch.exp(self.log_sigma_bias_ih)
                sigma_bias_hh = torch.exp(self.log_sigma_bias_hh)
                if self.shared_eps:
                    bias_ih = self.mu_bias_ih + sigma_bias_ih * self.buffer_eps_bias_ih
                    bias_hh = self.mu_bias_hh + sigma_bias_hh * self.buffer_eps_bias_hh
                else:
                    bias_ih = self.mu_bias_ih + sigma_bias_ih * torch.randn(self.mu_bias_ih.shape,
                                                                            device=self.mu_bias_ih.device)
                    bias_hh = self.mu_bias_hh + sigma_bias_hh * torch.randn(self.mu_bias_hh.shape,
                                                                            device=self.mu_bias_hh.device)
            else:
                bias_ih = self.mu_bias_ih
                bias_hh = self.mu_bias_hh
        else:
            bias_ih, bias_hh = None, None

        if self.nonlinearity == "tanh":
            # ret = _VF.rnn_tanh_cell(
            #     input, hx,
            #     self.weight_ih, self.weight_hh,
            #     self.bias_ih, self.bias_hh,
            # )
            igates = torch.mm(input, weight_ih.t()) + bias_ih
            hgates = torch.mm(hx, weight_hh.t()) + bias_hh
            ret = torch.tanh(igates + hgates)
        elif self.nonlinearity == "relu":
            # ret = _VF.rnn_relu_cell(
            #     input, hx,
            #     self.weight_ih, self.weight_hh,
            #     self.bias_ih, self.bias_hh,
            # )
            igates = torch.mm(input, weight_ih.t()) + bias_ih
            hgates = torch.mm(hx, weight_hh.t()) + bias_hh
            ret = torch.relu(igates + hgates)
        else:
            ret = input  # TODO: remove when jit supports exception flow
            raise RuntimeError(
                "Unknown nonlinearity: {}".format(self.nonlinearity))
        return ret


class RandLSTMCell(RandRNNCellBase):
    """
    replace _VF with specific implementation
    Examples::
    >>> nn.LSTMCell
    """

    def __init__(self, input_size: int, hidden_size: int, bias: bool = True,
                 random_bias: bool = True, kl_bias: bool = True, sigma_pi: float = 1.0,
                 sigma_start: float = 1.0) -> None:
        super(RandLSTMCell, self).__init__(input_size, hidden_size, bias, num_chunks=4, random_bias=random_bias,
                                           kl_bias=kl_bias, sigma_pi=sigma_pi, sigma_start=sigma_start)

    def forward(self, input: Tensor, hx: Optional[Tuple[Tensor, Tensor]] = None) -> Tuple[Tensor, Tensor]:
        self.check_forward_input(input)  # (batch_size, input_size)
        if hx is None:
            zeros = torch.zeros(input.size(0), self.hidden_size, dtype=input.dtype, device=input.device)
            hx = (zeros, zeros)
        self.check_forward_hidden(input, hx[0], '[0]')  # (batch_size, hidden_size)
        self.check_forward_hidden(input, hx[1], '[1]')  # (batch_size, hidden_size)

        sigma_weight_ih = torch.exp(self.log_sigma_weight_ih)
        sigma_weight_hh = torch.exp(self.log_sigma_weight_hh)
        if self.shared_eps:
            weight_ih = self.mu_weight_ih + sigma_weight_ih * self.buffer_eps_weight_ih
            weight_hh = self.mu_weight_hh + sigma_weight_hh * self.buffer_eps_weight_hh
        else:
            weight_ih = self.mu_weight_ih + sigma_weight_ih * torch.randn(self.mu_weight_ih.shape,
                                                                          device=self.mu_weight_ih.device)
            weight_hh = self.mu_weight_hh + sigma_weight_hh * torch.randn(self.mu_weight_hh.shape,
                                                                          device=self.mu_weight_hh.device)
        if self.mu_bias_ih is not None:
            if self.random_bias:
                sigma_bias_ih = torch.exp(self.log_sigma_bias_ih)
                sigma_bias_hh = torch.exp(self.log_sigma_bias_hh)
                if self.shared_eps:
                    bias_ih = self.mu_bias_ih + sigma_bias_ih * self.buffer_eps_bias_ih
                    bias_hh = self.mu_bias_hh + sigma_bias_hh * self.buffer_eps_bias_hh
                else:
                    bias_ih = self.mu_bias_ih + sigma_bias_ih * torch.randn(self.mu_bias_ih.shape,
                                                                            device=self.mu_bias_ih.device)
                    bias_hh = self.mu_bias_hh + sigma_bias_hh * torch.randn(self.mu_bias_hh.shape,
                                                                            device=self.mu_bias_hh.device)
            else:
                bias_ih = self.mu_bias_ih
                bias_hh = self.mu_bias_hh
        else:
            bias_ih, bias_hh = None, None

        # return _VF.lstm_cell(
        #     input, hx,
        #     self.weight_ih, self.weight_hh,
        #     self.bias_ih, self.bias_hh,
        # )
        hx, cx = hx
        gates = torch.mm(input, weight_ih.t()) + torch.mm(hx, weight_hh.t()) + bias_ih + bias_hh

        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * torch.tanh(cy)

        return hy, (hy, cy)


class RandGRUCell(RandRNNCellBase):
    """
    replace _VF with specific implementation
    Examples::
    >>> nn.GRUCell
    """

    def __init__(self, input_size: int, hidden_size: int, bias: bool = True,
                 random_bias: bool = True, kl_bias: bool = True, sigma_pi: float = 1.0,
                 sigma_start: float = 1.0) -> None:
        super(RandGRUCell, self).__init__(input_size, hidden_size, bias, num_chunks=3, random_bias=random_bias,
                                          kl_bias=kl_bias, sigma_pi=sigma_pi, sigma_start=sigma_start)

    def forward(self, input: Tensor, hx: Optional[Tensor] = None) -> Tensor:
        self.check_forward_input(input)
        if hx is None:
            hx = torch.zeros(input.size(0), self.hidden_size, dtype=input.dtype, device=input.device)
        self.check_forward_hidden(input, hx, '')

        sigma_weight_ih = torch.exp(self.log_sigma_weight_ih)
        sigma_weight_hh = torch.exp(self.log_sigma_weight_hh)
        if self.shared_eps:
            weight_ih = self.mu_weight_ih + sigma_weight_ih * self.buffer_eps_weight_ih
            weight_hh = self.mu_weight_hh + sigma_weight_hh * self.buffer_eps_weight_hh
        else:
            weight_ih = self.mu_weight_ih + sigma_weight_ih * torch.randn(self.mu_weight_ih.shape,
                                                                          device=self.mu_weight_ih.device)
            weight_hh = self.mu_weight_hh + sigma_weight_hh * torch.randn(self.mu_weight_hh.shape,
                                                                          device=self.mu_weight_hh.device)
        if self.mu_bias_ih is not None:
            if self.random_bias:
                sigma_bias_ih = torch.exp(self.log_sigma_bias_ih)
                sigma_bias_hh = torch.exp(self.log_sigma_bias_hh)
                if self.shared_eps:
                    bias_ih = self.mu_bias_ih + sigma_bias_ih * self.buffer_eps_bias_ih
                    bias_hh = self.mu_bias_hh + sigma_bias_hh * self.buffer_eps_bias_hh
                else:
                    bias_ih = self.mu_bias_ih + sigma_bias_ih * torch.randn(self.mu_bias_ih.shape,
                                                                            device=self.mu_bias_ih.device)
                    bias_hh = self.mu_bias_hh + sigma_bias_hh * torch.randn(self.mu_bias_hh.shape,
                                                                            device=self.mu_bias_hh.device)
            else:
                bias_ih = self.mu_bias_ih
                bias_hh = self.mu_bias_hh
        else:
            bias_ih, bias_hh = None, None

        # return _VF.gru_cell(
        #     input, hx,
        #     self.weight_ih, self.weight_hh,
        #     self.bias_ih, self.bias_hh,
        # )
        gi = torch.mm(input, weight_ih.t()) + bias_ih
        gh = torch.mm(hx, weight_hh.t()) + bias_hh
        i_r, i_i, i_n = gi.chunk(3, 1)
        h_r, h_i, h_n = gh.chunk(3, 1)

        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + resetgate * h_n)
        hy = newgate + inputgate * (hx - newgate)

        return hy
