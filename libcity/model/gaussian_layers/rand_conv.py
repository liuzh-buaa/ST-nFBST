import math
from itertools import repeat
from typing import Tuple, Union, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter


class _RandConvNd(nn.Module):
    """
        pi(weight) ~ N(0, sigma_pi^2)
        p(weight) ~ N(mu_weight, exp(log_sigma_weight)^2)
        Sometimes, bias needn't be a gaussian random, then set random_bias=False;
        Sometimes, we needn't calculate kl of bias, then set kl_bias=False;
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Tuple[int, ...],
                 stride: Tuple[int, ...],
                 padding: Tuple[int, ...],
                 dilation: Tuple[int, ...],
                 transposed: bool,
                 output_padding: Tuple[int, ...],
                 groups: int,
                 bias: bool,
                 padding_mode: str,
                 random_bias: bool,
                 kl_bias: bool,
                 sigma_pi: int,
                 sigma_start: int) -> None:
        super(_RandConvNd, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        valid_padding_modes = {'zeros', 'reflect', 'replicate', 'circular'}
        if padding_mode not in valid_padding_modes:
            raise ValueError("padding_mode must be one of {}, but got padding_mode='{}'".format(
                valid_padding_modes, padding_mode))

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        self.padding_mode = padding_mode
        self.sigma_pi = sigma_pi
        self.random_bias = bias and random_bias
        self.kl_bias = self.random_bias and kl_bias

        # `_reversed_padding_repeated_twice` is the padding to be passed to
        # `F.pad` if needed (e.g., for non-zero padding types that are
        # implemented as two ops: padding + conv). `F.pad` accepts paddings in
        # reverse order than the dimension.
        def _reverse_repeat_tuple(t, n):
            r"""Reverse the order of `t` and repeat each element for `n` times.

            This can be used to translate padding arg used by Conv and Pooling modules
            to the ones used by `F.pad`.
            """
            return tuple(x for x in reversed(t) for _ in range(n))

        self._reversed_padding_repeated_twice = _reverse_repeat_tuple(self.padding, 2)

        if transposed:
            self.mu_weight = Parameter(torch.Tensor(in_channels, out_channels // groups, *kernel_size))
            self.log_sigma_weight = Parameter(torch.Tensor(in_channels, out_channels // groups, *kernel_size))
            self.register_buffer('buffer_eps_weight',
                                 torch.Tensor(in_channels, out_channels // groups, *kernel_size))
        else:
            self.mu_weight = Parameter(torch.Tensor(out_channels, in_channels // groups, *kernel_size))
            self.log_sigma_weight = Parameter(torch.Tensor(out_channels, in_channels // groups, *kernel_size))
            self.register_buffer('buffer_eps_weight',
                                 torch.Tensor(out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.mu_bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('mu_bias', None)
        if self.random_bias:
            self.log_sigma_bias = Parameter(torch.Tensor(out_channels))
            self.register_buffer('buffer_eps_bias', torch.Tensor(out_channels))
        else:
            self.register_parameter('log_sigma_bias', None)
            self.register_buffer('buffer_eps_bias', None)

        torch.nn.init.kaiming_uniform_(self.mu_weight, a=math.sqrt(5))
        self.log_sigma_weight.data.fill_(math.log(sigma_start))
        self.buffer_eps_weight.data.zero_()
        if bias:
            def _calculate_fan_in_and_fan_out(tensor):
                dimensions = tensor.dim()
                if dimensions < 2:
                    raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")

                num_input_fmaps = tensor.size(1)
                num_output_fmaps = tensor.size(0)
                receptive_field_size = 1
                if tensor.dim() > 2:
                    receptive_field_size = tensor[0][0].numel()
                fan_in = num_input_fmaps * receptive_field_size
                fan_out = num_output_fmaps * receptive_field_size

                return fan_in, fan_out

            fan_in, _ = _calculate_fan_in_and_fan_out(self.mu_weight)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.mu_bias, -bound, bound)
            if self.random_bias:
                self.log_sigma_bias.data.fill_(math.log(sigma_start))
                self.buffer_eps_bias.data.zero_()

        self.shared_eps = False

    def get_kl_sum(self):
        kl_weight = - self.log_sigma_weight + 0.5 * (  # +math.log(self.sigma_pi)
                torch.exp(self.log_sigma_weight) ** 2 + self.mu_weight ** 2) / (self.sigma_pi ** 2)
        if self.kl_bias:
            kl_bias = - self.log_sigma_bias + 0.5 * (  # +math.log(self.sigma_pi)
                    torch.exp(self.log_sigma_bias) ** 2 + self.mu_bias ** 2) / (self.sigma_pi ** 2)
        else:
            kl_bias = 0

        return kl_weight.sum() + kl_bias.sum()

    def set_shared_eps(self):
        self.shared_eps = True
        torch.nn.init.normal_(self.buffer_eps_weight)
        if self.buffer_eps_bias is not None:
            torch.nn.init.normal_(self.buffer_eps_bias)

    def clear_shared_eps(self):
        self.shared_eps = False

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}, random_bias={random_bias}, kl_bias={kl_bias}, sigma_pi={sigma_pi}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        if self.padding_mode != 'zeros':
            s += ', padding_mode={padding_mode}'
        return s.format(**self.__dict__)


class RandConv1d(_RandConvNd):
    """
        Examples::
        >>> nn.Conv1d(16, 33, 3, stride=2)
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int]],
                 stride: Union[int, Tuple[int]] = 1,
                 padding: Union[int, Tuple[int]] = 0,
                 dilation: Union[int, Tuple[int]] = 1,
                 groups: int = 1,
                 bias: bool = True,
                 padding_mode: str = 'zeros',  # TODO: refine this type
                 random_bias: bool = True,
                 kl_bias: bool = True,
                 sigma_pi: float = 1.0,
                 sigma_start: float = 1.0):
        # we create new variables below to make mypy happy since kernel_size has
        # type Union[int, Tuple[int]] and kernel_size_ has type Tuple[int]
        def _single(x):
            if isinstance(x, int):
                return tuple(repeat(x, 1))
            return x

        kernel_size_ = _single(kernel_size)
        stride_ = _single(stride)
        padding_ = _single(padding)
        dilation_ = _single(dilation)
        super(RandConv1d, self).__init__(
            in_channels, out_channels, kernel_size_, stride_, padding_, dilation_,
            False, _single(0), groups, bias, padding_mode, random_bias, kl_bias, sigma_pi, sigma_start
        )

    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
        if self.padding_mode != 'zeros':
            return F.conv1d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            weight, bias, self.stride,
                            (0,), self.dilation, self.groups)
        return F.conv1d(input, weight, bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, input):
        sigma_weight = torch.exp(self.log_sigma_weight)
        if self.shared_eps:
            weight = self.mu_weight + sigma_weight * self.buffer_eps_weight
        else:
            weight = self.mu_weight + sigma_weight * torch.randn(self.mu_weight.shape, device=self.mu_weight.device)
        bias = None
        if self.mu_bias is not None:
            if self.random_bias:
                sigma_bias = torch.exp(self.log_sigma_bias)
                if self.shared_eps:
                    bias = self.mu_bias + sigma_bias * self.buffer_eps_bias
                else:
                    bias = self.mu_bias + sigma_bias * torch.randn(self.mu_bias.shape, device=self.mu_bias.device)
            else:
                bias = self.mu_bias
        return self._conv_forward(input, weight, bias)


class RandConv2d(_RandConvNd):
    """
        Examples::
        >>> nn.Conv2d(16, 33, 3, stride=2)
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int, int]],
                 stride: Union[int, Tuple[int, int]] = 1,
                 padding: Union[int, Tuple[int, int]] = 0,
                 dilation: Union[int, Tuple[int, int]] = 1,
                 groups: int = 1,
                 bias: bool = True,
                 padding_mode: str = 'zeros',  # TODO: refine this type
                 random_bias: bool = True,
                 kl_bias: bool = True,
                 sigma_pi: float = 1.0,
                 sigma_start: float = 1.0):
        def _pair(x):
            if isinstance(x, int):
                return tuple(repeat(x, 2))
            return x

        kernel_size_ = _pair(kernel_size)
        stride_ = _pair(stride)
        padding_ = _pair(padding)
        dilation_ = _pair(dilation)
        super(RandConv2d, self).__init__(
            in_channels, out_channels, kernel_size_, stride_, padding_, dilation_,
            False, _pair(0), groups, bias, padding_mode, random_bias, kl_bias, sigma_pi, sigma_start
        )

    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            weight, bias, self.stride,
                            (0, 0), self.dilation, self.groups)
        return F.conv2d(input, weight, bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, input: Tensor) -> Tensor:
        sigma_weight = torch.exp(self.log_sigma_weight)
        if self.shared_eps:
            weight = self.mu_weight + sigma_weight * self.buffer_eps_weight
        else:
            weight = self.mu_weight + sigma_weight * torch.randn(self.mu_weight.shape, device=self.mu_weight.device)
        bias = None
        if self.mu_bias is not None:
            if self.random_bias:
                sigma_bias = torch.exp(self.log_sigma_bias)
                if self.shared_eps:
                    bias = self.mu_bias + sigma_bias * self.buffer_eps_bias
                else:
                    bias = self.mu_bias + sigma_bias * torch.randn(self.mu_bias.shape, device=self.mu_bias.device)
            else:
                bias = self.mu_bias
        return self._conv_forward(input, weight, bias)


class RandConv3d(_RandConvNd):
    """
        Examples::
        >>> nn.Conv3d(16, 33, 3, stride=2)
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int, int, int]],
                 stride: Union[int, Tuple[int, int, int]] = 1,
                 padding: Union[int, Tuple[int, int, int]] = 0,
                 dilation: Union[int, Tuple[int, int, int]] = 1,
                 groups: int = 1,
                 bias: bool = True,
                 padding_mode: str = 'zeros',
                 random_bias: bool = True,
                 kl_bias: bool = True,
                 sigma_pi: float = 1.0,
                 sigma_start: float = 1.0):
        def _triple(x):
            if isinstance(x, int):
                return tuple(repeat(x, 3))
            return x

        kernel_size_ = _triple(kernel_size)
        stride_ = _triple(stride)
        padding_ = _triple(padding)
        dilation_ = _triple(dilation)
        super(RandConv3d, self).__init__(
            in_channels, out_channels, kernel_size_, stride_, padding_, dilation_,
            False, _triple(0), groups, bias, padding_mode, random_bias, kl_bias, sigma_pi, sigma_start
        )

    def forward(self, input: Tensor) -> Tensor:
        sigma_weight = torch.exp(self.log_sigma_weight)
        if self.shared_eps:
            weight = self.mu_weight + sigma_weight * self.buffer_eps_weight
        else:
            weight = self.mu_weight + sigma_weight * torch.randn(self.mu_weight.shape, device=self.mu_weight.device)
        bias = None
        if self.mu_bias is not None:
            if self.random_bias:
                sigma_bias = torch.exp(self.log_sigma_bias)
                if self.shared_eps:
                    bias = self.mu_bias + sigma_bias * self.buffer_eps_bias
                else:
                    bias = self.mu_bias + sigma_bias * torch.randn(self.mu_bias.shape, device=self.mu_bias.device)
            else:
                bias = self.mu_bias
        if self.padding_mode != 'zeros':
            return F.conv3d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            weight, bias, self.stride, (0, 0, 0),
                            self.dilation, self.groups)
        return F.conv3d(input, weight, bias, self.stride,
                        self.padding, self.dilation, self.groups)
