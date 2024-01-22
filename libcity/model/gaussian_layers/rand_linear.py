import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RandLinear(nn.Module):
    """
        pi(weight) ~ N(0, sigma_pi^2)
        p(weight) ~ N(mu_weight, exp(log_sigma_weight)^2)
        Sometimes, bias needn't be a gaussian random, then set random_bias=False;
        Sometimes, we needn't calculate kl of bias, then set kl_bias=False;
        Examples::
        >>> nn.Linear(20, 30)
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 random_bias: bool = True, kl_bias: bool = True, sigma_pi: float = 1.0, sigma_start: float = 1.0):
        super(RandLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_pi = sigma_pi
        self.random_bias = bias and random_bias
        self.kl_bias = self.random_bias and kl_bias
        self.mu_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.log_sigma_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.register_buffer('buffer_eps_weight', torch.Tensor(out_features, in_features))
        if bias:
            self.mu_bias = torch.nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('mu_bias', None)
        if self.random_bias:
            self.log_sigma_bias = torch.nn.Parameter(torch.Tensor(out_features))
            self.register_buffer('buffer_eps_bias', torch.Tensor(out_features))
        else:
            self.register_parameter('log_sigma_bias', None)
            self.register_buffer('buffer_eps_bias', None)

        torch.nn.init.kaiming_uniform_(self.mu_weight, a=math.sqrt(5))
        torch.nn.init.constant_(self.log_sigma_weight, math.log(sigma_start))
        torch.nn.init.constant_(self.buffer_eps_weight, 0)
        if self.mu_bias is not None:
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
                torch.nn.init.constant_(self.log_sigma_bias, math.log(sigma_start))
                torch.nn.init.constant_(self.buffer_eps_bias, 0)

        self.shared_eps = False

    def forward(self, input):
        sigma_weight = torch.exp(self.log_sigma_weight)
        if self.shared_eps:
            weight = self.mu_weight + sigma_weight * self.buffer_eps_weight
        else:
            weight = self.mu_weight + sigma_weight * torch.randn(self.mu_weight.shape, device=self.mu_weight.device)
        if self.mu_bias is not None:
            if self.random_bias:
                sigma_bias = torch.exp(self.log_sigma_bias)
                if self.shared_eps:
                    bias = self.mu_bias + sigma_bias * self.buffer_eps_bias
                else:
                    bias = self.mu_bias + sigma_bias * torch.randn(self.mu_bias.shape, device=self.mu_bias.device)
            else:
                bias = self.mu_bias
        else:
            bias = None
        return F.linear(input, weight, bias)

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

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}, random_bias={}, kl_bias={}, sigma_pi={}'.format(
            self.in_features, self.out_features, self.bias is not None, self.random_bias, self.kl_bias, self.sigma_pi
        )
