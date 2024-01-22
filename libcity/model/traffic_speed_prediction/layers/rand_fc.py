import math

import torch
import torch.nn as nn


class RandFC(nn.Module):
    def __init__(self, num_nodes, device, input_dim, hid_dim, output_dim, bias_start=0.0,
                 sigma_pi=1.0, sigma_start=1.0, init_func=torch.nn.init.xavier_normal_):
        super(RandFC, self).__init__()
        self._num_nodes = num_nodes
        self._device = device
        self._output_dim = output_dim
        self._sigma_pi = sigma_pi
        input_size = input_dim + hid_dim
        shape = (input_size, self._output_dim)
        self.mu_weight = torch.nn.Parameter(torch.empty(*shape, device=self._device))
        self.mu_biases = torch.nn.Parameter(torch.empty(self._output_dim, device=self._device))
        self.log_sigma_weight = torch.nn.Parameter(torch.empty(*shape, device=self._device))
        self.log_sigma_biases = torch.nn.Parameter(torch.empty(self._output_dim, device=self._device))
        self.register_buffer('buffer_eps_weight', torch.empty(*shape, device=self._device))
        self.register_buffer('buffer_eps_bias', torch.empty(self._output_dim, device=self._device))
        init_func(self.mu_weight)
        torch.nn.init.constant_(self.mu_biases, bias_start)
        torch.nn.init.constant_(self.log_sigma_weight, math.log(sigma_start))
        torch.nn.init.constant_(self.log_sigma_biases, math.log(sigma_start))
        torch.nn.init.constant_(self.buffer_eps_weight, 0)
        torch.nn.init.constant_(self.buffer_eps_bias, 0)
        self.shared_eps = False

    def forward(self, inputs, state):
        batch_size = inputs.shape[0]
        # Reshape input and state to (batch_size * self._num_nodes, input_dim/state_dim)
        inputs = torch.reshape(inputs, (batch_size * self._num_nodes, -1))
        state = torch.reshape(state, (batch_size * self._num_nodes, -1))
        inputs_and_state = torch.cat([inputs, state], dim=-1)
        # (batch_size * self._num_nodes, input_size(input_dim+state_dim))
        sigma_weight = torch.exp(self.log_sigma_weight)
        if self.shared_eps:
            weight = self.mu_weight + sigma_weight * self.buffer_eps_weight
        else:
            weight = self.mu_weight + sigma_weight * torch.randn(self.mu_weight.shape, device=self._device)
        value = torch.sigmoid(torch.matmul(inputs_and_state, weight))
        # (batch_size * self._num_nodes, self._output_dim)
        sigma_bias = torch.exp(self.log_sigma_biases)
        if self.shared_eps:
            bias = self.mu_biases + sigma_bias * self.buffer_eps_bias
        else:
            bias = self.mu_biases + sigma_bias * torch.randn(self.mu_biases.shape, device=self._device)
        value = value + bias
        # Reshape res back to 2D: (batch_size * num_nodes, state_dim) -> (batch_size, num_nodes * state_dim)
        return torch.reshape(value, [batch_size, self._num_nodes * self._output_dim])

    def get_kl_sum(self):
        kl_weight = - self.log_sigma_weight + 0.5 * (  # +math.log(self._sigma_pi)
                torch.exp(self.log_sigma_weight) ** 2 + self.mu_weight ** 2) / (self._sigma_pi ** 2)
        kl_bias = - self.log_sigma_biases + 0.5 * (  # +math.log(self._sigma_pi)
                torch.exp(self.log_sigma_biases) ** 2 + self.mu_biases ** 2) / (self._sigma_pi ** 2)
        return kl_weight.sum() + kl_bias.sum()

    def set_shared_eps(self):
        self.shared_eps = True
        torch.nn.init.normal_(self.buffer_eps_weight)
        torch.nn.init.normal_(self.buffer_eps_bias)

    def clear_shared_eps(self):
        self.shared_eps = False
