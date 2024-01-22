import math

import torch
import torch.nn as nn


class RandGCONV(nn.Module):
    def __init__(self, num_nodes, max_diffusion_step, supports, device, input_dim, hid_dim, output_dim, bias_start=0.0,
                 sigma_pi=1.0, sigma_start=1.0, init_func=torch.nn.init.xavier_normal_):
        super(RandGCONV, self).__init__()
        self._num_nodes = num_nodes
        self._max_diffusion_step = max_diffusion_step
        self._supports = supports
        self._device = device
        self._num_matrices = len(self._supports) * self._max_diffusion_step + 1  # Ks
        self._output_dim = output_dim
        self._sigma_pi = sigma_pi
        input_size = input_dim + hid_dim
        shape = (input_size * self._num_matrices, self._output_dim)
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

    @staticmethod
    def _concat(x, x_):
        x_ = x_.unsqueeze(0)
        return torch.cat([x, x_], dim=0)

    def forward(self, inputs, state):
        # 对X(t)和H(t-1)做图卷积，并加偏置bias
        # Reshape input and state to (batch_size, num_nodes, input_dim/state_dim)
        batch_size = inputs.shape[0]
        inputs = torch.reshape(inputs, (batch_size, self._num_nodes, -1))
        state = torch.reshape(state, (batch_size, self._num_nodes, -1))
        inputs_and_state = torch.cat([inputs, state], dim=2)
        # (batch_size, num_nodes, total_arg_size(input_dim+state_dim))
        input_size = inputs_and_state.size(2)  # =total_arg_size

        x = inputs_and_state
        # T0=I x0=T0*x=x
        x0 = x.permute(1, 2, 0)  # (num_nodes, total_arg_size, batch_size)
        x0 = torch.reshape(x0, shape=[self._num_nodes, input_size * batch_size])
        x = torch.unsqueeze(x0, 0)  # (1, num_nodes, total_arg_size * batch_size)

        # 3阶[T0,T1,T2]Chebyshev多项式近似g(theta)
        # 把图卷积公式中的~L替换成了随机游走拉普拉斯D^(-1)*W
        if self._max_diffusion_step == 0:
            pass
        else:
            for support in self._supports:
                # T1=L x1=T1*x=L*x
                x1 = torch.sparse.mm(support, x0)  # supports: n*n; x0: n*(total_arg_size * batch_size)
                x = self._concat(x, x1)  # (2, num_nodes, total_arg_size * batch_size)
                for k in range(2, self._max_diffusion_step + 1):
                    # T2=2LT1-T0=2L^2-1 x2=T2*x=2L^2x-x=2L*x1-x0...
                    # T3=2LT2-T1=2L(2L^2-1)-L x3=2L*x2-x1...
                    x2 = 2 * torch.sparse.mm(support, x1) - x0
                    x = self._concat(x, x2)  # (3, num_nodes, total_arg_size * batch_size)
                    x1, x0 = x2, x1  # 循环
        # x.shape (Ks, num_nodes, total_arg_size * batch_size)
        # Ks = len(supports) * self._max_diffusion_step + 1

        x = torch.reshape(x, shape=[self._num_matrices, self._num_nodes, input_size, batch_size])
        x = x.permute(3, 1, 2, 0)  # (batch_size, num_nodes, input_size, num_matrices)
        x = torch.reshape(x, shape=[batch_size * self._num_nodes, input_size * self._num_matrices])

        sigma_weight = torch.exp(self.log_sigma_weight)
        if self.shared_eps:
            weight = self.mu_weight + sigma_weight * self.buffer_eps_weight
        else:
            weight = self.mu_weight + sigma_weight * torch.randn(self.mu_weight.shape, device=self._device)
        x = torch.matmul(x, weight)  # (batch_size * self._num_nodes, self._output_dim)
        sigma_bias = torch.exp(self.log_sigma_biases)
        if self.shared_eps:
            bias = self.mu_biases + sigma_bias * self.buffer_eps_bias
        else:
            bias = self.mu_biases + sigma_bias * torch.randn(self.mu_biases.shape, device=self._device)
        x = x + bias
        # Reshape res back to 2D: (batch_size * num_nodes, state_dim) -> (batch_size, num_nodes * state_dim)
        return torch.reshape(x, [batch_size, self._num_nodes * self._output_dim])

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
