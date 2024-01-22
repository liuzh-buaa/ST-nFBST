import numpy as np
import torch
import torch.nn as nn

from libcity.model.traffic_speed_prediction.layers.functions import calculate_scaled_laplacian, \
    calculate_random_walk_matrix
from libcity.model.traffic_speed_prediction.layers.rand_fc import RandFC
from libcity.model.traffic_speed_prediction.layers.rand_gconv import RandGCONV


class RandDCGRUCell(nn.Module):
    def __init__(self, input_dim, num_units, adj_mx, max_diffusion_step, num_nodes, device, nonlinearity='tanh',
                 filter_type="laplacian", use_gc_for_ru=True, sigma_pi=1.0, sigma_start=1.0,
                 init_func=torch.nn.init.xavier_normal_):
        """

        Args:
            input_dim:
            num_units:
            adj_mx:
            max_diffusion_step:
            num_nodes:
            device:
            nonlinearity:
            filter_type: "laplacian", "random_walk", "dual_random_walk"
            use_gc_for_ru: whether to use Graph convolution to calculate the reset and update gates.
        """

        super().__init__()
        self._activation = torch.tanh if nonlinearity == 'tanh' else torch.relu
        self._num_nodes = num_nodes
        self._num_units = num_units
        self._device = device
        self._max_diffusion_step = max_diffusion_step
        self._supports = []
        self._use_gc_for_ru = use_gc_for_ru

        supports = []
        if filter_type == "laplacian":
            supports.append(calculate_scaled_laplacian(adj_mx, lambda_max=None))
        elif filter_type == "random_walk":
            supports.append(calculate_random_walk_matrix(adj_mx).T)
        elif filter_type == "dual_random_walk":
            supports.append(calculate_random_walk_matrix(adj_mx).T)
            supports.append(calculate_random_walk_matrix(adj_mx.T).T)
        else:
            supports.append(calculate_scaled_laplacian(adj_mx))
        for support in supports:
            self._supports.append(self._build_sparse_matrix(support, self._device))

        if self._use_gc_for_ru:
            self._fn = RandGCONV(self._num_nodes, self._max_diffusion_step, self._supports, self._device,
                                 input_dim=input_dim, hid_dim=self._num_units, output_dim=2 * self._num_units,
                                 bias_start=1.0, sigma_pi=sigma_pi, sigma_start=sigma_start, init_func=init_func)
        else:
            self._fn = RandFC(self._num_nodes, self._device, input_dim=input_dim,
                              hid_dim=self._num_units, output_dim=2 * self._num_units, bias_start=1.0,
                              sigma_pi=sigma_pi, sigma_start=sigma_start, init_func=init_func)
        self._gconv = RandGCONV(self._num_nodes, self._max_diffusion_step, self._supports, self._device,
                                input_dim=input_dim, hid_dim=self._num_units, output_dim=self._num_units,
                                bias_start=0.0, sigma_pi=sigma_pi, sigma_start=sigma_start, init_func=init_func)

    @staticmethod
    def _build_sparse_matrix(lap, device):
        lap = lap.tocoo()
        indices = np.column_stack((lap.row, lap.col))
        # this is to ensure row-major ordering to equal torch.sparse.sparse_reorder(L)
        indices = indices[np.lexsort((indices[:, 0], indices[:, 1]))]
        lap = torch.sparse_coo_tensor(indices.T, lap.data, lap.shape, device=device)
        return lap

    def forward(self, inputs, hx):
        """
        Gated recurrent unit (GRU) with Graph Convolution.

        Args:
            inputs: (B, num_nodes * input_dim)
            hx: (B, num_nodes * rnn_units)

        Returns:
            torch.tensor: shape (B, num_nodes * rnn_units)
        """
        output_size = 2 * self._num_units
        value = torch.sigmoid(self._fn(inputs, hx))  # (batch_size, num_nodes * output_size)
        value = torch.reshape(value, (-1, self._num_nodes, output_size))  # (batch_size, num_nodes, output_size)

        r, u = torch.split(tensor=value, split_size_or_sections=self._num_units, dim=-1)
        r = torch.reshape(r, (-1, self._num_nodes * self._num_units))  # (batch_size, num_nodes * _num_units)
        u = torch.reshape(u, (-1, self._num_nodes * self._num_units))  # (batch_size, num_nodes * _num_units)

        c = self._gconv(inputs, r * hx)  # (batch_size, num_nodes * _num_units)
        if self._activation is not None:
            c = self._activation(c)

        new_state = u * hx + (1.0 - u) * c
        return new_state  # (batch_size, num_nodes * _num_units)

    def get_kl_sum(self):
        return self._fn.get_kl_sum() + self._gconv.get_kl_sum()

    def set_shared_eps(self):
        self._fn.set_shared_eps()
        self._gconv.set_shared_eps()

    def clear_shared_eps(self):
        self._fn.clear_shared_eps()
        self._gconv.clear_shared_eps()
