from logging import getLogger

import numpy as np
import torch
import torch.nn as nn

import libcity.interpreter_methods
from libcity.model import loss
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel
from libcity.model.covert_dcrnn_to_b import convert_dcrnn_to_bdcrnn
from libcity.model.traffic_speed_prediction.layers.rand_dcgru_cell import RandDCGRUCell
from libcity.model.traffic_speed_prediction.layers.rand_linear import RandLinear


class Seq2SeqAttrs:
    def __init__(self, config, adj_mx):
        self.adj_mx = adj_mx
        self.max_diffusion_step = int(config.get('max_diffusion_step', 2))
        self.cl_decay_steps = int(config.get('cl_decay_steps', 1000))
        self.filter_type = config.get('filter_type', 'laplacian')
        self.num_nodes = int(config.get('num_nodes', 1))
        self.num_rnn_layers = int(config.get('num_rnn_layers', 2))
        self.rnn_units = int(config.get('rnn_units', 64))
        self.hidden_state_size = self.num_nodes * self.rnn_units
        self.input_dim = config.get('feature_dim', 1)
        self.device = config.get('device', torch.device('cpu'))
        self.sigma_pi = float(config.get('sigma_pi'))
        self.sigma_start = float(config.get('sigma_start'))
        self.sigma_sigma_pi = float(config.get('sigma_sigma_pi', 0))
        self.sigma_sigma_start = float(config.get('sigma_sigma_start', 0))
        self.reg_encoder = config.get('reg_encoder', False)
        self.reg_decoder = config.get('reg_decoder', False)
        self.reg_encoder_sigma_0 = config.get('reg_encoder_sigma_0', False)
        self.reg_decoder_sigma_0 = config.get('reg_decoder_sigma_0', False)
        self.loss_function = config.get('loss_function')
        self.clamp_function = config.get('clamp_function', 'null')
        self.switch_consistent = config.get('switch_consistent', False)


class EncoderModel(nn.Module, Seq2SeqAttrs):
    def __init__(self, config, adj_mx):
        nn.Module.__init__(self)
        Seq2SeqAttrs.__init__(self, config, adj_mx)
        self.dcgru_layers = nn.ModuleList()
        self.dcgru_layers.append(RandDCGRUCell(self.input_dim, self.rnn_units, adj_mx, self.max_diffusion_step,
                                               self.num_nodes, self.device, filter_type=self.filter_type,
                                               sigma_pi=self.sigma_pi, sigma_start=self.sigma_start))
        for i in range(1, self.num_rnn_layers):
            self.dcgru_layers.append(RandDCGRUCell(self.rnn_units, self.rnn_units, adj_mx, self.max_diffusion_step,
                                                   self.num_nodes, self.device, filter_type=self.filter_type,
                                                   sigma_pi=self.sigma_pi, sigma_start=self.sigma_start))

    def forward(self, inputs, hidden_state=None):
        """
        Encoder forward pass.

        Args:
            inputs: shape (batch_size, self.num_nodes * self.input_dim)
            hidden_state: (num_layers, batch_size, self.hidden_state_size),
                optional, zeros if not provided, hidden_state_size = num_nodes * rnn_units

        Returns:
            tuple: tuple contains:
                output: shape (batch_size, self.hidden_state_size) \n
                hidden_state: shape (num_layers, batch_size, self.hidden_state_size) \n
                (lower indices mean lower layers)

        """
        batch_size, _ = inputs.size()
        if hidden_state is None:
            hidden_state = torch.zeros((self.num_rnn_layers, batch_size, self.hidden_state_size), device=self.device)
        hidden_states = []
        output = inputs
        for layer_num, dcgru_layer in enumerate(self.dcgru_layers):
            next_hidden_state = dcgru_layer(output, hidden_state[layer_num])
            # next_hidden_state: (batch_size, self.num_nodes * self.rnn_units)
            hidden_states.append(next_hidden_state)
            output = next_hidden_state  # 循环
        return output, torch.stack(hidden_states)  # runs in O(num_layers) so not too slow

    def get_kl_sum(self):
        kl_sum = 0
        for dcgru_layer in self.dcgru_layers:
            kl_sum += dcgru_layer.get_kl_sum()
        return kl_sum

    def set_shared_eps(self):
        for dcgru_layer in self.dcgru_layers:
            dcgru_layer.set_shared_eps()

    def clear_shared_eps(self):
        for dcgru_layer in self.dcgru_layers:
            dcgru_layer.clear_shared_eps()


class DecoderModel(nn.Module, Seq2SeqAttrs):
    def __init__(self, config, adj_mx, variance=False):
        nn.Module.__init__(self)
        Seq2SeqAttrs.__init__(self, config, adj_mx)
        self.output_dim = config.get('output_dim', 1)
        self.projection_layer = RandLinear(self.rnn_units, self.output_dim, self.device,
                                           sigma_pi=self.sigma_pi, sigma_start=self.sigma_start)
        if variance:
            self.variance_layer = RandLinear(self.rnn_units, self.output_dim, self.device, sigma_pi=self.sigma_sigma_pi,
                                             sigma_start=self.sigma_sigma_start)
        else:
            self.variance_layer = None
        self.dcgru_layers = nn.ModuleList()
        self.dcgru_layers.append(RandDCGRUCell(self.output_dim, self.rnn_units, adj_mx, self.max_diffusion_step,
                                               self.num_nodes, self.device, filter_type=self.filter_type,
                                               sigma_pi=self.sigma_pi, sigma_start=self.sigma_start))
        for i in range(1, self.num_rnn_layers):
            self.dcgru_layers.append(RandDCGRUCell(self.rnn_units, self.rnn_units, adj_mx, self.max_diffusion_step,
                                                   self.num_nodes, self.device, filter_type=self.filter_type,
                                                   sigma_pi=self.sigma_pi, sigma_start=self.sigma_start))

    def forward(self, inputs, hidden_state=None):
        """
        Decoder forward pass.

        Args:
            inputs:  shape (batch_size, self.num_nodes * self.output_dim)
            hidden_state: (num_layers, batch_size, self.hidden_state_size),
                optional, zeros if not provided, hidden_state_size = num_nodes * rnn_units

        Returns:
            tuple: tuple contains:
                output: shape (batch_size, self.num_nodes * self.output_dim) \n
                hidden_state: shape (num_layers, batch_size, self.hidden_state_size) \n
                (lower indices mean lower layers)
        """
        hidden_states = []
        output = inputs
        for layer_num, dcgru_layer in enumerate(self.dcgru_layers):
            next_hidden_state = dcgru_layer(output, hidden_state[layer_num])
            # next_hidden_state: (batch_size, self.num_nodes * self.rnn_units)
            hidden_states.append(next_hidden_state)
            output = next_hidden_state
        projected = self.projection_layer(output.view(-1, self.rnn_units))
        projected = projected.view(-1, self.num_nodes * self.output_dim)
        if self.variance_layer is None:
            return projected, torch.stack(hidden_states)
        else:
            variance = self.variance_layer(output.view(-1, self.rnn_units))
            variance = variance.view(-1, self.num_nodes * self.output_dim)
            return projected, torch.stack(hidden_states), variance

    def get_kl_sum(self):
        kl_sum = self.projection_layer.get_kl_sum()
        for dcgru_layer in self.dcgru_layers:
            kl_sum += dcgru_layer.get_kl_sum()
        if self.variance_layer is not None:
            kl_sum += self.variance_layer.get_kl_sum()
        return kl_sum

    def set_shared_eps(self):
        self.projection_layer.set_shared_eps()
        for dcgru_layer in self.dcgru_layers:
            dcgru_layer.set_shared_eps()
        if self.variance_layer is not None:
            self.variance_layer.set_shared_eps()

    def clear_shared_eps(self):
        self.projection_layer.clear_shared_eps()
        for dcgru_layer in self.dcgru_layers:
            dcgru_layer.clear_shared_eps()
        if self.variance_layer is not None:
            self.variance_layer.clear_shared_eps()


class EncoderSigmaModel(nn.Module, Seq2SeqAttrs):
    def __init__(self, config, adj_mx):
        nn.Module.__init__(self)
        Seq2SeqAttrs.__init__(self, config, adj_mx)
        self.dcgru_layers = nn.ModuleList()

        self.dcgru_layers.append(RandDCGRUCell(self.input_dim, self.rnn_units, adj_mx, self.max_diffusion_step,
                                               self.num_nodes, self.device, filter_type=self.filter_type,
                                               sigma_pi=self.sigma_sigma_pi, sigma_start=self.sigma_sigma_start))
        for i in range(1, self.num_rnn_layers):
            self.dcgru_layers.append(RandDCGRUCell(self.rnn_units, self.rnn_units, adj_mx, self.max_diffusion_step,
                                                   self.num_nodes, self.device, filter_type=self.filter_type,
                                                   sigma_pi=self.sigma_sigma_pi, sigma_start=self.sigma_sigma_start))

    def forward(self, inputs, hidden_state=None):
        """
        Encoder forward pass.

        Args:
            inputs: shape (batch_size, self.num_nodes * self.input_dim)
            hidden_state: (num_layers, batch_size, self.hidden_state_size),
                optional, zeros if not provided, hidden_state_size = num_nodes * rnn_units

        Returns:
            tuple: tuple contains:
                output: shape (batch_size, self.hidden_state_size) \n
                hidden_state: shape (num_layers, batch_size, self.hidden_state_size) \n
                (lower indices mean lower layers)

        """
        batch_size, _ = inputs.size()
        if hidden_state is None:
            hidden_state = torch.zeros((self.num_rnn_layers, batch_size, self.hidden_state_size), device=self.device)
        hidden_states = []
        output = inputs
        for layer_num, dcgru_layer in enumerate(self.dcgru_layers):
            next_hidden_state = dcgru_layer(output, hidden_state[layer_num])
            # next_hidden_state: (batch_size, self.num_nodes * self.rnn_units)
            hidden_states.append(next_hidden_state)
            output = next_hidden_state  # 循环
        return output, torch.stack(hidden_states)  # runs in O(num_layers) so not too slow

    def get_kl_sum(self):
        kl_sum = 0
        for dcgru_layer in self.dcgru_layers:
            kl_sum += dcgru_layer.get_kl_sum()
        return kl_sum

    def set_shared_eps(self):
        for dcgru_layer in self.dcgru_layers:
            dcgru_layer.set_shared_eps()

    def clear_shared_eps(self):
        for dcgru_layer in self.dcgru_layers:
            dcgru_layer.clear_shared_eps()


class DecoderSigmaModel(nn.Module, Seq2SeqAttrs):
    def __init__(self, config, adj_mx):
        nn.Module.__init__(self)
        Seq2SeqAttrs.__init__(self, config, adj_mx)
        self.output_dim = config.get('output_dim', 1)
        self.projection_layer = RandLinear(self.rnn_units, self.output_dim, self.device,
                                           sigma_pi=self.sigma_sigma_pi, sigma_start=self.sigma_sigma_start)
        self.dcgru_layers = nn.ModuleList()

        self.dcgru_layers.append(RandDCGRUCell(self.output_dim, self.rnn_units, adj_mx, self.max_diffusion_step,
                                               self.num_nodes, self.device, filter_type=self.filter_type,
                                               sigma_pi=self.sigma_sigma_pi, sigma_start=self.sigma_sigma_start))
        for i in range(1, self.num_rnn_layers):
            self.dcgru_layers.append(RandDCGRUCell(self.rnn_units, self.rnn_units, adj_mx, self.max_diffusion_step,
                                                   self.num_nodes, self.device, filter_type=self.filter_type,
                                                   sigma_pi=self.sigma_sigma_pi, sigma_start=self.sigma_sigma_start))

    def forward(self, inputs, hidden_state=None):
        """
        Decoder forward pass.

        Args:
            inputs:  shape (batch_size, self.num_nodes * self.output_dim)
            hidden_state: (num_layers, batch_size, self.hidden_state_size),
                optional, zeros if not provided, hidden_state_size = num_nodes * rnn_units

        Returns:
            tuple: tuple contains:
                output: shape (batch_size, self.num_nodes * self.output_dim) \n
                hidden_state: shape (num_layers, batch_size, self.hidden_state_size) \n
                (lower indices mean lower layers)
        """
        hidden_states = []
        output = inputs
        for layer_num, dcgru_layer in enumerate(self.dcgru_layers):
            next_hidden_state = dcgru_layer(output, hidden_state[layer_num])
            # next_hidden_state: (batch_size, self.num_nodes * self.rnn_units)
            hidden_states.append(next_hidden_state)
            output = next_hidden_state
        projected = self.projection_layer(output.view(-1, self.rnn_units))
        output = projected.view(-1, self.num_nodes * self.output_dim)
        return output, torch.stack(hidden_states)

    def get_kl_sum(self):
        kl_sum = self.projection_layer.get_kl_sum()
        for dcgru_layer in self.dcgru_layers:
            kl_sum += dcgru_layer.get_kl_sum()
        return kl_sum

    def set_shared_eps(self):
        self.projection_layer.set_shared_eps()
        for dcgru_layer in self.dcgru_layers:
            dcgru_layer.set_shared_eps()

    def clear_shared_eps(self):
        self.projection_layer.clear_shared_eps()
        for dcgru_layer in self.dcgru_layers:
            dcgru_layer.clear_shared_eps()


class BDCRNNBase(AbstractTrafficStateModel, Seq2SeqAttrs):
    def __init__(self, config, data_feature, variance=False):
        self.adj_mx = data_feature.get('adj_mx')
        self.num_nodes = data_feature.get('num_nodes', 1)
        self.feature_dim = data_feature.get('feature_dim', 1)
        config['num_nodes'] = self.num_nodes
        config['feature_dim'] = self.feature_dim
        self.output_dim = data_feature.get('output_dim', 1)

        super().__init__(config, data_feature)
        Seq2SeqAttrs.__init__(self, config, self.adj_mx)
        self.encoder_model = EncoderModel(config, self.adj_mx)
        self.decoder_model = DecoderModel(config, self.adj_mx, variance)

        self.use_curriculum_learning = config.get('use_curriculum_learning', False)
        self.input_window = config.get('input_window', 1)
        self.output_window = config.get('output_window', 1)
        self.device = config.get('device', torch.device('cpu'))
        self._logger = getLogger()
        self._scaler = self.data_feature.get('scaler')

        if config['init_params_from_dcrnn']:
            convert_dcrnn_to_bdcrnn(self, self.device)

    def _compute_sampling_threshold(self, batches_seen):
        return self.cl_decay_steps / (
                self.cl_decay_steps + np.exp(batches_seen / self.cl_decay_steps))

    def encoder(self, inputs):
        raise NotImplementedError('BDCRNN encoder not implemented.')

    def decoder(self, encoder_hidden_state, labels=None, batches_seen=None):
        raise NotImplementedError('BDCRNN decoder not implemented.')

    def forward(self, batch, batches_seen=None):
        raise NotImplementedError('BDCRNN forward not implemented.')

    def predict(self, batch, batches_seen=None):
        raise NotImplementedError('BDCRNN predict not implemented.')

    def calculate_loss(self, batch, batches_seen=None, num_batches=1):
        raise NotImplementedError('BDCRNN calculate_loss not implemented.')

    def calculate_eval_loss(self, batch, batches_seen=None):
        y_true = batch['y']
        y_predicted = self.predict(batch, batches_seen)
        y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])
        return loss.masked_mae_torch(y_predicted, y_true, 0)

    def _get_kl_sum(self):
        kl_sum = 0
        if self.reg_encoder:
            kl_sum += self.encoder_model.get_kl_sum()
        if self.reg_decoder:
            kl_sum += self.decoder_model.get_kl_sum()
        if self.reg_encoder_sigma_0:
            kl_sum += self.encoder_sigma_model.get_kl_sum()
        if self.reg_decoder_sigma_0:
            kl_sum += self.decoder_sigma_model.get_kl_sum()
        return kl_sum

    def get_interpret(self, x, output_window, num_nodes, output_dim):
        interpreter = libcity.interpreter_methods.interpreter(self)
        x_clone = x.detach().clone().requires_grad_()
        statistic = interpreter.interpret(x_clone, output_window, num_nodes, output_dim)
        interpreter.release()
        return statistic

    def get_interpret_sigma(self, x, output_window, num_nodes, output_dim):
        interpreter = libcity.interpreter_methods.interpreter(self)
        x_clone = x.detach().clone().requires_grad_()
        statistic = interpreter.interpret_sigma(x_clone, output_window, num_nodes, output_dim)
        interpreter.release()
        return statistic
