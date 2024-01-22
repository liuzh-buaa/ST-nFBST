from logging import getLogger

import torch.nn.functional as F
import numpy as np
import torch
from torch import nn

from libcity.model import loss
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel
from libcity.model.traffic_speed_prediction.DCRNN import Seq2SeqAttrs, EncoderModel, DecoderModel, count_parameters


class DropoutModel(nn.Module, Seq2SeqAttrs):
    def __init__(self, input_size, layer_num: int, dropout_rates):
        super().__init__()
        if isinstance(dropout_rates, int):
            dropout_rates = [dropout_rates] * layer_num
        else:
            assert isinstance(dropout_rates, list)
            if len(dropout_rates) == 1:
                dropout_rates = dropout_rates * layer_num
            else:
                assert len(dropout_rates) == layer_num

        layers = [nn.Linear(input_size, 64), nn.ReLU()]
        for dropout_rate in dropout_rates:
            layers += [nn.Dropout(dropout_rate)]

        self.model = nn.Sequential(*layers)

    def forward(self, input):
        return self.model(input)


class DCRNNDropout(AbstractTrafficStateModel, Seq2SeqAttrs):
    def __init__(self, config, data_feature):
        self.adj_mx = data_feature.get('adj_mx')
        self.num_nodes = data_feature.get('num_nodes', 1)
        self.feature_dim = data_feature.get('feature_dim', 1)
        config['num_nodes'] = self.num_nodes
        config['feature_dim'] = self.feature_dim
        self.output_dim = data_feature.get('output_dim', 1)

        super().__init__(config, data_feature)
        Seq2SeqAttrs.__init__(self, config, self.adj_mx)
        self.encoder_model = EncoderModel(config, self.adj_mx)
        self.decoder_model1 = DecoderModel(config, self.adj_mx)
        self.decoder_model2 = DecoderModel(config, self.adj_mx)

        self.use_curriculum_learning = config.get('use_curriculum_learning', False)
        self.input_window = config.get('input_window', 1)
        self.output_window = config.get('output_window', 1)
        self.device = config.get('device', torch.device('cpu'))
        self._logger = getLogger()
        self._scaler = self.data_feature.get('scaler')

        self.mu_dropout = config.get('mu_dropout')
        self.sigma_dropout = config.get('sigma_dropout')
        self.loss_function = config.get('loss_function')
        self.clamp_function = config.get('clamp_function', 'null')
        self.reg_encoder = config.get('reg_encoder', False)
        self.reg_decoder = config.get('reg_decoder', False)

    def _compute_sampling_threshold(self, batches_seen):
        return self.cl_decay_steps / (
                self.cl_decay_steps + np.exp(batches_seen / self.cl_decay_steps))

    def encoder(self, inputs):
        """
        encoder forward pass on t time steps

        Args:
            inputs: shape (input_window, batch_size, num_sensor * input_dim)

        Returns:
            torch.tensor: (num_layers, batch_size, self.hidden_state_size)
        """
        encoder_hidden_state = None
        for t in range(self.input_window):
            _, encoder_hidden_state = self.encoder_model(inputs[t], encoder_hidden_state)
            # encoder_hidden_state: encoder的多层GRU的全部的隐层 (num_layers, batch_size, self.hidden_state_size)

        return encoder_hidden_state  # 最后一个隐状态

    def decoder1(self, encoder_hidden_state, labels=None, batches_seen=None):
        """
        Decoder forward pass

        Args:
            encoder_hidden_state: (num_layers, batch_size, self.hidden_state_size)
            labels:  (self.output_window, batch_size, self.num_nodes * self.output_dim)
                [optional, not exist for inference]
            batches_seen: global step [optional, not exist for inference]

        Returns:
            torch.tensor: (self.output_window, batch_size, self.num_nodes * self.output_dim)
        """
        batch_size = encoder_hidden_state.size(1)
        go_symbol = torch.zeros((batch_size, self.num_nodes * self.output_dim), device=self.device)
        decoder_hidden_state = encoder_hidden_state
        decoder_input = go_symbol

        outputs = []
        for t in range(self.output_window):
            decoder_output, decoder_hidden_state = self.decoder_model1(decoder_input, decoder_hidden_state)
            decoder_input = decoder_output  # (batch_size, self.num_nodes * self.output_dim)
            outputs.append(decoder_output)
            if self.training and self.use_curriculum_learning:
                c = np.random.uniform(0, 1)
                if c < self._compute_sampling_threshold(batches_seen):
                    decoder_input = labels[t]  # (batch_size, self.num_nodes * self.output_dim)
        outputs = torch.stack(outputs)
        return outputs

    def decoder2(self, encoder_hidden_state, labels=None, batches_seen=None):
        """
        Decoder forward pass

        Args:
            encoder_hidden_state: (num_layers, batch_size, self.hidden_state_size)
            labels:  (self.output_window, batch_size, self.num_nodes * self.output_dim)
                [optional, not exist for inference]
            batches_seen: global step [optional, not exist for inference]

        Returns:
            torch.tensor: (self.output_window, batch_size, self.num_nodes * self.output_dim)
        """
        batch_size = encoder_hidden_state.size(1)
        go_symbol = torch.zeros((batch_size, self.num_nodes * self.output_dim), device=self.device)
        decoder_hidden_state = encoder_hidden_state
        decoder_input = go_symbol

        outputs = []
        for t in range(self.output_window):
            decoder_output, decoder_hidden_state = self.decoder_model2(decoder_input, decoder_hidden_state)
            decoder_input = decoder_output  # (batch_size, self.num_nodes * self.output_dim)
            outputs.append(decoder_output)
            if self.training and self.use_curriculum_learning:
                c = np.random.uniform(0, 1)
                if c < self._compute_sampling_threshold(batches_seen):
                    decoder_input = labels[t]  # (batch_size, self.num_nodes * self.output_dim)
        outputs = torch.stack(outputs)
        return outputs

    def forward(self, batch, batches_seen=None, switch_outputs=True, switch_sigma_0=True):
        """
        seq2seq forward pass

        Args:
            batch: a batch of input,
                batch['X']: shape (batch_size, input_window, num_nodes, input_dim) \n
                batch['y']: shape (batch_size, output_window, num_nodes, output_dim) \n
            batches_seen: batches seen till now
            switch_outputs: whether to predict outputs
            switch_sigma_0: whether to predict sigma_0

        Returns:
            torch.tensor: (batch_size, self.output_window, self.num_nodes, self.output_dim)
        """
        inputs = batch['X']
        labels = batch['y']
        batch_size, _, num_nodes, input_dim = inputs.shape
        inputs = inputs.permute(1, 0, 2, 3)  # (input_window, batch_size, num_nodes, input_dim)
        inputs = inputs.view(self.input_window, batch_size, num_nodes * input_dim).to(self.device)
        self._logger.debug("X: {}".format(inputs.size()))  # (input_window, batch_size, num_nodes * input_dim)

        if labels is not None:
            labels = labels.permute(1, 0, 2, 3)  # (output_window, batch_size, num_nodes, output_dim)
            labels = labels[..., :self.output_dim].contiguous().view(
                self.output_window, batch_size, num_nodes * self.output_dim).to(self.device)
            self._logger.debug("y: {}".format(labels.size()))

        encoder_hidden_state = self.encoder(inputs)
        # (num_layers, batch_size, self.hidden_state_size)
        self._logger.debug("Encoder complete")
        outputs = sigma_0 = None
        if switch_outputs:
            # encoder_hidden_state1 = nn.Dropout(self.mu_dropout)(encoder_hidden_state)
            encoder_hidden_state1 = F.dropout(encoder_hidden_state, self.mu_dropout, True)
            outputs = self.decoder1(encoder_hidden_state1, labels, batches_seen=batches_seen)
            # (self.output_window, batch_size, self.num_nodes * self.output_dim)
            outputs = outputs.view(self.output_window, batch_size, self.num_nodes, self.output_dim).permute(1, 0, 2, 3)
            self._logger.debug("Decoder outputs complete")
        if switch_sigma_0:
            # encoder_hidden_state2 = nn.Dropout(self.sigma_dropout)(encoder_hidden_state)
            encoder_hidden_state2 = F.dropout(encoder_hidden_state, self.sigma_dropout, True)
            sigma_0 = self.decoder2(encoder_hidden_state2, labels, batches_seen=batches_seen)
            # (self.output_window, batch_size, self.num_nodes * self.output_dim)
            sigma_0 = sigma_0.view(self.output_window, batch_size, self.num_nodes, self.output_dim).permute(1, 0, 2, 3)
            self._logger.debug("Decoder sigma complete")

        if batches_seen == 0:
            self._logger.info("Total trainable parameters {}".format(count_parameters(self)))

        return outputs, sigma_0

    def calculate_loss(self, batch, batches_seen=None, num_batches=1):
        y_true = batch['y']
        y_predicted, sigma_0 = self.forward(batch, batches_seen)
        y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])
        ll = self.clamp_function.split('_')
        if self.loss_function == 'masked_mae' and ll[0] == 'relu':
            return loss.masked_mae_relu_reg_torch(y_predicted, y_true, sigma_0, self._get_kl_sum() / num_batches, 0,
                                                  float(ll[1]))
        elif self.loss_function == 'masked_mae' and ll[0] == 'Softplus':
            return loss.masked_mae_softplus_reg_torch(y_predicted, y_true, sigma_0, self._get_kl_sum() / num_batches, 0,
                                                      int(ll[1]))
        elif self.loss_function == 'masked_mse' and ll[0] == 'relu':
            return loss.masked_mse_relu_reg_torch(y_predicted, y_true, sigma_0, self._get_kl_sum() / num_batches, 0,
                                                  float(ll[1]))
        elif self.loss_function == 'masked_mse' and ll[0] == 'Softplus':
            return loss.masked_mse_softplus_reg_torch(y_predicted, y_true, sigma_0, self._get_kl_sum() / num_batches, 0,
                                                      int(ll[1]))
        else:
            raise NotImplementedError('Unrecognized loss function.')

    def calculate_eval_loss(self, batch, batches_seen=None):
        y_true = batch['y']
        y_predicted = self.predict(batch, batches_seen)
        y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])
        return loss.masked_mae_torch(y_predicted, y_true, 0)

    def predict(self, batch, batches_seen=None):
        return self.forward(batch, batches_seen, switch_outputs=True, switch_sigma_0=False)[0]

    def predict_sigma(self, batch, batches_seen=None):
        sigma_0 = self.forward(batch, batches_seen, switch_outputs=False, switch_sigma_0=True)[1]
        ll = self.clamp_function.split('_')
        if ll[0] == 'relu':
            return torch.clamp(sigma_0, min=float(ll[1]))
        elif ll[0] == 'Softplus':
            return torch.nn.Softplus(beta=int(ll[1]))(sigma_0)
        else:
            raise NotImplementedError('Unrecognized loss function.')

    def _get_kl_sum(self):
        kl_sum = 0
        if self.reg_encoder:
            kl_sum += self.encoder_model.get_kl_sum()
        if self.reg_decoder:
            kl_sum += self.decoder_model.get_kl_sum()
        return kl_sum
