import numpy as np
import torch

from libcity.model import loss
from libcity.model.traffic_speed_prediction.BDCRNNBase import BDCRNNBase, DecoderSigmaModel, EncoderSigmaModel
from libcity.model.traffic_speed_prediction.layers.functions import count_parameters


class BDCRNNVariable(BDCRNNBase):
    def __init__(self, config, data_feature):
        super(BDCRNNVariable, self).__init__(config, data_feature)

        self.encoder_sigma_model = EncoderSigmaModel(config, self.adj_mx)
        self.decoder_sigma_model = DecoderSigmaModel(config, self.adj_mx)

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

    def encoder_sigma(self, inputs):
        encoder_hidden_state = None
        for t in range(self.input_window):
            _, encoder_hidden_state = self.encoder_sigma_model(inputs[t], encoder_hidden_state)
            # encoder_hidden_state: encoder的多层GRU的全部的隐层 (num_layers, batch_size, self.hidden_state_size)

        return encoder_hidden_state  # 最后一个隐状态

    def decoder(self, encoder_hidden_state, labels=None, batches_seen=None):
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
            decoder_output, decoder_hidden_state = self.decoder_model(decoder_input, decoder_hidden_state)
            decoder_input = decoder_output  # (batch_size, self.num_nodes * self.output_dim)
            outputs.append(decoder_output)
            if self.training and self.use_curriculum_learning:
                c = np.random.uniform(0, 1)
                if c < self._compute_sampling_threshold(batches_seen):
                    decoder_input = labels[t]  # (batch_size, self.num_nodes * self.output_dim)
        outputs = torch.stack(outputs)
        return outputs

    def decoder_sigma(self, encoder_hidden_state, labels=None, batches_seen=None):
        batch_size = encoder_hidden_state.size(1)
        go_symbol = torch.zeros((batch_size, self.num_nodes * self.output_dim), device=self.device)
        decoder_hidden_state = encoder_hidden_state
        decoder_input = go_symbol

        outputs = []
        for t in range(self.output_window):
            decoder_output, decoder_hidden_state = self.decoder_sigma_model(decoder_input, decoder_hidden_state)
            decoder_input = decoder_output  # (batch_size, self.num_nodes * self.output_dim)
            outputs.append(decoder_output)
            # if self.training and self.use_curriculum_learning:
            #     c = np.random.uniform(0, 1)
            #     if c < self._compute_sampling_threshold(batches_seen):
            #         decoder_input = labels[t]  # (batch_size, self.num_nodes * self.output_dim)
        outputs = torch.stack(outputs)
        return outputs

    def forward(self, batch, batches_seen=None):
        """
        seq2seq forward pass

        Args:
            batch: a batch of input,
                batch['X']: shape (batch_size, input_window, num_nodes, input_dim) \n
                batch['y']: shape (batch_size, output_window, num_nodes, output_dim) \n
            batches_seen: batches seen till now

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
        outputs = self.decoder(encoder_hidden_state, labels, batches_seen=batches_seen)
        # (self.output_window, batch_size, self.num_nodes * self.output_dim)
        self._logger.debug("Decoder complete")

        if batches_seen == 0:
            self._logger.info("Total trainable parameters {}".format(count_parameters(self)))
        outputs = outputs.view(self.output_window, batch_size, self.num_nodes, self.output_dim).permute(1, 0, 2, 3)
        return outputs

    def forward_sigma(self, batch, batches_seen=None):
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

        encoder_hidden_state = self.encoder_sigma(inputs)
        # (num_layers, batch_size, self.hidden_state_size)
        self._logger.debug("Encoder sigma complete")
        outputs = self.decoder_sigma(encoder_hidden_state, labels, batches_seen=batches_seen)
        # (self.output_window, batch_size, self.num_nodes * self.output_dim)
        self._logger.debug("Decoder sigma complete")

        if batches_seen == 0:
            self._logger.info("Total trainable parameters {}".format(count_parameters(self)))
        outputs = outputs.view(self.output_window, batch_size, self.num_nodes, self.output_dim).permute(1, 0, 2, 3)
        return outputs

    def calculate_loss(self, batch, batches_seen=None, num_batches=1):
        y_true = batch['y']
        y_predicted = self.predict(batch, batches_seen)
        y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])
        sigma_0 = self.forward_sigma(batch, batches_seen)
        ll = self.clamp_function.split('_')
        if self.loss_function == 'masked_mae' and ll[0] == 'relu':
            return loss.masked_mae_relu_reg_torch(y_predicted, y_true, sigma_0, self._get_kl_sum() / num_batches, 0, float(ll[1]), switch_consistent=self.switch_consistent)
        elif self.loss_function == 'masked_mae' and ll[0] == 'Softplus':
            return loss.masked_mae_softplus_reg_torch(y_predicted, y_true, sigma_0, self._get_kl_sum() / num_batches, 0, int(ll[1]), switch_consistent=self.switch_consistent)
        elif self.loss_function == 'masked_mse' and ll[0] == 'relu':
            return loss.masked_mse_relu_reg_torch(y_predicted, y_true, sigma_0, self._get_kl_sum() / num_batches, 0, float(ll[1]), switch_consistent=self.switch_consistent)
        elif self.loss_function == 'masked_mse' and ll[0] == 'Softplus':
            return loss.masked_mse_softplus_reg_torch(y_predicted, y_true, sigma_0, self._get_kl_sum() / num_batches, 0, int(ll[1]), switch_consistent=self.switch_consistent)
        else:
            raise NotImplementedError('Unrecognized loss function.')

    def predict(self, batch, batches_seen=None):
        return self.forward(batch, batches_seen)

    def predict_sigma(self, batch, batches_seen=None):
        ll = self.clamp_function.split('_')
        if ll[0] == 'relu':
            return torch.clamp(self.forward_sigma(batch, batches_seen), min=float(ll[1]))
        elif ll[0] == 'Softplus':
            return torch.nn.Softplus(beta=int(ll[1]))(self.forward_sigma(batch, batches_seen))
        else:
            raise NotImplementedError('Unrecognized loss function.')
