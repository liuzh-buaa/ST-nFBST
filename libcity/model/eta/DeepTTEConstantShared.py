import torch
import torch.nn as nn

from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel
from libcity.model.eta.DeepTTEBayes import Attr, RandSpatioTemporal, RandEntireEstimator, RandLocalEstimator, normalize, \
    unnormalize, get_local_seq


class DeepTTEConstantShared(AbstractTrafficStateModel):
    def __init__(self, config, data_feature):
        super(DeepTTEConstantShared, self).__init__(config, data_feature)
        self.config = config
        self.data_feature = data_feature
        self.device = config.get('device', torch.device('cpu'))

        uid_emb_size = config.get("uid_emb_size", 16)
        weekid_emb_size = config.get("weekid_emb_size", 3)
        timdid_emb_size = config.get("timdid_emb_size", 8)
        uid_size = data_feature.get("uid_size", 24000)
        embed_dims = [
            ('uid', uid_size, uid_emb_size),
            ('weekid', 7, weekid_emb_size),
            ('timeid', 1440, timdid_emb_size),
        ]

        # parameter of attribute / spatio-temporal component
        self.kernel_size = config.get('kernel_size', 3)
        num_filter = config.get('num_filter', 32)
        pooling_method = config.get("pooling_method", "attention")

        # parameter of multi-task learning component
        num_final_fcs = config.get('num_final_fcs', 3)
        final_fc_size = config.get('final_fc_size', 128)
        self.alpha = config.get('alpha', 0.3)

        rnn_type = config.get('rnn_type', 'LSTM')
        rnn_num_layers = config.get('rnn_num_layers', 1)
        hidden_size = config.get('hidden_size', 128)

        self.eps = config.get('eps', 10)

        # attribute component
        self.attr_net = Attr(embed_dims, data_feature)

        sigma_pi = config.get('sigma_pi')
        sigma_start = config.get('sigma_start')
        self.sigma_0 = config.get('sigma_0')
        self.loss_function = config.get('loss_function')

        # spatio-temporal component
        self.spatio_temporal = RandSpatioTemporal(
            attr_size=self.attr_net.out_size(),
            kernel_size=self.kernel_size,
            num_filter=num_filter,
            pooling_method=pooling_method,
            rnn_type=rnn_type,
            rnn_num_layers=rnn_num_layers,
            hidden_size=hidden_size,
            data_feature=data_feature,
            device=self.device,
            sigma_pi=sigma_pi,
            sigma_start=sigma_start
        )

        self.entire_estimate = RandEntireEstimator(
            input_size=self.spatio_temporal.out_size() + self.attr_net.out_size(),
            num_final_fcs=num_final_fcs,
            hidden_size=final_fc_size,
            sigma_pi=sigma_pi,
            sigma_start=sigma_start
        )

        self.local_estimate = RandLocalEstimator(
            input_size=self.spatio_temporal.out_size(),
            eps=self.eps,
            sigma_pi=sigma_pi,
            sigma_start=sigma_start
        )

        self._init_weight()

    def _init_weight(self):
        for name, param in self.named_parameters():
            if name.find('.bias') != -1:
                param.data.fill_(0)
            elif name.find('.weight') != -1:
                nn.init.xavier_uniform_(param.data)

    def forward(self, batch):
        self.set_shared_eps()
        attr_t = self.attr_net(batch)

        # sptm_s: hidden sequence (B * T * F); sptm_l: lens (list of int);
        # sptm_t: merged tensor after attention/mean pooling
        sptm_s, sptm_l, sptm_t = self.spatio_temporal(batch, attr_t)

        entire_out = self.entire_estimate(attr_t, sptm_t)

        # sptm_s is a packed sequence (see pytorch doc for details), only used during the training
        if self.training:
            local_out = self.local_estimate(sptm_s[0])
            self.clear_shared_eps()
            return entire_out, (local_out, sptm_l)
        else:
            self.clear_shared_eps()
            return entire_out

    def calculate_loss(self, batch):
        assert self.training, 'Call calculate_loss func incorrectly.'
        entire_out, (local_out, local_length) = self.predict(batch)

        time_mean, time_std = self.data_feature["time_mean"], self.data_feature["time_std"]
        entire_out = normalize(entire_out, time_mean, time_std)
        time = normalize(batch["time"], time_mean, time_std)
        entire_loss = self.entire_estimate.eval_on_batch(entire_out, time, time_mean, time_std, mode=self.loss_function,
                                                         sigma_0=self.sigma_0, reg=self.entire_estimate.get_kl_sum())

        # get the mean/std of each local path
        time_gap_mean, time_gap_std = self.data_feature["time_gap_mean"], self.data_feature["time_gap_std"]
        mean, std = (self.kernel_size - 1) * time_gap_mean, (self.kernel_size - 1) * time_gap_std
        current_tim = normalize(batch["current_tim"], time_gap_mean, time_gap_std)

        # get ground truth of each local path
        local_label = get_local_seq(current_tim, self.kernel_size, mean, std, self.device)
        local_loss = self.local_estimate.eval_on_batch(local_out, local_length, local_label, mean, std,
                                                       mode=self.loss_function,
                                                       sigma_0=self.sigma_0, reg=self.local_estimate.get_kl_sum())

        return (1 - self.alpha) * entire_loss + self.alpha * local_loss

    def calculate_eval_loss(self, batch):
        entire_out = self.predict(batch)
        time_mean, time_std = self.data_feature["time_mean"], self.data_feature["time_std"]
        entire_out = normalize(entire_out, time_mean, time_std)
        time = normalize(batch["time"], time_mean, time_std)
        entire_loss = self.entire_estimate.eval_on_batch(entire_out, time, time_mean, time_std)
        return entire_loss

    def predict(self, batch):
        time_mean, time_std = self.data_feature["time_mean"], self.data_feature["time_std"]
        if self.training:
            entire_out, (local_out, local_length) = self.forward(batch)
            entire_out = unnormalize(entire_out, time_mean, time_std)
            return entire_out, (local_out, local_length)
        else:
            entire_out = self.forward(batch)
            entire_out = unnormalize(entire_out, time_mean, time_std)
            return entire_out

    def set_shared_eps(self):
        self.spatio_temporal.set_shared_eps()
        self.entire_estimate.set_shared_eps()
        self.local_estimate.set_shared_eps()

    def clear_shared_eps(self):
        self.spatio_temporal.clear_shared_eps()
        self.entire_estimate.clear_shared_eps()
        self.local_estimate.clear_shared_eps()

    def get_kl_sum(self):
        return self.spatio_temporal.get_kl_sum() + self.entire_estimate.get_kl_sum() + self.local_estimate.get_kl_sum()
