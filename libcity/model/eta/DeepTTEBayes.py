import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from libcity.model import loss
from libcity.model.gaussian_layers.rand_conv import RandConv1d
from libcity.model.gaussian_layers.rand_linear import RandLinear
from libcity.model.gaussian_layers.rand_rnn import RandLSTM


def normalize(data, mean, std):
    return (data - mean) / std


def unnormalize(data, mean, std):
    return data * std + mean


def get_local_seq(full_seq, kernel_size, mean, std, device=torch.device('cpu')):
    seq_len = full_seq.size()[1]

    indices = torch.LongTensor(seq_len).to(device)

    torch.arange(0, seq_len, out=indices)

    indices = Variable(indices, requires_grad=False)

    first_seq = torch.index_select(full_seq, dim=1, index=indices[kernel_size - 1:])
    second_seq = torch.index_select(full_seq, dim=1, index=indices[:-kernel_size + 1])

    local_seq = first_seq - second_seq

    local_seq = (local_seq - mean) / std

    return local_seq


class RandGeoConv(nn.Module):
    def __init__(self, kernel_size, num_filter, data_feature={}, device=torch.device('cpu'),
                 random_bias=True, kl_bias=True, sigma_pi=1.0, sigma_start=1.0):
        super(RandGeoConv, self).__init__()

        self.kernel_size = kernel_size
        self.num_filter = num_filter
        self.data_feature = data_feature
        self.device = device

        self.state_em = nn.Embedding(2, 2)
        self.process_coords = RandLinear(4, 16, random_bias=random_bias, kl_bias=kl_bias,
                                         sigma_pi=sigma_pi, sigma_start=sigma_start)
        self.conv = RandConv1d(16, self.num_filter, self.kernel_size, random_bias=random_bias, kl_bias=kl_bias,
                               sigma_pi=sigma_pi, sigma_start=sigma_start)

    def forward(self, batch):
        longi_mean, longi_std = self.data_feature["longi_mean"], self.data_feature["longi_std"]
        current_longi = normalize(batch["current_longi"], longi_mean, longi_std)
        lngs = torch.unsqueeze(current_longi, dim=2)
        lati_mean, lati_std = self.data_feature["lati_mean"], self.data_feature["lati_std"]
        current_lati = normalize(batch["current_lati"], lati_mean, lati_std)
        lats = torch.unsqueeze(current_lati, dim=2)

        states = self.state_em(batch['current_state'].long())

        locs = torch.cat((lngs, lats, states), dim=2)

        # map the coords into 16-dim vector
        locs = torch.tanh(self.process_coords(locs))
        locs = locs.permute(0, 2, 1)

        conv_locs = F.elu(self.conv(locs)).permute(0, 2, 1)

        dist_gap_mean, dist_gap_std = self.data_feature["dist_gap_mean"], self.data_feature["dist_gap_std"]
        current_dis = normalize(batch["current_dis"], dist_gap_mean, dist_gap_std)

        # calculate the dist for local paths
        local_dist = get_local_seq(current_dis, self.kernel_size, dist_gap_mean, dist_gap_std, self.device)
        local_dist = torch.unsqueeze(local_dist, dim=2)

        conv_locs = torch.cat((conv_locs, local_dist), dim=2)

        return conv_locs

    def set_shared_eps(self):
        self.process_coords.set_shared_eps()
        self.conv.set_shared_eps()

    def clear_shared_eps(self):
        self.process_coords.clear_shared_eps()
        self.conv.clear_shared_eps()

    def get_kl_sum(self):
        return self.process_coords.get_kl_sum() + self.conv.get_kl_sum()


class Attr(nn.Module):
    def __init__(self, embed_dims, data_feature):
        super(Attr, self).__init__()

        self.embed_dims = embed_dims
        self.data_feature = data_feature

        for name, dim_in, dim_out in self.embed_dims:
            self.add_module(name + '_em', nn.Embedding(dim_in, dim_out))

    def out_size(self):
        sz = 0
        for _, _, dim_out in self.embed_dims:
            sz += dim_out
        # append total distance
        return sz + 1

    def forward(self, batch):
        em_list = []
        for name, _, _ in self.embed_dims:
            embed = getattr(self, name + '_em')
            attr_t = batch[name]

            attr_t = torch.squeeze(embed(attr_t))

            em_list.append(attr_t)

        dist_mean, dist_std = self.data_feature["dist_mean"], self.data_feature["dist_std"]
        dist = normalize(batch["dist"], dist_mean, dist_std)
        dist = normalize(dist, dist_mean, dist_std)
        em_list.append(dist)

        return torch.cat(em_list, dim=1)


class RandSpatioTemporal(nn.Module):
    """
    attr_size: the dimension of attr_net output
    pooling optitions: last, mean, attention
    """

    def __init__(self, attr_size, kernel_size=3, num_filter=32, pooling_method='attention',
                 rnn_type='LSTM', rnn_num_layers=1, hidden_size=128,
                 data_feature={}, device=torch.device('cpu'),
                 random_bias=True, kl_bias=True, sigma_pi=1.0, sigma_start=1.0):
        super(RandSpatioTemporal, self).__init__()

        self.kernel_size = kernel_size
        self.num_filter = num_filter
        self.pooling_method = pooling_method
        self.hidden_size = hidden_size

        self.data_feature = data_feature
        self.device = device

        self.geo_conv = RandGeoConv(
            kernel_size=kernel_size,
            num_filter=num_filter,
            data_feature=data_feature,
            device=device,
            random_bias=random_bias,
            kl_bias=kl_bias,
            sigma_pi=sigma_pi,
            sigma_start=sigma_start
        )
        # num_filter: output size of each GeoConv + 1:distance of local path + attr_size: output size of attr component
        if rnn_type.upper() == 'LSTM':
            self.rnn = RandLSTM(
                input_size=num_filter + 1 + attr_size,
                hidden_size=hidden_size,
                num_layers=rnn_num_layers,
                batch_first=True,
                random_bias=random_bias,
                kl_bias=kl_bias,
                sigma_pi=sigma_pi,
                sigma_start=sigma_start
            )
        # elif rnn_type.upper() == 'RNN':
        #     self.rnn = nn.RNN(
        #         input_size=num_filter + 1 + attr_size,
        #         hidden_size=hidden_size,
        #         num_layers=rnn_num_layers,
        #         batch_first=True
        #     )
        else:
            raise ValueError('invalid rnn_type, please select `RNN` or `LSTM`')
        if pooling_method == 'attention':
            self.attr2atten = RandLinear(attr_size, hidden_size, random_bias=random_bias, kl_bias=kl_bias,
                                         sigma_pi=sigma_pi, sigma_start=sigma_start)
        else:
            self.attr2atten = None

    def out_size(self):
        # return the output size of spatio-temporal component
        return self.hidden_size

    def mean_pooling(self, hiddens, lens):
        # note that in pad_packed_sequence, the hidden states are padded with all 0
        hiddens = torch.sum(hiddens, dim=1, keepdim=False)

        lens = torch.FloatTensor(lens).to(self.device)

        lens = Variable(torch.unsqueeze(lens, dim=1), requires_grad=False)

        hiddens = hiddens / lens

        return hiddens

    def atten_pooling(self, hiddens, attr_t):
        atten = torch.tanh(self.attr2atten(attr_t)).permute(0, 2, 1)

        # hidden b*s*f atten b*f*1 alpha b*s*1 (s is length of sequence)
        alpha = torch.bmm(hiddens, atten)
        alpha = torch.exp(-alpha)

        # The padded hidden is 0 (in pytorch), so we do not need to calculate the mask
        alpha = alpha / torch.sum(alpha, dim=1, keepdim=True)

        hiddens = hiddens.permute(0, 2, 1)
        hiddens = torch.bmm(hiddens, alpha)
        hiddens = torch.squeeze(hiddens)

        return hiddens

    def forward(self, batch, attr_t):
        conv_locs = self.geo_conv(batch)

        attr_t = torch.unsqueeze(attr_t, dim=1)
        expand_attr_t = attr_t.expand(conv_locs.size()[:2] + (attr_t.size()[-1],))

        # concat the loc_conv and the attributes
        conv_locs = torch.cat((conv_locs, expand_attr_t), dim=2)

        lens = [batch["current_longi"].shape[1]] * batch["current_longi"].shape[0]
        lens = list(map(lambda x: x - self.kernel_size + 1, lens))

        packed_inputs = nn.utils.rnn.pack_padded_sequence(conv_locs, lens, batch_first=True)

        packed_hiddens, _ = self.rnn(packed_inputs)
        hiddens, lens = nn.utils.rnn.pad_packed_sequence(packed_hiddens, batch_first=True)

        if self.pooling_method == 'mean':
            return packed_hiddens, lens, self.mean_pooling(hiddens, lens)
        else:
            # self.pooling_method == 'attention'
            return packed_hiddens, lens, self.atten_pooling(hiddens, attr_t)

    def set_shared_eps(self):
        self.geo_conv.set_shared_eps()
        self.rnn.set_shared_eps()
        if self.attr2atten is not None:
            self.attr2atten.set_shared_eps()

    def clear_shared_eps(self):
        self.geo_conv.clear_shared_eps()
        self.rnn.clear_shared_eps()
        if self.attr2atten is not None:
            self.attr2atten.clear_shared_eps()

    def get_kl_sum(self):
        if self.attr2atten is None:
            return self.geo_conv.get_kl_sum() + self.rnn.get_kl_sum()
        return self.geo_conv.get_kl_sum() + self.rnn.get_kl_sum() + self.attr2atten.get_kl_sum()


class RandEntireEstimator(nn.Module):
    def __init__(self, input_size, num_final_fcs, hidden_size=128,
                 random_bias=True, kl_bias=True, sigma_pi=1.0, sigma_start=1.0):
        super(RandEntireEstimator, self).__init__()

        self.input2hid = RandLinear(input_size, hidden_size, random_bias=random_bias, kl_bias=kl_bias,
                                    sigma_pi=sigma_pi, sigma_start=sigma_start)

        self.residuals = nn.ModuleList()
        for i in range(num_final_fcs):
            self.residuals.append(RandLinear(hidden_size, hidden_size, random_bias=random_bias, kl_bias=kl_bias,
                                             sigma_pi=sigma_pi, sigma_start=sigma_start))

        self.hid2out = RandLinear(hidden_size, 1, random_bias=random_bias, kl_bias=kl_bias,
                                  sigma_pi=sigma_pi, sigma_start=sigma_start)

    def forward(self, attr_t, sptm_t):
        inputs = torch.cat((attr_t, sptm_t), dim=1)

        hidden = F.leaky_relu(self.input2hid(inputs))

        for i in range(len(self.residuals)):
            residual = F.leaky_relu(self.residuals[i](hidden))
            hidden = hidden + residual

        out = self.hid2out(hidden)

        return out

    def eval_on_batch(self, pred, label, mean, std, mode='masked_mape', **kwargs):
        label = label

        label = label * std + mean
        pred = pred * std + mean

        if mode == 'masked_mape':
            return loss.masked_mape_torch(pred, label)
        elif mode == 'masked_mae':
            return loss.masked_mae_const_reg_torch(pred, label, **kwargs)
        elif mode == 'masked_mse':
            return loss.masked_mse_const_reg_torch(pred, label, **kwargs)
        else:
            raise NotImplementedError('No such loss function for eval_on_batch in EntireEstimator.')

    def set_shared_eps(self):
        self.input2hid.set_shared_eps()
        for layer in self.residuals:
            layer.set_shared_eps()
        self.hid2out.set_shared_eps()

    def clear_shared_eps(self):
        self.input2hid.clear_shared_eps()
        for layer in self.residuals:
            layer.clear_shared_eps()
        self.hid2out.clear_shared_eps()

    def get_kl_sum(self):
        kl_sum = self.input2hid.get_kl_sum() + self.hid2out.get_kl_sum()
        for layer in self.residuals:
            kl_sum += layer.get_kl_sum()
        return kl_sum


class RandLocalEstimator(nn.Module):
    def __init__(self, input_size, eps=10,
                 random_bias=True, kl_bias=True, sigma_pi=1.0, sigma_start=1.0):
        super(RandLocalEstimator, self).__init__()

        self.input2hid = RandLinear(input_size, 64, random_bias=random_bias, kl_bias=kl_bias,
                                    sigma_pi=sigma_pi, sigma_start=sigma_start)
        self.hid2hid = RandLinear(64, 32, random_bias=random_bias, kl_bias=kl_bias,
                                  sigma_pi=sigma_pi, sigma_start=sigma_start)
        self.hid2out = RandLinear(32, 1, random_bias=random_bias, kl_bias=kl_bias,
                                  sigma_pi=sigma_pi, sigma_start=sigma_start)

        self.eps = eps

    def forward(self, sptm_s):
        hidden = F.leaky_relu(self.input2hid(sptm_s))

        hidden = F.leaky_relu(self.hid2hid(hidden))

        out = self.hid2out(hidden)

        return out

    def eval_on_batch(self, pred, lens, label, mean, std, mode='masked_mape', **kwargs):
        label = nn.utils.rnn.pack_padded_sequence(label, lens, batch_first=True)[0]
        label = label

        label = label * std + mean
        pred = pred * std + mean

        if mode == 'masked_mape':
            return loss.masked_mape_torch(pred, label, eps=self.eps)
        elif mode == 'masked_mae':
            return loss.masked_mae_const_reg_torch(pred, label, **kwargs)
        elif mode == 'masked_mse':
            return loss.masked_mse_const_reg_torch(pred, label, **kwargs)
        else:
            raise NotImplementedError('No such loss function for eval_on_batch in EntireEstimator.')

    def set_shared_eps(self):
        self.input2hid.set_shared_eps()
        self.hid2hid.set_shared_eps()
        self.hid2out.set_shared_eps()

    def clear_shared_eps(self):
        self.input2hid.clear_shared_eps()
        self.hid2hid.clear_shared_eps()
        self.hid2out.clear_shared_eps()

    def get_kl_sum(self):
        return self.input2hid.get_kl_sum() + self.hid2hid.get_kl_sum() + self.hid2out.get_kl_sum()
