"""
    python analyse_result.py --model BDCRNNVariableDecoder --dataset METR_LA --exp_id 89148
    python analyse_result.py --model BDCRNNVariableDecoderShared --dataset PEMS_BAY --exp_id 67654
"""
import argparse
import os.path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from analyse_testing_result import get_datetime
from libcity.config import ConfigParser
from libcity.model import loss
from libcity.utils import get_logger, ensure_dir, get_local_time

if __name__ == '__main__':
    plt.rc('font', family='Times New Roman', size=20)
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='traffic_state_pred')

    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--exp_id', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)

    args = parser.parse_args()
    task, model_name, dataset_name, exp_id = args.task, args.model, args.dataset, args.exp_id

    # load config
    config = ConfigParser(task, model_name, dataset_name, saved_model=False, train=False)
    config['exp_id'] = exp_id
    config['batch_size'] = 288  # 24h=24*60/5
    # logger
    logger = get_logger(config)
    logger.info('Begin analyzing result, task={}, model_name={}, dataset_name={}, exp_id={}'.
                format(str(task), str(model_name), str(dataset_name), str(exp_id)))
    logger.info(config.config)

    batch_size = config['batch_size']

    evaluate_cache_dir = './libcity/cache/{}/evaluate_cache'.format(exp_id)
    analyze_cache_dir = './libcity/cache/{}/analyze_cache/{}'.format(exp_id, get_local_time())
    images_cache_dir = '{}/images'.format(analyze_cache_dir)
    assert os.path.exists(evaluate_cache_dir)
    ensure_dir(analyze_cache_dir)
    ensure_dir(images_cache_dir)

    visited = False
    for filename in os.listdir(evaluate_cache_dir):
        if filename[-4:] == '.npz':
            assert not visited
            arr = np.load(f'{evaluate_cache_dir}/{filename}')
            visited = True

    assert visited
    prediction, truth, outputs, sigmas = arr['prediction'], arr['truth'], arr['outputs'], arr['sigmas']
    logger.info(f'prediction shape: {prediction.shape}')  # (6912, 12, 207, 1) (10432, 12, 325, 1)
    logger.info(f'truth shape: {truth.shape}')  # (6912, 12, 207, 1) (10432, 12, 325, 1)
    logger.info(f'outputs shape: {outputs.shape}')  # (30, 6912, 12, 207, 1) (30, 10432, 12, 325, 1)
    logger.info(f'sigmas shape: {sigmas.shape}')  # (30, 6912, 12, 207, 1) (30, 10432, 12, 325, 1)

    assert outputs.shape == sigmas.shape
    evaluate_rep, num_data, output_window, num_nodes, output_dim = outputs.shape
    prediction, truth, outputs, sigmas = prediction[..., 0], truth[..., 0], outputs[..., 0], sigmas[..., 0]

    if dataset_name == 'METR_LA':
        nodes_list = [0, 149, 181]
    elif dataset_name == 'PEMS_BAY':
        nodes_list = [2, 4, 8, 57, 188, 113, 212, 258]
    else:
        raise NotImplementedError(f'No such dataset of {dataset_name}.')

    metrics = ['masked_MAE', 'masked_MSE', 'masked_RMSE', 'masked_MAPE', 'MAE', 'MSE', 'RMSE', 'MAPE']

    for i in nodes_list:
        logger.info(f'Analyzing node {i}...')
        t_dir = '{}/{}'.format(images_cache_dir, i)
        ensure_dir(t_dir)
        # (num_data, output_window), (evaluate_rep, num_data, output_window)
        prediction_node, truth_node, outputs_node, sigmas_node = prediction[:, :, i], truth[:, :, i], \
            outputs[:, :, :, i], sigmas[:, :, :, i]
        output_res = np.zeros((12, 8))
        for j in range(12):
            for k, metric in enumerate(metrics):
                if metric == 'masked_MAE':
                    output_res[j, k] = loss.masked_mae_torch(torch.Tensor(prediction_node[:, j - 1]), torch.Tensor(truth_node[:, j - 1]), 0).item()
                elif metric == 'masked_MSE':
                    output_res[j, k] = loss.masked_mse_torch(torch.Tensor(prediction_node[:, j - 1]), torch.Tensor(truth_node[:, j - 1]), 0).item()
                elif metric == 'masked_RMSE':
                    output_res[j, k] = loss.masked_rmse_torch(torch.Tensor(prediction_node[:, j - 1]), torch.Tensor(truth_node[:, j - 1]), 0).item()
                elif metric == 'masked_MAPE':
                    output_res[j, k] = loss.masked_mape_torch(torch.Tensor(prediction_node[:, j - 1]), torch.Tensor(truth_node[:, j - 1]), 0).item()
                elif metric == 'MAE':
                    output_res[j, k] = loss.masked_mae_torch(torch.Tensor(prediction_node[:, j - 1]), torch.Tensor(truth_node[:, j - 1])).item()
                elif metric == 'MSE':
                    output_res[j, k] = loss.masked_mse_torch(torch.Tensor(prediction_node[:, j - 1]), torch.Tensor(truth_node[:, j - 1])).item()
                elif metric == 'RMSE':
                    output_res[j, k] = loss.masked_rmse_torch(torch.Tensor(prediction_node[:, j - 1]), torch.Tensor(truth_node[:, j - 1])).item()
                elif metric == 'MAPE':
                    output_res[j, k] = loss.masked_mape_torch(torch.Tensor(prediction_node[:, j - 1]), torch.Tensor(truth_node[:, j - 1])).item()
        df = pd.DataFrame(output_res, columns=metrics)
        # 显示所有列
        pd.set_option('display.max_columns', None)
        # 显示所有行
        pd.set_option('display.max_rows', None)
        # 设置value的显示长度为100，默认为50
        pd.set_option('max_colwidth', 100)
        logger.info(df)
        sigmas_node_2 = sigmas_node * sigmas_node
        outputs_node_2 = outputs_node * outputs_node
        prediction_node_2 = prediction_node * prediction_node
        writer = pd.ExcelWriter(f'{analyze_cache_dir}/{dataset_name}_node_{i}.xlsx')
        for j, time in zip([2, 5, 11], ['15min', '30min', '1h']):
            p, t, o, s = prediction_node[:, j], truth_node[:, j], outputs_node[:, :, j], sigmas_node[:, :, j]
            p2, s2, o2 = prediction_node_2[:, j], sigmas_node_2[:, :, j], outputs_node_2[:, :, j]
            error = p - t
            a_uncertainty = 1 / evaluate_rep * np.sum(s, axis=0)
            e_uncertainty = 1 / evaluate_rep * np.sum(o2, axis=0) - p2
            uncertainty = a_uncertainty + e_uncertainty
            logger.info(f'Masked Aleatoric={loss.masked_uncertainty_torch(torch.Tensor(a_uncertainty), torch.Tensor(t), 0).item()}, '
                        f'Masked Epistemic={loss.masked_uncertainty_torch(torch.Tensor(e_uncertainty), torch.Tensor(t), 0).item()}, '
                        f'Masked Uncertainty={loss.masked_uncertainty_torch(torch.Tensor(uncertainty), torch.Tensor(t), 0).item()}.')
            res = np.stack((p, t, error, uncertainty, a_uncertainty, e_uncertainty), axis=1)
            res = np.concatenate((res, o[:5].T, s[:5].T), axis=1)
            columns_name = ['pred', 'truth', 'error', 'uncertainty', 'a_uncertainty', 'e_uncertainty']
            # columns_name.extend(['output_{}'.format(i) for i in range(evaluate_rep)])
            # columns_name.extend(['sigma_{}'.format(i) for i in range(evaluate_rep)])
            columns_name.extend(['output_{}'.format(i) for i in range(5)])
            columns_name.extend(['sigma_{}'.format(i) for i in range(5)])
            pd_data = pd.DataFrame(res, columns=columns_name)
            pd_data.to_excel(writer, sheet_name=time, float_format='%.4f')
            if j == 2:
                start = 235 if dataset_name == 'METR_LA' else 60  # 2012-06-05 00:00:00 / 2017-05-26 00:00:00
            elif j == 5:
                start = 232 if dataset_name == 'METR_LA' else 57
            elif j == 11:
                start = 226 if dataset_name == 'METR_LA' else 51
            else:
                raise NotImplementedError(f'No such timestamp of {j}')
            # for k in range(start, num_data, batch_size):
            #     t_num = min(batch_size + 1, num_data - k)
            #     if t_num < batch_size + 1:
            #         continue
            #     x = np.arange(k, k + t_num)
            #     mask = np.where(t[k: k + t_num], 1, 0)
            #     plt.xlim(k, k + t_num - 1)
            #     plt.xticks(np.arange(k, k + t_num, 36), [get_datetime(dataset_name, k, j, fmt='%H:%M\n%b-%d')] +
            #                [f'{_}:00' for _ in range(3, 24, 3)] +
            #                [get_datetime(dataset_name, k + t_num - 1, j, fmt='%H:%M\n%b-%d')])
            #     plt.ylabel('mile/h')
            #     plt.plot(x, np.abs(error[k:k + t_num]) * mask, label='|error|')
            #     plt.plot(x, a_uncertainty[k:k + t_num] * mask, label='a_uncertainty')
            #     plt.plot(x, e_uncertainty[k:k + t_num] * mask, label='e_uncertainty')
            #     plt.plot(x, uncertainty[k:k + t_num] * mask, label='uncertainty')
            #     plt.legend(labelspacing=0.05)
            #     plt.savefig(f'{t_dir}/{dataset_name}_node_{i}_batch_{k // batch_size}_{time}.svg',
            #                 bbox_inches='tight')
            #     plt.close()
        writer.close()
