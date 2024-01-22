"""
    根据某个传感器某个时刻的速度绘制此时受其他传感器的影响
    `index`: 对应某个时刻
    `nn`: 对应某个传感器
    `output_window`: [2,5,11]，根据输出的哪个预测窗口算平均
    `input_dim`: [0,1,-1,-2]，对哪个输入维度做检验，如果是-1则取两个维度平均值，-2取两个维度的最大值
"""
import argparse
import os.path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyse_testing_result import get_exp_id
from libcity.utils import ensure_dir
from visualize_sensor import visualize_sensor_marked, visualize_sensor_varying


def get_metr_la_adjacency_matrix(dataset):
    # TrafficStateDataset._load_geo()
    geofile = pd.read_csv(f'./raw_data/{dataset}/{dataset}.geo')
    geo_ids = list(geofile['geo_id'])
    geo_to_ind = {}
    ind_to_geo = {}
    for index, idx in enumerate(geo_ids):
        geo_to_ind[idx] = index
        ind_to_geo[index] = idx
    # TrafficStateDataset._load_rel()
    relfile = pd.read_csv(f'./raw_data/{dataset}/{dataset}.rel')
    distance_df = relfile[~relfile['cost'].isna()][[
        'origin_id', 'destination_id', 'cost']]
    # 把数据转换成矩阵的形式
    adj_mx = np.zeros((len(geo_ids), len(geo_ids)), dtype=np.float32)
    adj_mx[:] = np.inf
    for row in distance_df.values:
        if row[0] not in geo_to_ind or row[1] not in geo_to_ind:
            continue
        # 保留原始的距离数值
        adj_mx[geo_to_ind[row[0]], geo_to_ind[row[1]]] = row[2]
    return adj_mx


if __name__ == '__main__':
    plt.rc('font', family='Times New Roman', size=20)
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='METR_LA', choices=['METR_LA', 'PEMS_BAY'])
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--nn', type=int, default=184, help='node number')
    parser.add_argument('--od', type=int, default=0, help='output dim')
    parser.add_argument('--output_window', type=int, default=2, choices=[2, 5, 11])
    parser.add_argument('--input_dim', type=int, default=0, choices=[0, 1, -1, -2])

    # 解析参数
    args = parser.parse_args()

    sensors = 207 if args.dataset == 'METR_LA' else 325
    adj_mx = get_metr_la_adjacency_matrix('METR_LA')

    speeds_dir = f'raw_data/METR_LA'

    for args.index in range(0, 10):
        for args.output_window in [2, 5, 11]:
            exp_id = get_exp_id(args.dataset, args.index)

            res_dir = f'analyse_testing_result6/{args.dataset}/{args.nn}/{args.output_window}_{args.input_dim}'
            ensure_dir(res_dir)

            testing_res_dir = './libcity/cache/{}/testing_cache'.format(exp_id)
            assert os.path.exists(testing_res_dir)

            ext = {}
            speeds = {}
            for test_nn in range(sensors):
                if adj_mx[test_nn][args.nn] == np.inf:
                    continue

                speeds[test_nn] = np.array(pd.read_csv(os.path.join(speeds_dir, f'{test_nn}.csv')))[27408+args.index+args.output_window]

                filename = 'ps_testing_{}_{}_{}_{}_{}.npy'.format(args.index, args.output_window, args.nn, args.od, test_nn)
                read_data = 1 - np.load(os.path.join(testing_res_dir, filename))

                if test_nn == args.nn:
                    if args.input_dim == -1:
                        ext[test_nn] = np.mean(read_data[11, :])
                    elif args.input_dim == -2:
                        ext[test_nn] = np.max(read_data[11, :])
                    else:
                        ext[test_nn] = read_data[11, args.input_dim]
                else:
                    if args.input_dim == -1:
                        ext[test_nn] = np.mean(read_data[0, :])
                    elif args.input_dim == -2:
                        ext[test_nn] = np.max(read_data[0, :])
                    else:
                        ext[test_nn] = read_data[0, args.input_dim]

            visualize_sensor_varying('METR_LA', args.nn, ext, filename=f'{res_dir}/{args.nn}_{args.index}.html', speeds=speeds)
