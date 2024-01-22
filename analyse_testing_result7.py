"""
    根据某个传感器某个时刻的速度绘制此时受其他传感器的影响（与analyse_testing_result6.py功能相同，不过挑选出我们关注的点分析）
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
from analyse_testing_result6 import get_metr_la_adjacency_matrix
from libcity.utils import ensure_dir
from visualize_sensor import visualize_sensor_marked, visualize_sensor_varying

# {output_window: {sensor: index}}
tot_data = {
    2: {
        181: [1, 2, 3, 4, 35, 38, 39, 41],
        149: [1, 2, 3, 4, 153, 154, 164, 165]
    },
    11: {
        181: [32]
    }
}


if __name__ == '__main__':
    plt.rc('font', family='Times New Roman', size=20)
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='METR_LA', choices=['METR_LA', 'PEMS_BAY'])
    parser.add_argument('--od', type=int, default=0, help='output dim')
    parser.add_argument('--input_dim', type=int, default=0, choices=[0, 1, -1, -2])

    # 解析参数
    args = parser.parse_args()

    sensors = 207 if args.dataset == 'METR_LA' else 325
    adj_mx = get_metr_la_adjacency_matrix('METR_LA')

    speeds_dir = f'raw_data/METR_LA'

    for args.output_window, kv in tot_data.items():
        for args.nn, indices in kv.items():
            for args.index in indices:
                exp_id = get_exp_id(args.dataset, args.index)

                res_dir = f'analyse_testing_result7/{args.dataset}/{args.nn}/{args.output_window}_{args.input_dim}'
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

                visualize_sensor_varying('METR_LA', args.nn, ext, filename=f'{res_dir}/{args.nn}_{args.index}.html', speeds=speeds, normalized=True)
