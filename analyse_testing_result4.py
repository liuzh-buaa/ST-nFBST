"""
    根据某个传感器某个时刻的速度绘制此时受其他传感器的影响
    `index`: 对应某个时刻
    `nn`: 对应某个传感器
    `output_window`: [2,5,11]，根据输出的哪个预测窗口算平均
    `input_dim`: [0,1,-1]，对哪个输入维度做检验，如果是-1则取两个维度平均值
"""
import argparse
import os.path

import matplotlib.pyplot as plt
import numpy as np

from analyse_testing_result import get_exp_id
from libcity.utils import ensure_dir
from visualize_sensor import visualize_sensor_varying

if __name__ == '__main__':
    plt.rc('font', family='Times New Roman', size=20)
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='METR_LA', choices=['METR_LA', 'PEMS_BAY'])
    parser.add_argument('--index', type=int, default=19)
    parser.add_argument('--nn', type=int, default=0, help='node number')
    parser.add_argument('--od', type=int, default=0, help='output dim')
    parser.add_argument('--output_window', type=int, default=2, choices=[2, 5, 11])
    parser.add_argument('--input_dim', type=int, default=0, choices=[0, 1, -1])

    # 解析参数
    args = parser.parse_args()

    sensors = 207 if args.dataset == 'METR_LA' else 325
    exp_id = get_exp_id(args.dataset, args.index)

    res_dir = f'analyse_testing_result4/{args.dataset}/{args.nn}/{args.output_window}_{args.input_dim}'
    ensure_dir(res_dir)

    testing_res_dir = './libcity/cache/{}/testing_cache'.format(exp_id)
    assert os.path.exists(testing_res_dir)

    ext = {}
    for test_nn in range(sensors):
        filename = 'ps_testing_{}_{}_{}_{}_{}.npy'.format(args.index, args.output_window, args.nn, args.od, test_nn)
        read_data = 1 - np.load(os.path.join(testing_res_dir, filename))

        if test_nn == args.nn:
            if args.input_dim == -1:
                ext[test_nn] = np.mean(read_data[11, :])
            else:
                ext[test_nn] = read_data[11, args.input_dim]
        else:
            if args.input_dim == -1:
                ext[test_nn] = np.mean(read_data[0, :])
            else:
                ext[test_nn] = read_data[0, args.input_dim]

    visualize_sensor_varying('METR_LA', args.nn, ext, filename=f'{res_dir}/{args.nn}_{args.index}.html')
