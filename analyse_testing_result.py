"""
    python analyse_testing_result.py (BDCRNNVariableDecoder; METR_LA )
    python analyse_testing_result.py (BDCRNNVariableDecoderShared; PEMS_BAY )
"""
import argparse
import os.path
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from visualize_sensor import visualize_sensor


def get_datetime(dataset, index, output_window=2, fmt=None):
    if dataset == 'METR_LA':
        if output_window == 2:
            base_time = datetime(2012, 6, 4, 4, 25)  # 初始时间
        elif output_window == 5:
            base_time = datetime(2012, 6, 4, 4, 40)  # 初始时间
        elif output_window == 11:
            base_time = datetime(2012, 6, 4, 5, 10)  # 初始时间
        else:
            raise NotImplementedError(f'No such output_window support.')
    elif dataset == 'PEMS_BAY':
        if output_window == 2:
            base_time = datetime(2017, 5, 25, 19, 00)  # 初始时间
        elif output_window == 5:
            base_time = datetime(2017, 5, 25, 19, 15)  # 初始时间
        elif output_window == 11:
            base_time = datetime(2017, 5, 25, 19, 45)  # 初始时间
        else:
            raise NotImplementedError(f'No such output_window support.')
    else:
        raise NotImplementedError(f'No such dataset of {dataset}.')

    time_interval = timedelta(minutes=5)  # 时间间隔为5分钟
    result_time = base_time + index * time_interval  # 计算结果时间

    if fmt is None:
        return result_time.strftime('%Y-%m-%d %H:%M')  # 格式化输出字符串
    else:
        return result_time.strftime(fmt)  # 格式化输出字符串


def get_exp_id(data, index):
    if data == 'METR_LA':
        if index < 10:
            return 11529
        elif index < 20:
            return 15333
        elif index < 26:
            return 98337
        elif index < 27:
            return 80588
        elif index < 32:
            return 63269
        elif index < 37:
            return 90374
        elif index < 42:
            return 70110
        elif index < 48:
            return 72874
        elif index < 54:
            return 27302
        elif index < 152:
            raise NotImplementedError()
        elif index < 158:
            return 31246
        elif index < 164:
            return 3555
        elif index < 170:
            return 5112
        elif index < 175:
            return 36239
        elif index < 182:
            return 21982
        elif index < 188:
            return 57707
        elif index < 235:
            raise NotImplementedError()
        elif index < 240:
            return 67445    # note test based on the mean of (speed and time)
        elif index < 245:
            return 31478    # note test based on the mean of (speed and time)
        elif index < 250:
            return 29188    # note test based on the mean of (speed and time)
        elif index < 255:
            return 72032    # note test based on the mean of (speed and time)
        else:
            raise NotImplementedError(f'Have not testing index {index} of {data}')
    elif data == 'PEMS_BAY':
        if index < 10:
            return 53968
        elif index < 20:
            return 54352
        else:
            raise NotImplementedError(f'Have not testing index {index} of {data}')
    else:
        raise NotImplementedError(f'No such data {data}')


if __name__ == '__main__':
    plt.rc('font', family='Times New Roman', size=20)
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='METR_LA', choices=['METR_LA', 'PEMS_BAY'])
    parser.add_argument('--index', type=int, default=1)
    parser.add_argument('--ow', type=int, default=2, help='output window')
    parser.add_argument('--nn', type=int, default=0, help='node number')
    parser.add_argument('--od', type=int, default=0, help='output dim')

    # 解析参数
    args = parser.parse_args()

    sensors = 207 if args.dataset == 'METR_LA' else 325
    if args.ow == -1:
        args.ow = [2, 5, 11]
    else:
        args.ow = [args.ow]

    if args.nn == -1:
        args.nn = list(range(sensors))
    else:
        args.nn = [args.nn]

    exp_id = get_exp_id(args.dataset, args.index)

    testing_res_dir = './libcity/cache/{}/testing_cache'.format(exp_id)
    assert os.path.exists(testing_res_dir)

    # tot_data = []
    # for nn in args.nn:
    #     for ow in args.ow:
    #         filename = 'ps_testing_{}_{}_{}_{}_{}.npy'.format(args.index, ow, args.nn, args.od, nn)
    #         read_data = 1 - np.load(os.path.join(testing_res_dir, filename))
    #
    #         print('Reading testing results of DATA {}: Output window {} of Node {} wrt Node {} - {}'.format(
    #             args.index, ow, nn, nn, read_data.shape))
    #         print(read_data.T)  # (12, 2)
    #         tot_data.append(read_data[:, 0])
    #
    # tot_data = np.stack(tot_data, axis=1)
    # mean_tot_data = np.mean(tot_data, axis=1)
    # print(mean_tot_data)

    for nn in args.nn:
        tot_data = []
        indices = []
        for test_nn in range(sensors):
            filename = 'ps_testing_{}_{}_{}_{}_{}.npy'.format(args.index, 2, nn, args.od, test_nn)
            read_data = 1 - np.load(os.path.join(testing_res_dir, filename))

            if test_nn == nn:
                tot_data.append(read_data[2, :])
            else:
                tot_data.append(read_data[0, :])

            if tot_data[-1][0] < 0.1:
                indices.append(test_nn)

        tot_data = np.stack(tot_data, axis=0)
        print('Reading testing results of DATA {}: Output window {} of Node {} wrt total nodes - {}'.format(
            args.index, 2, nn, tot_data.shape))

        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        df = pd.DataFrame(tot_data, index=range(len(tot_data)))
        print(df)

        visualize_sensor(args.dataset, [nn], indices, f'tmp_{nn}_sig.html')
        visualize_sensor(args.dataset, [nn], list(set(range(sensors)).difference(indices)) + [nn], f'tmp_{nn}_insig.html')
