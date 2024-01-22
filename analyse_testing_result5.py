"""
    根据所有预测结果，评估某个传感器对其他传感器的影响平均值
    `indices`: 所有的预测结果范围
    `include_self`: True/False，计算影响时是否包含对自己的影响
    `output_window`: [2,5,11]，根据输出的哪个预测窗口算平均
    `input_dim`: [0,1,-1]，对哪个输入维度做检验，如果是-1则取两个维度平均值
"""
import argparse
import os.path

import matplotlib.pyplot as plt
import numpy as np

from analyse_testing_result import get_exp_id
from libcity.utils import ensure_dir, str2bool
from visualize_sensor import visualize_sensor_varying

if __name__ == '__main__':
    plt.rc('font', family='Times New Roman', size=20)
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='METR_LA', choices=['METR_LA', 'PEMS_BAY'])
    parser.add_argument('--include_self', type=str2bool)
    parser.add_argument('--output_window', type=int, default=2, choices=[2, 5, 11])
    parser.add_argument('--input_dim', type=int, default=0, choices=[0, 1, -1])

    # 解析参数
    args = parser.parse_args()

    sensors = 207 if args.dataset == 'METR_LA' else 325
    indices = list(range(0, 25))
    res_dir = f'analyse_testing_result5/{args.dataset}/{args.include_self}_{args.output_window}_{args.input_dim}'
    ensure_dir(res_dir)

    ll = []
    for index in indices:
        ext_filename = f'{res_dir}/ext_{index}.npy'
        if os.path.isfile(ext_filename):
            ext = np.load(ext_filename, allow_pickle=True).item()
            print(f'Loading from {ext_filename}.')
        else:
            ext = {sensor: 0 for sensor in range(sensors)}
            exp_id = get_exp_id(args.dataset, index)
            testing_res_dir = './libcity/cache/{}/testing_cache'.format(exp_id)
            assert os.path.exists(testing_res_dir)
            for nn in range(sensors):
                for test_nn in range(sensors):
                    if not args.include_self and test_nn == nn:
                        continue

                    filename = 'ps_testing_{}_{}_{}_{}_{}.npy'.format(index, args.output_window, nn, 0, test_nn)
                    read_data = 1 - np.load(os.path.join(testing_res_dir, filename))

                    if test_nn == nn:
                        if args.input_dim == -1:
                            ext[test_nn] += np.mean(read_data[11, :])
                        else:
                            ext[test_nn] += read_data[11, args.input_dim]
                    else:
                        if args.input_dim == -1:
                            ext[test_nn] += np.mean(read_data[0, :])
                        else:
                            ext[test_nn] += read_data[0, args.input_dim]
            np.save(f'{ext_filename}', ext)
            print(f'Saving ext to {ext_filename}.')
        ll.append(ext)
        visualize_sensor_varying('METR_LA', -1, ext, filename=f'{res_dir}/significance_{index}.html', adjust=True)
        print(f'Finish visualize index {index}.')

    ext_filename = f'{res_dir}/ext.npy'
    if os.path.isfile(ext_filename):
        ext = np.load(ext_filename, allow_pickle=True).item()
    else:
        ext = {sensor: sum([ext_l[sensor] for ext_l in ll]) for sensor in range(sensors)}
        np.save(f'{ext_filename}', ext)
    visualize_sensor_varying('METR_LA', -1, ext, filename=f'{res_dir}/significance.html', adjust=True)
    print(f'Finish visualize.')
