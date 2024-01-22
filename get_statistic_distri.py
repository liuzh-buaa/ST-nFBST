"""
    python get_statistic_distri.py --task traffic_state_pred --model BDCRNNVariableDecoder --dataset METR_LA --model_cache_id 77290
    python get_statistic_distri.py --task traffic_state_pred --model BDCRNNVariableDecoderShared --dataset PEMS_BAY --model_cache_id 42464
"""
import argparse
import os
import random

from libcity.config import ConfigParser
from libcity.data import get_dataset
from libcity.utils import str2bool, add_general_args, get_logger, set_random_seed, get_executor, get_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 增加指定的参数
    parser.add_argument('--task', type=str,
                        default='traffic_state_pred', help='the name of task')
    parser.add_argument('--model', type=str,
                        default='BDCRNNVariableDecoder', help='the name of model')
    parser.add_argument('--dataset', type=str,
                        default='METR_LA', help='the name of dataset')
    parser.add_argument('--config_file', type=str,
                        default=None, help='the file name of config file')
    parser.add_argument('--saved_model', type=str2bool,
                        default=False, help='whether save the trained model')
    parser.add_argument('--train', type=str2bool, default=False,
                        help='whether re-train model if the model is trained before')
    parser.add_argument('--exp_id', type=str, default=None, help='id of experiment')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--model_cache_id', type=str, required=True,
                        help='load model config from {model_cache_id}')
    parser.add_argument('--start', type=int, default=0,
                        help='begin testing at [start_batch]')
    parser.add_argument('--end', type=int, default=10,
                        help='testing [num_batch] batches')
    parser.add_argument('--testing_samples', type=int, default=50,
                        help='nums of samples for testing')
    # 增加其他可选的参数
    add_general_args(parser)
    # 解析参数
    args = parser.parse_args()
    args.shuffle = False
    args.batch_size = 1
    dict_args = vars(args)
    other_args = {key: val for key, val in dict_args.items() if key not in [
        'task', 'model', 'dataset', 'config_file', 'saved_model', 'train'] and
                  val is not None}

    # load config
    config = ConfigParser(args.task, args.model, args.dataset,
                          args.config_file, args.saved_model, args.train, other_args)
    exp_id = config.get('exp_id', None)
    if exp_id is None:
        # Make a new experiment ID
        exp_id = int(random.SystemRandom().random() * 100000)
        config['exp_id'] = exp_id
    # logger
    logger = get_logger(config)
    logger.info('Begin pipeline, task={}, model_name={}, dataset_name={}, exp_id={}'.
                format(str(args.task), str(args.model), str(args.dataset), str(exp_id)))
    logger.info(config.config)
    # seed
    seed = config.get('seed', 0)
    set_random_seed(seed)
    # 加载数据集
    dataset = get_dataset(config)
    # 转换数据，并划分数据集
    train_data, valid_data, test_data = dataset.get_data()
    data_feature = dataset.get_data_feature()
    # 加载执行器
    assert config.get('model_cache_id') is not None
    model_cache_file = './libcity/cache/{}/model_cache/{}_{}.m'.format(
        config.get('model_cache_id'), args.model, args.dataset)
    model = get_model(config, data_feature)
    executor = get_executor(config, model, data_feature)
    assert os.path.exists(model_cache_file)
    executor.load_model(model_cache_file)
    # 评估，评估结果将会放在 cache/evaluate_cache 下
    # executor.evaluate(test_data)
    executor.testing(test_data, args.start, args.end, config.get('output_window'), config.get('output_dim'),
                     args.testing_samples)
