import os
import time
from functools import partial

import numpy as np
import torch
from ray import tune

from libcity.executor.traffic_state_executor import TrafficStateExecutor
from libcity.model import loss
from libcity.utils import ensure_dir
from libcity.utils.stats import kde_bayes_factor


class BDCRNNExecutor(TrafficStateExecutor):
    def __init__(self, config, model, data_feature):
        TrafficStateExecutor.__init__(self, config, model, data_feature)
        self.testing_res_dir = './libcity/cache/{}/testing_cache'.format(self.exp_id)
        ensure_dir(self.testing_res_dir)

    def _build_train_loss(self):
        """
        根据全局参数`train_loss`选择训练过程的loss函数
        如果该参数为none，则需要使用模型自定义的loss函数
        注意，loss函数应该接收`Batch`对象作为输入，返回对应的loss(torch.tensor)
        """
        if self.train_loss.lower() == 'none':
            self._logger.warning('Received none train loss func and will use the loss func defined in the model.')
            return None
        raise Exception('bdcrnn loss_fun should be defined in class itself.')
        if self.train_loss.lower() not in ['mae', 'mse', 'rmse', 'mape', 'logcosh', 'huber', 'quantile', 'masked_mae',
                                           'masked_mse', 'masked_rmse', 'masked_mape', 'r2', 'evar']:
            self._logger.warning('Received unrecognized train loss function, set default mae loss func.')
        else:
            self._logger.info('You select `{}` as train loss function.'.format(self.train_loss.lower()))

        def func(batch, batches_seen=None):
            y_true = batch['y']
            y_predicted = self.model.predict(batch, batches_seen)
            y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
            y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])
            if self.train_loss.lower() == 'mae':
                lf = loss.masked_mae_torch
            elif self.train_loss.lower() == 'mse':
                lf = loss.masked_mse_torch
            elif self.train_loss.lower() == 'rmse':
                lf = loss.masked_rmse_torch
            elif self.train_loss.lower() == 'mape':
                lf = loss.masked_mape_torch
            elif self.train_loss.lower() == 'logcosh':
                lf = loss.log_cosh_loss
            elif self.train_loss.lower() == 'huber':
                lf = loss.huber_loss
            elif self.train_loss.lower() == 'quantile':
                lf = loss.quantile_loss
            elif self.train_loss.lower() == 'masked_mae':
                lf = partial(loss.masked_mae_torch, null_val=0)
            elif self.train_loss.lower() == 'masked_mse':
                lf = partial(loss.masked_mse_torch, null_val=0)
            elif self.train_loss.lower() == 'masked_rmse':
                lf = partial(loss.masked_rmse_torch, null_val=0)
            elif self.train_loss.lower() == 'masked_mape':
                lf = partial(loss.masked_mape_torch, null_val=0)
            elif self.train_loss.lower() == 'r2':
                lf = loss.r2_score_torch
            elif self.train_loss.lower() == 'evar':
                lf = loss.explained_variance_score_torch
            else:
                lf = loss.masked_mae_torch
            return lf(y_predicted, y_true)

        return func

    def evaluate(self, test_dataloader):
        """
        use model to test data

        Args:
            test_dataloader(torch.Dataloader): Dataloader
        """
        self._logger.info('Start evaluating ...')
        with torch.no_grad():
            self.model.eval()
            # self.evaluator.clear()
            y_truths, y_preds, outputs, sigmas = [], [], [], []
            for batch in test_dataloader:
                batch.to_tensor(self.device)
                output = torch.stack([self.model.predict(batch).detach().clone() for _ in range(self.evaluate_rep)])
                sigma = torch.stack(
                    [self.model.predict_sigma(batch).detach().clone() for _ in range(self.evaluate_rep)])
                y_pred = torch.mean(output, dim=0)
                y_true = self._scaler.inverse_transform(batch['y'][..., :self.output_dim])
                y_pred = self._scaler.inverse_transform(y_pred[..., :self.output_dim])
                output = self._scaler.inverse_transform(output[..., :self.output_dim])
                y_truths.append(y_true.cpu().numpy())
                y_preds.append(y_pred.cpu().numpy())
                outputs.append(output.cpu().numpy())
                sigmas.append(sigma.cpu().numpy())
                # evaluate_input = {'y_true': y_true, 'y_pred': y_pred}
                # self.evaluator.collect(evaluate_input)
            # self.evaluator.save_result(self.evaluate_res_dir)
            y_preds = np.concatenate(y_preds, axis=0)
            y_truths = np.concatenate(y_truths, axis=0)
            outputs = np.concatenate(outputs, axis=1)  # concatenate on batch
            sigmas = np.concatenate(sigmas, axis=1)
            res = {'prediction': y_preds, 'truth': y_truths, 'outputs': outputs, 'sigmas': sigmas}
            filename = \
                time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime(time.time())) + '_' \
                + self.config['model'] + '_' + self.config['dataset'] + '_predictions.npz'
            np.savez_compressed(os.path.join(self.evaluate_res_dir, filename), **res)
            self.evaluator.clear()
            self.evaluator.collect({'y_true': torch.tensor(y_truths), 'y_pred': torch.tensor(y_preds)})
            test_result = self.evaluator.save_result(self.evaluate_res_dir)
            return test_result

    def testing(self, test_dataloader, start, end, output_window, output_dim, testing_samples):
        """
        use model to test data

        Args:
            test_dataloader(torch.Dataloader): Dataloader
        """
        self._logger.info('Start hypothesis testing ...')
        for i, batch in enumerate(test_dataloader):
            if i >= end:
                break
            if i < start:
                continue
            batch.to_tensor(self.device)
            batch_size, input_window, num_nodes, input_dim = batch['X'].shape
            y_true = self._scaler.inverse_transform(batch['y'][..., :self.output_dim])
            assert batch_size == 1 and (batch_size, output_window, num_nodes, output_dim) == y_true.shape
            for ow in [2, 5, 11]:
                for nn in range(num_nodes):
                    for od in range(output_dim):
                        self._logger.info('Start hypothesis testing {}={}: ow={}, nn={}, od={}'.format(
                            i, y_true[0, ow, nn, od], ow, nn, od))
                        samples = torch.stack([self.model.get_interpret(batch['X'], ow, nn, od)
                                               for _ in range(testing_samples)]).cpu().numpy()
                        self._logger.info('Finish getting samples of {}'.format(samples.shape))
                        filename = 'gradient_samples_{}_{}_{}_{}.npy'.format(i, ow, nn, od)
                        np.save(os.path.join(self.testing_res_dir, filename), samples)
                        # testing_results = np.apply_along_axis(
                        #     lambda x: kde_bayes_factor(x), axis=0, arr=samples.reshape(testing_samples, -1)).reshape(
                        #     2, input_window, num_nodes, input_dim)
                        # filename = 'ps_testing_{}_{}_{}_{}.npy'.format(i, ow, nn, od)
                        # np.save(os.path.join(self.testing_res_dir, filename), testing_results[0])
                        # filename = 'kde_bandwidth_{}_{}_{}_{}.npy'.format(i, ow, nn, od)
                        # np.save(os.path.join(self.testing_res_dir, filename), testing_results[1])
                        for test_nn in range(num_nodes):
                            if test_nn == nn:
                                testing_results = np.apply_along_axis(
                                    lambda x: kde_bayes_factor(x), axis=0,
                                    arr=samples[..., test_nn, :].reshape(testing_samples, -1)).reshape(
                                    2, input_window, input_dim)  # only test current node for all input_window
                            else:
                                testing_results = np.apply_along_axis(
                                    lambda x: kde_bayes_factor(x), axis=0,
                                    arr=samples[..., input_window - 1, test_nn, :].reshape(testing_samples,
                                                                                           -1)).reshape(
                                    2, 1, input_dim)  # only test other nodes for the nearest input_window
                            filename = 'ps_testing_{}_{}_{}_{}_{}.npy'.format(i, ow, nn, od, test_nn)
                            np.save(os.path.join(self.testing_res_dir, filename), testing_results[0])
                            filename = 'kde_bandwidth_{}_{}_{}_{}_{}.npy'.format(i, ow, nn, od, test_nn)
                            np.save(os.path.join(self.testing_res_dir, filename), testing_results[1])
        self._logger.info('Finish hypothesis testing ...')

    def testing2(self, test_dataloader, start, end, output_window, output_dim, testing_samples):
        """
        use model to test data

        Args:
            test_dataloader(torch.Dataloader): Dataloader
        """
        self._logger.info('Start hypothesis testing ...')
        for i, batch in enumerate(test_dataloader):
            if i >= end:
                break
            if i < start:
                continue
            batch.to_tensor(self.device)
            batch_size, input_window, num_nodes, input_dim = batch['X'].shape
            y_true = self._scaler.inverse_transform(batch['y'][..., :self.output_dim])
            assert batch_size == 1 and (batch_size, output_window, num_nodes, output_dim) == y_true.shape
            for ow in [2]:
                for nn in range(num_nodes):
                    for od in range(output_dim):
                        self._logger.info('Start hypothesis testing {}={}: ow={}, nn={}, od={}'.format(
                            i, y_true[0, ow, nn, od], ow, nn, od))
                        samples = torch.stack([self.model.get_interpret(batch['X'], ow, nn, od)
                                               for _ in range(testing_samples)]).cpu().numpy()
                        self._logger.info('Finish getting samples of {}'.format(samples.shape))
                        filename = 'gradient_samples_{}_{}_{}_{}.npy'.format(i, ow, nn, od)
                        np.save(os.path.join(self.testing_res_dir, filename), samples)
                        # testing_results = np.apply_along_axis(
                        #     lambda x: kde_bayes_factor(x), axis=0, arr=samples.reshape(testing_samples, -1)).reshape(
                        #     2, input_window, num_nodes, input_dim)
                        # filename = 'ps_testing_{}_{}_{}_{}.npy'.format(i, ow, nn, od)
                        # np.save(os.path.join(self.testing_res_dir, filename), testing_results[0])
                        # filename = 'kde_bandwidth_{}_{}_{}_{}.npy'.format(i, ow, nn, od)
                        # np.save(os.path.join(self.testing_res_dir, filename), testing_results[1])
                        samples = np.mean(samples, axis=-1, keepdims=True)
                        for test_nn in range(num_nodes):
                            if test_nn == nn:
                                testing_results = np.apply_along_axis(
                                    lambda x: kde_bayes_factor(x), axis=0,
                                    arr=samples[..., test_nn, :].reshape(testing_samples, -1)).reshape(
                                    2, input_window, 1)  # only test current node for all input_window
                            else:
                                testing_results = np.apply_along_axis(
                                    lambda x: kde_bayes_factor(x), axis=0,
                                    arr=samples[..., input_window - 1, test_nn, :].reshape(testing_samples,
                                                                                           -1)).reshape(2, 1, 1)  # only test other nodes for the nearest input_window
                            filename = 'ps_testing_{}_{}_{}_{}_{}.npy'.format(i, ow, nn, od, test_nn)
                            np.save(os.path.join(self.testing_res_dir, filename), testing_results[0])
                            # filename = 'kde_bandwidth_{}_{}_{}_{}_{}.npy'.format(i, ow, nn, od, test_nn)
                            # np.save(os.path.join(self.testing_res_dir, filename), testing_results[1])
        self._logger.info('Finish hypothesis testing ...')

    def train(self, train_dataloader, eval_dataloader):
        """
        use data to train model with config

        Args:
            train_dataloader(torch.Dataloader): Dataloader
            eval_dataloader(torch.Dataloader): Dataloader
        """
        self._logger.info('Start training ...')
        min_val_loss = float('inf')
        wait = 0
        best_epoch = 0
        train_time = []
        eval_time = []
        num_batches = len(train_dataloader)
        self._logger.info("num_batches:{}".format(num_batches))

        batches_seen = num_batches * self._epoch_num
        for epoch_idx in range(self._epoch_num, self.epochs):
            start_time = time.time()
            losses, batches_seen = self._train_epoch(train_dataloader, epoch_idx, batches_seen, self.loss_func,
                                                     num_batches)
            t1 = time.time()
            train_time.append(t1 - start_time)
            self._writer.add_scalar('training loss', np.mean(losses), batches_seen)
            self._logger.info("epoch complete!")

            self._logger.info("evaluating now!")
            t2 = time.time()
            val_loss = self._valid_epoch(eval_dataloader, epoch_idx, batches_seen, self.loss_func)
            end_time = time.time()
            eval_time.append(end_time - t2)
            val_train_loss = self._valid_epoch(train_dataloader, epoch_idx, batches_seen, self.loss_func)

            if self.lr_scheduler is not None:
                if self.lr_scheduler_type.lower() == 'reducelronplateau':
                    self.lr_scheduler.step(val_loss)
                else:
                    self.lr_scheduler.step()

            if (epoch_idx % self.log_every) == 0:
                log_lr = self.optimizer.param_groups[0]['lr']
                message = 'Epoch [{}/{}] ({}) train_loss: {:.4f} ({:.4f}), val_loss: {:.4f}, lr: {:.6f}, {:.2f}s'. \
                    format(epoch_idx, self.epochs, batches_seen, np.mean(losses), val_train_loss, val_loss,
                           log_lr, (end_time - start_time))
                self._logger.info(message)

            if self.hyper_tune:
                # use ray tune to checkpoint
                with tune.checkpoint_dir(step=epoch_idx) as checkpoint_dir:
                    path = os.path.join(checkpoint_dir, "checkpoint")
                    self.save_model(path)
                # ray tune use loss to determine which params are best
                tune.report(loss=val_loss)

            if val_loss < min_val_loss:
                wait = 0
                if self.saved:
                    model_file_name = self.save_model_with_epoch(epoch_idx)
                    self._logger.info('Val loss decrease from {:.4f} to {:.4f}, '
                                      'saving to {}'.format(min_val_loss, val_loss, model_file_name))
                min_val_loss = val_loss
                best_epoch = epoch_idx
            else:
                wait += 1
                if wait == self.patience and self.use_early_stop:
                    self._logger.warning('Early stopping at epoch: %d' % epoch_idx)
                    break
        if len(train_time) > 0:
            self._logger.info('Trained totally {} epochs, average train time is {:.3f}s, '
                              'average eval time is {:.3f}s'.
                              format(len(train_time), sum(train_time) / len(train_time),
                                     sum(eval_time) / len(eval_time)))
        if self.load_best_epoch:
            self.load_model_with_epoch(best_epoch)
        return min_val_loss

    def _train_epoch(self, train_dataloader, epoch_idx, batches_seen=None, loss_func=None, num_batches=1):
        """
        完成模型一个轮次的训练

        Args:
            train_dataloader: 训练数据
            epoch_idx: 轮次数
            batches_seen: 全局batch数
            loss_func: 损失函数

        Returns:
            tuple: tuple contains
                losses(list): 每个batch的损失的数组 \n
                batches_seen(int): 全局batch数
        """
        self.model.train()
        loss_func = loss_func if loss_func is not None else self.model.calculate_loss
        losses = []
        for batch in train_dataloader:
            self.optimizer.zero_grad()
            batch.to_tensor(self.device)
            loss = loss_func(batch, batches_seen, num_batches)
            self._logger.debug(loss.item())
            losses.append(loss.item())
            batches_seen += 1
            loss.backward()
            if self.clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()
        return losses, batches_seen

    def _valid_epoch(self, eval_dataloader, epoch_idx, batches_seen=None, loss_func=None):
        """
        完成模型一个轮次的评估

        Args:
            eval_dataloader: 评估数据
            epoch_idx: 轮次数
            batches_seen: 全局batch数
            loss_func: 损失函数

        Returns:
            float: 评估数据的平均损失值
        """
        with torch.no_grad():
            self.model.eval()
            loss_func = loss_func if loss_func is not None else self.model.calculate_eval_loss
            losses = []
            for batch in eval_dataloader:
                batch.to_tensor(self.device)
                loss = loss_func(batch, batches_seen)
                self._logger.debug(loss.item())
                losses.append(loss.item())
            mean_loss = np.mean(losses)
            self._writer.add_scalar('eval loss', mean_loss, batches_seen)
            return mean_loss
