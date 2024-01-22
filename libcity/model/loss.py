import math

import numpy as np
import torch
from sklearn.metrics import r2_score, explained_variance_score


def consistent_loss(sigma_0):
    ret = 0
    for batch_sigma_0 in torch.unbind(sigma_0, 2):  # (batch_size, output_window, num_nodes, output_dim)
        sequence_sigma_0 = torch.unbind(batch_sigma_0, 0)
        for i in range(len(sequence_sigma_0) - 1):
            ret += torch.abs(torch.sum(sequence_sigma_0[i][1:] - sequence_sigma_0[i + 1][:-1]))
    return ret


def masked_mae_loss(y_pred, y_true):
    mask = (y_true != 0).float()
    mask /= mask.mean()
    loss = torch.abs(y_pred - y_true)
    loss = loss * mask
    # trick for nans:
    # https://discuss.pytorch.org/t/how-to-set-nan-in-tensor-to-0/3918/3
    loss[loss != loss] = 0
    return loss.mean()


def masked_uncertainty_torch(uncertainty, labels, null_val=np.nan):
    labels[torch.abs(labels) < 1e-4] = 0
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = labels.ne(null_val)
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(uncertainty)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_mae_torch(preds, labels, null_val=np.nan):
    labels[torch.abs(labels) < 1e-4] = 0
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = labels.ne(null_val)
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(torch.sub(preds, labels))
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_mae_const_reg_torch(preds, labels, sigma_0, reg, null_val=np.nan):
    labels[torch.abs(labels) < 1e-4] = 0
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = labels.ne(null_val)
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)

    log_sigma_0 = math.log(sigma_0)

    loss = torch.abs(torch.sub(preds, labels)) / sigma_0
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)

    log_sigma_0 = log_sigma_0 * mask
    log_sigma_0 = torch.where(mask == 0, torch.zeros_like(log_sigma_0), log_sigma_0)

    valid_size = torch.sum(torch.where(mask == 0, torch.zeros_like(labels), torch.ones_like(labels)))

    return torch.mean(loss) + torch.mean(log_sigma_0) + reg / valid_size


def masked_mae_relu_reg_torch(preds, labels, sigma_0, reg, null_val=np.nan, custom_relu_eps=0.0, switch_consistent=False):
    labels[torch.abs(labels) < 1e-4] = 0
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = labels.ne(null_val)
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)

    class MyReLU(torch.autograd.Function):

        @staticmethod  # 静态方法，无需添加 self 参数
        def forward(ctx, input_data):
            """
            前向传播，通过 ReLU 函数处理输入数据并输出
            """
            ctx.save_for_backward(input_data)
            output = torch.clamp(input_data, min=custom_relu_eps)  # 使用 clamp() 方法将输入数据裁剪到大于等于 1e-3
            return output  # 返回输出结果

        @staticmethod
        def backward(ctx, grad_output):
            """
            反向传播，计算 ReLU 函数的梯度
            """
            input_data, = ctx.saved_tensors
            grad_input = grad_output.clone()
            grad_input[input_data < custom_relu_eps] = 0  # 根据输出对输入数据进行裁剪和梯度计算
            return grad_input

    sigma_0 = MyReLU.apply(sigma_0)
    sigma_0 = sigma_0 * mask
    sigma_0 = torch.where(mask == 0, torch.ones_like(sigma_0), sigma_0)

    log_sigma_0 = torch.log(sigma_0)

    loss = torch.abs(torch.sub(preds, labels)) / sigma_0
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)

    # log_sigma_0 = log_sigma_0 * mask
    # log_sigma_0 = torch.where(mask == 0, torch.zeros_like(log_sigma_0), log_sigma_0)

    valid_size = torch.sum(torch.where(mask == 0, torch.zeros_like(labels), torch.ones_like(labels)))

    if switch_consistent:
        return torch.mean(loss) + torch.mean(log_sigma_0) + reg / valid_size + consistent_loss(sigma_0)

    return torch.mean(loss) + torch.mean(log_sigma_0) + reg / valid_size


def masked_mae_softplus_reg_torch(preds, labels, sigma_0, reg, null_val=np.nan, custom_softplus_beta=1, switch_consistent=False):
    labels[torch.abs(labels) < 1e-4] = 0
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = labels.ne(null_val)
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)

    sigma_0 = torch.nn.Softplus(beta=custom_softplus_beta)(sigma_0)
    sigma_0 = sigma_0 * mask
    sigma_0 = torch.where(mask == 0, torch.ones_like(sigma_0), sigma_0)

    log_sigma_0 = torch.log(sigma_0)

    loss = torch.abs(torch.sub(preds, labels)) / sigma_0
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)

    # log_sigma_0 = log_sigma_0 * mask
    # log_sigma_0 = torch.where(mask == 0, torch.zeros_like(log_sigma_0), log_sigma_0)

    valid_size = torch.sum(torch.where(mask == 0, torch.zeros_like(labels), torch.ones_like(labels)))

    if switch_consistent:
        return torch.mean(loss) + torch.mean(log_sigma_0) + reg / valid_size + consistent_loss(sigma_0)

    return torch.mean(loss) + torch.mean(log_sigma_0) + reg / valid_size


def masked_mae_log_reg_torch(preds, labels, log_sigma_0, reg, null_val=np.nan, switch_consistent=False):
    labels[torch.abs(labels) < 1e-4] = 0
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = labels.ne(null_val)
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)

    log_sigma_0 = log_sigma_0 * mask
    log_sigma_0 = torch.where(mask == 0, torch.zeros_like(log_sigma_0), log_sigma_0)

    if type(log_sigma_0) == torch.Tensor:
        sigma_0 = torch.exp(log_sigma_0)
    else:
        sigma_0 = math.exp(log_sigma_0)

    loss = torch.abs(torch.sub(preds, labels)) / sigma_0
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)

    valid_size = torch.sum(torch.where(mask == 0, torch.zeros_like(labels), torch.ones_like(labels)))

    if switch_consistent:
        return torch.mean(loss) + torch.mean(log_sigma_0) + reg / valid_size + consistent_loss(sigma_0)

    return torch.mean(loss) + torch.mean(log_sigma_0) + reg / valid_size


def log_cosh_loss(preds, labels):
    loss = torch.log(torch.cosh(preds - labels))
    return torch.mean(loss)


def huber_loss(preds, labels, delta=1.0):
    residual = torch.abs(preds - labels)
    condition = torch.le(residual, delta)
    small_res = 0.5 * torch.square(residual)
    large_res = delta * residual - 0.5 * delta * delta
    return torch.mean(torch.where(condition, small_res, large_res))
    # lo = torch.nn.SmoothL1Loss()
    # return lo(preds, labels)


def quantile_loss(preds, labels, delta=0.25):
    condition = torch.ge(labels, preds)
    large_res = delta * (labels - preds)
    small_res = (1 - delta) * (preds - labels)
    return torch.mean(torch.where(condition, large_res, small_res))


def masked_mape_torch(preds, labels, null_val=np.nan, eps=0):
    labels[torch.abs(labels) < 1e-4] = 0
    if np.isnan(null_val) and eps != 0:
        loss = torch.abs((preds - labels) / (labels + eps))
        return torch.mean(loss)
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = labels.ne(null_val)
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs((preds - labels) / labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_mse_torch(preds, labels, null_val=np.nan):
    labels[torch.abs(labels) < 1e-4] = 0
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = labels.ne(null_val)
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.square(torch.sub(preds, labels))
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_mse_const_reg_torch(preds, labels, sigma_0, reg, null_val=np.nan):
    labels[torch.abs(labels) < 1e-4] = 0
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = labels.ne(null_val)
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)

    log_sigma_0 = math.log(sigma_0)

    loss = torch.square(torch.sub(preds, labels)) / 2 / torch.mul(sigma_0, sigma_0)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)

    log_sigma_0 = log_sigma_0 * mask
    log_sigma_0 = torch.where(mask == 0, torch.zeros_like(log_sigma_0), log_sigma_0)

    valid_size = torch.sum(torch.where(mask == 0, torch.zeros_like(labels), torch.ones_like(labels)))

    return torch.mean(loss) + torch.mean(log_sigma_0) + reg / valid_size


def masked_mse_relu_reg_torch(preds, labels, sigma_0, reg, null_val=np.nan, custom_relu_eps=0.0, switch_consistent=False):
    labels[torch.abs(labels) < 1e-4] = 0
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = labels.ne(null_val)
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)

    class MyReLU(torch.autograd.Function):

        @staticmethod  # 静态方法，无需添加 self 参数
        def forward(ctx, input_data):
            """
            前向传播，通过 ReLU 函数处理输入数据并输出
            """
            ctx.save_for_backward(input_data)
            output = torch.clamp(input_data, min=custom_relu_eps)  # 使用 clamp() 方法将输入数据裁剪到大于等于 1e-3
            return output  # 返回输出结果

        @staticmethod
        def backward(ctx, grad_output):
            """
            反向传播，计算 ReLU 函数的梯度
            """
            input_data, = ctx.saved_tensors
            grad_input = grad_output.clone()
            grad_input[input_data < custom_relu_eps] = 0  # 根据输出对输入数据进行裁剪和梯度计算
            return grad_input

    sigma_0 = MyReLU.apply(sigma_0)
    sigma_0 = sigma_0 * mask
    sigma_0 = torch.where(mask == 0, torch.ones_like(sigma_0), sigma_0)

    log_sigma_0 = torch.log(sigma_0)

    loss = torch.square(torch.sub(preds, labels)) / 2 / torch.mul(sigma_0, sigma_0)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)

    # log_sigma_0 = log_sigma_0 * mask
    # log_sigma_0 = torch.where(mask == 0, torch.zeros_like(log_sigma_0), log_sigma_0)

    valid_size = torch.sum(torch.where(mask == 0, torch.zeros_like(labels), torch.ones_like(labels)))

    if switch_consistent:
        return torch.mean(loss) + torch.mean(log_sigma_0) + reg / valid_size + consistent_loss(sigma_0)

    return torch.mean(loss) + torch.mean(log_sigma_0) + reg / valid_size


def masked_mse_softplus_reg_torch(preds, labels, sigma_0, reg, null_val=np.nan, custom_softplus_beta=1, switch_consistent=False):
    labels[torch.abs(labels) < 1e-4] = 0
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = labels.ne(null_val)
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)

    sigma_0 = torch.nn.Softplus(custom_softplus_beta)(sigma_0)
    sigma_0 = sigma_0 * mask
    sigma_0 = torch.where(mask == 0, torch.ones_like(sigma_0), sigma_0)

    log_sigma_0 = torch.log(sigma_0)

    loss = torch.square(torch.sub(preds, labels)) / 2 / torch.mul(sigma_0, sigma_0)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)

    # log_sigma_0 = log_sigma_0 * mask
    # log_sigma_0 = torch.where(mask == 0, torch.zeros_like(log_sigma_0), log_sigma_0)

    valid_size = torch.sum(torch.where(mask == 0, torch.zeros_like(labels), torch.ones_like(labels)))

    if switch_consistent:
        return torch.mean(loss) + torch.mean(log_sigma_0) + reg / valid_size + consistent_loss(sigma_0)

    return torch.mean(loss) + torch.mean(log_sigma_0) + reg / valid_size


def masked_mse_log_reg_torch(preds, labels, log_sigma_0, reg, null_val=np.nan, switch_consistent=False):
    labels[torch.abs(labels) < 1e-4] = 0
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = labels.ne(null_val)
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)

    log_sigma_0 = log_sigma_0 * mask
    log_sigma_0 = torch.where(mask == 0, torch.zeros_like(log_sigma_0), log_sigma_0)

    if type(log_sigma_0) == torch.Tensor:
        sigma_0 = torch.exp(log_sigma_0)
    else:
        sigma_0 = math.exp(log_sigma_0)

    loss = torch.square(torch.sub(preds, labels)) / 2 / torch.mul(sigma_0, sigma_0)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)

    valid_size = torch.sum(torch.where(mask == 0, torch.zeros_like(labels), torch.ones_like(labels)))

    if switch_consistent:
        return torch.mean(loss) + torch.mean(log_sigma_0) + reg / valid_size + consistent_loss(sigma_0)

    return torch.mean(loss) + torch.mean(log_sigma_0) + reg / valid_size


def masked_rmse_torch(preds, labels, null_val=np.nan):
    labels[torch.abs(labels) < 1e-4] = 0
    return torch.sqrt(masked_mse_torch(preds=preds, labels=labels,
                                       null_val=null_val))


def r2_score_torch(preds, labels):
    preds = preds.cpu().flatten()
    labels = labels.cpu().flatten()
    return r2_score(labels, preds)


def explained_variance_score_torch(preds, labels):
    preds = preds.cpu().flatten()
    labels = labels.cpu().flatten()
    return explained_variance_score(labels, preds)


def masked_rmse_np(preds, labels, null_val=np.nan):
    return np.sqrt(masked_mse_np(preds=preds, labels=labels,
                                 null_val=null_val))


def masked_mse_np(preds, labels, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(labels)
        else:
            mask = np.not_equal(labels, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        rmse = np.square(np.subtract(preds, labels)).astype('float32')
        rmse = np.nan_to_num(rmse * mask)
        return np.mean(rmse)


def masked_mae_np(preds, labels, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(labels)
        else:
            mask = np.not_equal(labels, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mae = np.abs(np.subtract(preds, labels)).astype('float32')
        mae = np.nan_to_num(mae * mask)
        return np.mean(mae)


def masked_mape_np(preds, labels, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(labels)
        else:
            mask = np.not_equal(labels, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mape = np.abs(np.divide(np.subtract(
            preds, labels).astype('float32'), labels))
        mape = np.nan_to_num(mask * mape)
        return np.mean(mape)


def r2_score_np(preds, labels):
    preds = preds.flatten()
    labels = labels.flatten()
    return r2_score(labels, preds)


def explained_variance_score_np(preds, labels):
    preds = preds.flatten()
    labels = labels.flatten()
    return explained_variance_score(labels, preds)
