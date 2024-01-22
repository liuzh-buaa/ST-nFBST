import torch


def convert_dcrnn_to_bdcrnn(model, device, exp_id=91971, model_name='DCRNN', dataset_name='METR_LA'):
    """
    encoder_model.dcgru_layers.0._fn.weight/biases
    encoder_model.dcgru_layers.0._gconv.weight/biases
    encoder_model.dcgru_layers.1._fn.weight/biases
    encoder_model.dcgru_layers.1._gconv.weight/biases
    decoder_model.projection_layer.weight/bias
    decoder_model.dcgru_layers.0._fn.weight/biases
    decoder_model.dcgru_layers.0._gconv.weight/biases
    decoder_model.dcgru_layers.1._fn.weight/biases
    decoder_model.dcgru_layers.1._gconv.weight/biases
    """
    cache_file = './libcity/cache/{}/model_cache/{}_{}.m'.format(exp_id, model_name, dataset_name)
    model_state, _ = torch.load(cache_file, map_location=device)
    b_model_state = {}
    for k, v in model_state.items():
        splits = k.rsplit('.', 1)
        layer_name, param_name = splits[0], splits[1]
        b_model_state[layer_name + '.mu_' + param_name] = v
    model.load_state_dict(b_model_state, strict=False)


# if __name__ == '__main__':
#     convert_dcrnn_to_bdcrnn(None, 'cuda:2')
