import torch


class Interpreter(object):
    def __init__(self, model):
        self.model = model
        self.handles = []

    # (batch_size, input_window, num_nodes, input_dim)->(batch_size, input_window, num_nodes, input_dim)
    def interpret(self, x, output_window, num_nodes, output_dim):
        self.model.eval()
        with torch.enable_grad():
            self.model.zero_grad()
            model_output = self.model.predict_without_y(x)
            model_output_sum = torch.sum(model_output[:, output_window, num_nodes, output_dim])
            model_output_sum.backward()
            return x.grad.data.clone().detach()

    # (batch_size, input_window, num_nodes, input_dim)->(batch_size, input_window, num_nodes, input_dim)
    def interpret_sigma(self, x, output_window, num_nodes, output_dim):
        self.model.eval()
        with torch.enable_grad():
            self.model.zero_grad()
            model_output = self.model.predict_sigma_without_y(x)
            model_output_sum = torch.sum(model_output[:, output_window, num_nodes, output_dim])
            model_output_sum.backward()
            return x.grad.data.clone().detach()

    def release(self):
        """
            释放hook和内存，每次计算saliency后都要调用release()
        """
        for handle in self.handles:
            handle.remove()
        for p in self.model.parameters():
            del p.grad
            p.grad = None
        self.handles = []
