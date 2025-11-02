from typing import Callable, Dict, List, Union, Tuple, Type
from torch import (Tensor, kron, is_grad_enabled, no_grad, zeros_like,preserve_format, ones_like, cat, einsum, sum, inf)
from torch.optim import Optimizer
import torch.distributed as dist
from torch.nn.functional import pad
from math import prod
from torch.nn import Module, Linear, Conv2d, BatchNorm2d, LayerNorm, Parameter


def smart_detect_inf(tensor: Tensor) -> Tensor:
    result_tensor = tensor.clone()
    result_tensor[tensor == inf] = 1.
    result_tensor[tensor == -inf] = 0.
    return result_tensor


def MinMaxNormalization(tensor: Tensor, epsilon: float = 1e-6) -> Tensor:
    tensor = smart_detect_inf(tensor)
    min_tensor = tensor.min()
    max_tensor = tensor.max()
    range_tensor = max_tensor - min_tensor
    return tensor.add_(-min_tensor).div_(range_tensor + epsilon)


def update_running_avg(new: Tensor, current: Tensor, gamma: float):
    current *= gamma * 1e-1
    current += (gamma * 1e-2) * new


def _extract_patches(x: Tensor, kernel_size: Tuple[int],
                     stride: Tuple[int],
                     padding: Tuple[int],
                     groups: int) -> Tensor:

    if padding[0] + padding[1] > 0:
        x = pad(x, (padding[1], padding[1], padding[0], padding[0]))
    batch_size, in_channels, height, width = x.size()
    x = x.view(batch_size, groups, in_channels // groups, height, width)
    x = x.unfold(3, kernel_size[0], stride[0])
    x = x.unfold(4, kernel_size[1], stride[1])
    x = x.permute(0, 1, 3, 4, 2, 5, 6).contiguous()
    x = x.view(batch_size, groups, -1, in_channels // groups * kernel_size[0] * kernel_size[1])
    x = x.view(batch_size, -1, x.size(2), x.size(3))
    return x


class Compute_H_D:
    @classmethod
    def compute_H_D(cls, h, layer) -> Tensor:
        return cls.__call__(h, layer)

    @classmethod
    def __call__(cls, h: Tensor, layer: Module) -> Tensor:
        if isinstance(layer, Linear):
            H_D = cls.linear(h, layer)
        elif isinstance(layer, Conv2d):
            H_D = cls.conv2d(h, layer)
        elif isinstance(layer, BatchNorm2d):
            H_D = cls.batchnorm2d(h, layer)
        elif isinstance(layer, LayerNorm):
            H_D = cls.layernorm(h, layer)
        else:
            raise NotImplementedError

        return H_D

    @staticmethod
    def conv2d(h: Tensor, layer: Conv2d) -> Tensor:
        batch_size = h.size(0)
        h = _extract_patches(h, layer.kernel_size, layer.stride, layer.padding, layer.groups)
        spatial_size = h.size(2) * h.size(3)
        h = h.reshape(-1, h.size(-1))
        if layer.bias is not None:
            h_bar = cat([h, h.new(h.size(0), 1).fill_(1)], 1)
        return einsum('ij,ij->j', h_bar, h_bar) / (batch_size * spatial_size) if layer.bias is not None \
            else einsum('ij,ij->j', h, h) / (batch_size * spatial_size)

    @staticmethod
    def linear(h: Tensor, layer: Linear) -> Tensor:
        if len(h.shape) > 2:
            h = h.reshape(-1, h.shape[-1])
        batch_size = h.size(0)
        if layer.bias is not None:
            h_bar = cat([h, h.new(h.size(0), 1).fill_(1)], 1)
        return einsum('ij,ij->j', h_bar, h_bar) / batch_size if layer.bias is not None \
            else einsum('ij,ij->j', h, h) / batch_size

    @staticmethod
    def batchnorm2d(h: Tensor, layer: BatchNorm2d) -> Tensor:
        batch_size, spatial_size = h.size(0), h.size(2) * h.size(3)
        sum_h = sum(h, dim=(0, 2, 3)).unsqueeze(1) / (spatial_size ** 2)
        h_bar = cat([sum_h, sum_h.new(sum_h.size(0), 1).fill_(1)], 1)
        return einsum('ij,ij->j', h_bar, h_bar) / (batch_size ** 2)

    @staticmethod
    def layernorm(h: Tensor, layer: LayerNorm) -> Tensor:
        dim_to_reduce = [d for d in range(h.ndim) if d != 1]
        batch_size, dim_norm = h.shape[0], prod([h.shape[dim] for dim in dim_to_reduce if dim != 0])
        sum_h = sum(h, dim=dim_to_reduce).unsqueeze(1) / (dim_norm ** 2)
        h_bar = cat([sum_h, sum_h.new(sum_h.size(0), 1).fill_(1)], 1)
        return einsum('ij,ij->j', h_bar, h_bar) / (batch_size ** 2)


class Compute_S_D:
    @classmethod
    def compute_S_D(cls, s: Tensor, layer: Module) -> Tensor:
        return cls.__call__(s, layer)

    @classmethod
    def __call__(cls, s: Tensor, layer: Module) -> Tensor:
        if isinstance(layer, Conv2d):
            S_D = cls.conv2d(s, layer)
        elif isinstance(layer, Linear):
            S_D = cls.linear(s, layer)
        elif isinstance(layer, BatchNorm2d):
            S_D = cls.batchnorm2d(s, layer)
        elif isinstance(layer, LayerNorm):
            S_D = cls.layernorm(s, layer)
        else:
            raise NotImplementedError
        return S_D

    @staticmethod
    def conv2d(s: Tensor, layer: Conv2d) -> Tensor:
        batch_size = s.shape[0]
        spatial_size = s.size(2) * s.size(3)
        s = s.transpose(1, 2).transpose(2, 3)
        s = s.reshape(-1, s.size(-1))
        return einsum('ij,ij->j', s, s) / (batch_size * spatial_size)

    @staticmethod
    def linear(s: Tensor, layer: Linear) -> Tensor:
        if len(s.shape) > 2:
            s = s.reshape(-1, s.shape[-1])
        batch_size = s.size(0)
        return einsum('ij,ij->j', s, s) / batch_size

    @staticmethod
    def batchnorm2d(s: Tensor, layer: BatchNorm2d) -> Tensor:
        batch_size = s.size(0)
        sum_s = sum(s, dim=(0, 2, 3))
        return einsum('i,i->i', sum_s, sum_s) / batch_size

    @staticmethod
    def layernorm(s: Tensor, layer: LayerNorm) -> Tensor:
        batch_size = s.size(0)
        sum_s = sum(s, dim=tuple(range(s.ndim - 1)))
        return einsum('i,i->i', sum_s, sum_s) / batch_size

class AdaFisherBackBone(Optimizer):
    SUPPORTED_MODULES: Tuple[Type[str], ...] = ("Linear", "Conv2d", "BatchNorm2d", "LayerNorm")

    def __init__(self,
                 model: Module,
                 lr: float = 1e-3,
                 beta: float = 0.9,
                 Lambda: float = 1e-3,
                 gamma: float = 0.8,
                 TCov: int = 100,
                 weight_decay: float = 0,
                 dist: bool = False
                 ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= beta < 1.0:
            raise ValueError(f"Invalid beta parameter: {beta}")
        if not 0.0 <= gamma < 1.0:
            raise ValueError(f"Invalid gamma parameter: {gamma}")
        if not TCov > 0:
            raise ValueError(f"Invalid TCov parameter: {TCov}")
        defaults = dict(lr=lr, beta=beta,
                        weight_decay=weight_decay)

        self.gamma = gamma
        self.Lambda = Lambda
        self.model = model
        self.TCov = TCov
        self.dist = dist
        self.steps = 0

        self.H_D: Dict[Module, Tensor] = {}
        self.S_D: Dict[Module, Tensor] = {}
        self.modules: List[Module] = []
        self.Compute_H_D = Compute_H_D()
        self.Compute_S_D = Compute_S_D()
        self._prepare_model()
        super(AdaFisherBackBone, self).__init__(model.parameters(), defaults)

    def _save_input(self, module: Module, input: Tensor, output: Tensor):
        if is_grad_enabled() and self.steps % self.TCov == 0:
            H_D_i = self.Compute_H_D(input[0].data, module)
            if self.steps == 0:
                self.H_D[module] = H_D_i.new(H_D_i.size(0)).fill_(1)
            update_running_avg(MinMaxNormalization(H_D_i), self.H_D[module], self.gamma)
            

    def _save_grad_output(self, module: Module, grad_input: Tensor, grad_output: Tensor):
        if self.steps % self.TCov == 0:
            S_D_i = self.Compute_S_D(grad_output[0].data, module)
            if self.steps == 0:
                self.S_D[module] = S_D_i.new(S_D_i.size(0)).fill_(1)
            update_running_avg(MinMaxNormalization(S_D_i), self.S_D[module], self.gamma)

    def _prepare_model(self):
        for module in self.model.modules():
            classname = module.__class__.__name__
            if classname in self.SUPPORTED_MODULES:
                self.modules.append(module)
                module.register_forward_hook(self._save_input)
                module.register_full_backward_hook(self._save_grad_output)
                
    def state_dict(self):
        state_dict = super().state_dict()
        state_dict['H_D'] = [self.H_D[module] for module in self.modules]
        state_dict['S_D'] = [self.S_D[module] for module in self.modules]
        state_dict['steps'] = self.steps
        return state_dict

    def load_state_dict(self, state_dict):
        H_D_list = state_dict.pop('H_D', [])
        S_D_list = state_dict.pop('S_D', [])
        self.steps = state_dict.pop('steps', 0)
        super().load_state_dict(state_dict)
        self.H_D = {}
        self.S_D = {}
        for module, h_bar, s in zip(self.modules, H_D_list, S_D_list):
            self.H_D[module] = h_bar
            self.S_D[module] = s
    
    def aggregate_kronecker_factors(self, module: Module):
        dist.all_reduce(self.H_D[module], op=dist.ReduceOp.SUM)
        dist.all_reduce(self.S_D[module], op=dist.ReduceOp.SUM)

        self.H_D[module] /= dist.get_world_size()
        self.S_D[module] /= dist.get_world_size()

    def _get_F_tilde(self, module: Module):
        if self.dist:
            self.aggregate_kronecker_factors(module = module)
        F_tilde = kron(self.H_D[module].unsqueeze(1), self.S_D[module].unsqueeze(0)).t() + self.Lambda
        if module.bias is not None:
            F_tilde = [F_tilde[:, :-1], F_tilde[:, -1:]]
            F_tilde[0] = F_tilde[0].view(*module.weight.grad.data.size())
            F_tilde[1] = F_tilde[1].view(*module.bias.grad.data.size())
            return F_tilde
        else:
            return F_tilde.reshape(module.weight.grad.data.size())

    def _check_dim(self, param: List[Parameter], idx_module: int, idx_param: int) -> bool:
        params = param[idx_param]
        module = self.modules[idx_module]
        param_size = params.data.size()
        return param_size == module.weight.data.size() or (module.bias is not None and param_size == module.bias.data.size())


class AdaFisher(AdaFisherBackBone):
    def __init__(self, 
                 model: Module,
                 lr: float = 1e-3,
                 beta: float = 0.9,
                 Lambda: float = 1e-3,
                 gamma: float = 0.8,
                 TCov: int = 100,
                 weight_decay: float = 0,
                 dist: bool = False
                ):
        super(AdaFisher, self).__init__(model,
                                        lr = lr,
                                        beta = beta,
                                        Lambda = Lambda,
                                        gamma = gamma,
                                        TCov = TCov,
                                        dist = dist,
                                        weight_decay = weight_decay)
        
    @no_grad()
    def _step(self, hyperparameters: Dict[str, float], param: Parameter, F_tilde: Tensor):
        grad = param.grad
        state = self.state[param]
        if len(state) == 0:
            state['step'] = 0
            state['exp_avg'] = zeros_like(
                param, memory_format=preserve_format)
        exp_avg = state['exp_avg']
        beta = hyperparameters['beta']
        state['step'] += 1
        bias_correction = 1 - beta ** state['step']
        if hyperparameters['weight_decay'] != 0:
            grad = grad.add(param, alpha=hyperparameters['weight_decay'])
        exp_avg.mul_(beta).add_(grad, alpha= 1 - beta)
        step_size = hyperparameters['lr'] / bias_correction
        param.addcdiv_(exp_avg, F_tilde, value=-step_size)

    @no_grad()
    def step(self, closure: Union[None, Callable[[], Tensor]] = None):
        if closure is not None:
            raise NotImplementedError("Closure not supported.")
        for group in self.param_groups:
            idx_param, idx_module, buffer_count = 0, 0, 0
            param, hyperparameters = group['params'], {"weight_decay": group['weight_decay'], "beta": group['beta'], "lr": group['lr']}
            for _ in range(len(self.modules)):
                if param[idx_param].grad is None:
                    idx_param += 1
                    if param[idx_param].ndim > 1:
                        idx_module += 1
                    else:
                        buffer_count += 1
                    if buffer_count == 2:
                        idx_module += 1
                        buffer_count = 0
                    continue
                m = self.modules[idx_module]
                if self._check_dim(param, idx_module, idx_param):
                    F_tilde = self._get_F_tilde(m)
                    idx_module += 1
                else:
                    F_tilde = ones_like(param[idx_param]) 
                if isinstance(F_tilde, list):
                    for F_tilde_i in F_tilde:
                        self._step(hyperparameters, param[idx_param], F_tilde_i)
                        idx_param += 1
                else:
                    self._step(hyperparameters, param[idx_param], F_tilde)
                    idx_param += 1
        self.steps += 1


class AdaFisherW(AdaFisherBackBone):
    def __init__(self, 
                 model: Module,
                 lr: float = 1e-3,
                 beta: float = 0.9,
                 Lambda: float = 1e-3,
                 gamma: float = 0.8,
                 TCov: int = 100,
                 weight_decay: float = 0,
                 dist: bool = False
                ):
        super(AdaFisherW, self).__init__(model,
                                        lr = lr,
                                        beta = beta,
                                        Lambda = Lambda,
                                        gamma = gamma,
                                        TCov = TCov,
                                        dist = dist,
                                        weight_decay = weight_decay)
    
    @no_grad()
    def _step(self, hyperparameters: Dict[str, float], param: Parameter, F_tilde: Tensor):
        grad = param.grad
        state = self.state[param]
        if len(state) == 0:
            state['step'] = 0
            state['exp_avg'] = zeros_like(
                param, memory_format=preserve_format)
        exp_avg = state['exp_avg']
        beta = hyperparameters['beta']
        state['step'] += 1
        bias_correction = 1 - beta ** state['step']
        exp_avg.mul_(beta).add_(grad, alpha= 1 - beta)
        param.data -= hyperparameters['lr'] * (exp_avg / bias_correction / F_tilde + hyperparameters['weight_decay'] * param.data)


    @no_grad()
    def step(self, closure: Union[None, Callable[[], Tensor]] = None):
        if closure is not None:
            raise NotImplementedError("Closure not supported.")
        for group in self.param_groups:
            idx_param, idx_module, buffer_count = 0, 0, 0
            param, hyperparameters = group['params'], {"weight_decay": group['weight_decay'], "beta": group['beta'], "lr": group['lr']}
            for _ in range(len(self.modules)):
                if param[idx_param].grad is None:
                    idx_param += 1
                    if param[idx_param].ndim > 1:
                        idx_module += 1
                    else:
                        buffer_count += 1
                    if buffer_count == 2:
                        idx_module += 1
                        buffer_count = 0
                    continue
                m = self.modules[idx_module]
                if self._check_dim(param, idx_module, idx_param):
                    F_tilde = self._get_F_tilde(m)
                    idx_module += 1
                else:
                    F_tilde = ones_like(param[idx_param])
                if isinstance(F_tilde, list):
                    for F_tilde_i in F_tilde:
                        self._step(hyperparameters, param[idx_param], F_tilde_i)
                        idx_param += 1
                else:
                    self._step(hyperparameters, param[idx_param], F_tilde)
                    idx_param += 1
        self.steps += 1