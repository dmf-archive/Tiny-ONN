import torch
from torch.optim.optimizer import Optimizer


class F3EO(Optimizer):
    def __init__(self,
                 params,
                 lr=1e-3,
                 betas=(0.9, 0.999),
                 eps=1e-8,
                 weight_decay=0,
                 amsgrad=False,
                 maximize=False,
                 single_gpu=True,
                 orthogonalize=False):
        if not lr >= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not eps >= 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not weight_decay >= 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad, maximize=maximize, orthogonalize=orthogonalize)

        self.single_gpu = single_gpu
        self.grad_norm = 0.0
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        params_with_grad = []
        grads = []
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    if p.grad.is_sparse:
                        raise RuntimeError('F3EO does not support sparse gradients.')
                    if p.grad.grad_fn is None:
                        raise RuntimeError('Gradient tensor does not have grad_fn. When calling loss.backward(), make sure the option create_graph is set to True.')
                    params_with_grad.append(p)
                    grads.append(p.grad)

        if not grads:
            return loss

        grad_norm_sq = sum(g.pow(2).sum() for g in grads)
        self.grad_norm = grad_norm_sq.item()


        meta_grads = torch.autograd.grad(grad_norm_sq, params_with_grad, retain_graph=False, allow_unused=True)

        with torch.no_grad():
            param_idx = 0
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue

                    meta_grad = meta_grads[param_idx]
                    first_grad = grads[param_idx]
                    param_idx += 1

                    if meta_grad is None:
                        effective_grad = first_grad
                    else:
                        if group['orthogonalize']:
                            first_grad_dot = torch.dot(first_grad.reshape(-1), first_grad.reshape(-1))
                            if first_grad_dot > 1e-8:
                                projection_scale = torch.dot(meta_grad.reshape(-1), first_grad.reshape(-1)) / first_grad_dot
                                meta_grad = meta_grad - projection_scale * first_grad

                        effective_grad = first_grad + meta_grad


                    state = self.state[p]
                    if len(state) == 0:
                        state['step'] = 0
                        state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        if group['amsgrad']:
                            state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                    exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                    if group['amsgrad']:
                        max_exp_avg_sq = state['max_exp_avg_sq']

                    state['step'] += 1
                    beta1, beta2 = group['betas']

                    bias_correction1 = 1 - beta1 ** state['step']
                    bias_correction2 = 1 - beta2 ** state['step']

                    if group['weight_decay'] != 0:
                        p.add_(p, alpha=-group['weight_decay'] * group['lr'])

                    exp_avg.mul_(beta1).add_(effective_grad, alpha=1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(effective_grad, effective_grad, value=1 - beta2)

                    if group['amsgrad']:
                        torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                        denom = (max_exp_avg_sq.sqrt() / bias_correction2**0.5).add_(group['eps'])
                    else:
                        denom = (exp_avg_sq.sqrt() / bias_correction2**0.5).add_(group['eps'])

                    step_size = group['lr'] / bias_correction1
                    p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss
