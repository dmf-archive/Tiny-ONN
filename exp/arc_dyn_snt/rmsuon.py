
import torch


@torch.jit.script
def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.to(dtype=torch.bfloat16) if G.dtype == torch.float32 else G

    if G.size(-2) > G.size(-1):
        X = X.mT

    X = X.div(X.norm(p=2.0, dim=[-2, -1], keepdim=True).add(1e-7))

    for _ in range(steps):
        A = X.matmul(X.mT)
        B = torch.addmm(A, A, A, beta=b, alpha=c)
        X = torch.addmm(X, B, X, beta=a, alpha=1.0)

    if G.size(-2) > G.size(-1):
        X = X.mT

    return X


@torch.jit.script
def _update_adam_stats(
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    grad: torch.Tensor,
    beta1: float,
    beta2: float,
):
    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

def _calculate_energy_and_momentum(
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    step: int,
    beta1: float,
    beta2: float,
    eps: float
) -> tuple[torch.Tensor, torch.Tensor]:
    bias_correction1 = 1 - beta1 ** step
    bias_correction2 = 1 - beta2 ** step

    m_hat = exp_avg / bias_correction1
    v_hat = exp_avg_sq / bias_correction2

    denom = v_hat.sqrt().add_(eps)

    # This is memory-intensive, but we need the norm
    adam_update = m_hat.div(denom)
    energy = adam_update.norm()

    return m_hat, energy


@torch.jit.script
def adamw_step_kernel(
    param: torch.Tensor,
    grad: torch.Tensor,
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    beta1: float,
    beta2: float,
    step: int,
    lr: float,
    weight_decay: float,
    eps: float
):
    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

    bias_correction1 = 1 - beta1 ** step
    bias_correction2 = 1 - beta2 ** step

    denom = (exp_avg_sq.sqrt() / (bias_correction2 ** 0.5)).add_(eps)

    step_size = lr / bias_correction1

    if weight_decay != 0:
        param.mul_(1 - lr * weight_decay)

    param.addcdiv_(exp_avg, denom, value=-step_size)

class RMSuon(torch.optim.Optimizer):
    def __init__(self, params, **kwargs):
        # This optimizer is designed to be initialized via a factory that prepares param_groups.
        # The `__init__` is simplified to accept pre-structured param_groups.
        defaults = {
            'lr': 1e-3,
            'betas': (0.9, 0.999),
            'eps': 1e-8,
            'weight_decay': 0.01,
            'ns_steps': 5,
        }
        # The factory will set group-specific hyperparams.
        # We only need to pass the params (which are already groups) and a default dict.
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            is_rmsuon_group = group.get('is_rmsuon_group', False)

            if is_rmsuon_group:
                self._rmsuon_step(group)
            else:
                self._adamw_step(group)

        return loss

    def _rmsuon_step(self, group: dict):
        beta1, beta2 = group['betas']
        lr = group['lr']
        eps = group['eps']
        weight_decay = group['weight_decay']
        ns_steps = group['ns_steps']

        for p in group['params']:
            if p.grad is None:
                continue

            grad = p.grad
            state = self.state[p]

            if len(state) == 0:
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

            state['step'] += 1
            step = state['step']
            exp_avg = state['exp_avg']
            exp_avg_sq = state['exp_avg_sq']

            # Step 1: Update stats (lightweight, JIT-able)
            _update_adam_stats(exp_avg, exp_avg_sq, grad, beta1, beta2)

            # Step 2: Calculate energy and bias-corrected momentum (heavier)
            m_hat, energy = _calculate_energy_and_momentum(exp_avg, exp_avg_sq, step, beta1, beta2, eps)

            # Step 3: Orthogonalize
            original_shape = m_hat.shape
            m_hat_flat = m_hat.view(m_hat.size(0), -1) if p.ndim == 4 else m_hat

            # Note: The performance bottleneck might be the CPU->GPU transfer if m_hat is not on the correct device
            # or the data type conversion inside zeropower. Let's assume m_hat is on the correct device.
            O = zeropower_via_newtonschulz5(m_hat_flat, steps=ns_steps)

            if p.ndim == 4:
                O = O.view(original_shape)

            # Step 4: Apply update
            base_energy = O.norm().add_(1e-10)
            scale = energy / base_energy

            p.mul_(1 - lr * weight_decay)
            p.add_(O, alpha=-lr * scale)

    def _adamw_step(self, group: dict):
        beta1, beta2 = group['betas']
        lr = group['lr']
        eps = group['eps']
        weight_decay = group['weight_decay']

        for p in group['params']:
            if p.grad is None:
                continue

            grad = p.grad
            state = self.state[p]

            if len(state) == 0:
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)

            state['step'] += 1
            step = state['step']
            exp_avg = state['exp_avg']
            exp_avg_sq = state['exp_avg_sq']

            adamw_step_kernel(
                p, grad, exp_avg, exp_avg_sq,
                beta1, beta2, step, lr, weight_decay, eps
            )
