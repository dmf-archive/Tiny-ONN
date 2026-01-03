from collections.abc import Callable

import torch
from torch.optim.optimizer import Optimizer

from src.optimizers.utils import zeropower_via_newtonschulz5


class ARSOptimizer(Optimizer):
    def __init__(self, params, **kwargs) -> None:
        defaults: dict = {
            'lr': 1e-3,
            'betas': (0.9, 0.999),
            'eps': 1e-8,
            'weight_decay': 0.01,
            'ns_steps': 5,
            'rho': 0.05,
            'k': 1,
            'alpha': 0.7,
            'adaptive': True,
        }
        for k, v in kwargs.items():
            if k in defaults:
                defaults[k] = v
        super().__init__(params, defaults)

    def step(self, closure: Callable[[], torch.Tensor] | None = None) -> torch.Tensor: # type: ignore
        if closure is None:
            raise ValueError("ARS requires a closure.")

        first_p: torch.Tensor = self.param_groups[0]['params'][0]
        state_p: dict = self.state[first_p]
        if 'step' not in state_p: state_p['step'] = 0
        state_p['step'] += 1
        global_step: int = state_p['step']

        k: int = self.param_groups[0]['k']
        is_sync_step: bool = (global_step % k == 1) or (k <= 1)

        loss: torch.Tensor | None = None

        if is_sync_step:
            with torch.enable_grad():
                loss = closure()
                loss.backward()

            with torch.no_grad():
                for group in self.param_groups:
                    rho: float = group['rho']
                    adaptive: bool = group['adaptive']
                    eps_val: float = group['eps']
                    beta2: float = group['betas'][1]

                    for p in group['params']:
                        if p.grad is None: continue
                        state: dict = self.state[p]

                        if 'exp_avg_sq' not in state:
                            state['exp_avg_sq'] = torch.zeros_like(p)

                        v_hat: torch.Tensor = state['exp_avg_sq'] / (1 - beta2 ** max(1, global_step - 1) + 1e-12)

                        g_nat: torch.Tensor = p.grad / (v_hat.sqrt() + eps_val)
                        if adaptive:
                            g_nat.mul_(p.abs())

                        norm: torch.Tensor = g_nat.norm() + 1e-12
                        perturb: torch.Tensor = g_nat * (rho / norm)

                        state['last_eps'] = perturb
                        state['g_base'] = p.grad.clone()
                        p.add_(perturb)

            self.zero_grad()
            with torch.enable_grad():
                loss_adv: torch.Tensor = closure()
                loss_adv.backward()

            with torch.no_grad():
                for group in self.param_groups:
                    for p in group['params']:
                        if p.grad is None: continue
                        state = self.state[p]
                        p.sub_(state['last_eps'])

                        beta1, beta2 = group['betas']
                        state['exp_avg_sq'].mul_(beta2).addcmul_(p.grad, p.grad, value=1 - beta2)

                        if k > 1:
                            g_base: torch.Tensor = state['g_base']
                            g_adv: torch.Tensor = p.grad
                            dot: torch.Tensor = (g_adv * g_base).sum()
                            base_norm_sq: torch.Tensor = (g_base * g_base).sum() + 1e-12
                            state['flatness_v'] = g_adv - (dot / base_norm_sq) * g_base
        else:
            with torch.enable_grad():
                loss = closure()
                loss.backward()

            with torch.no_grad():
                for group in self.param_groups:
                    alpha: float = group['alpha']
                    beta2 = group['betas'][1]
                    for p in group['params']:
                        if p.grad is None: continue
                        state = self.state[p]

                        if 'exp_avg_sq' not in state:
                            state['exp_avg_sq'] = torch.zeros_like(p)
                        state['exp_avg_sq'].mul_(beta2).addcmul_(p.grad, p.grad, value=1 - beta2)

                        if 'flatness_v' in state:
                            v: torch.Tensor = state['flatness_v']
                            g_norm: torch.Tensor = p.grad.norm()
                            v_norm: torch.Tensor = v.norm() + 1e-12
                            p.grad.add_(v, alpha=alpha * (g_norm / v_norm))

        with torch.no_grad():
            self._ada_rmsuon_update(global_step)

        if loss is None:
            # This should not happen given the logic above
            raise RuntimeError("Loss was not computed in ARSOptimizer.step")

        return loss

    @torch.no_grad()
    def _ada_rmsuon_update(self, global_step: int) -> None:
        for group in self.param_groups:
            is_rmsuon: bool = group.get('is_rmsuon_group', False)
            beta1: float
            beta2: float
            beta1, beta2 = group['betas']
            lr: float = group['lr']
            eps: float = group['eps']
            wd: float = group['weight_decay']
            ns_steps: int = group.get('ns_steps', 5)

            for p in group['params']:
                if p.grad is None: continue
                state: dict = self.state[p]
                if 'exp_avg' not in state:
                    state['exp_avg'] = torch.zeros_like(p)

                exp_avg: torch.Tensor = state['exp_avg']
                exp_avg.mul_(beta1).add_(p.grad, alpha=1 - beta1)

                m_hat: torch.Tensor = exp_avg / (1 - beta1 ** global_step)
                v_hat: torch.Tensor = state['exp_avg_sq'] / (1 - beta2 ** global_step)

                if is_rmsuon:
                    m_scaled: torch.Tensor = m_hat / (v_hat.sqrt() + eps)
                    energy: torch.Tensor = m_scaled.norm()

                    m_flat: torch.Tensor = m_scaled.view(m_scaled.size(0), -1) if p.ndim == 4 else m_scaled
                    s_ortho: torch.Tensor = zeropower_via_newtonschulz5(m_flat, steps=ns_steps)
                    if p.ndim == 4: s_ortho = s_ortho.view(m_scaled.shape)

                    update: torch.Tensor = energy * s_ortho
                    if wd != 0: p.mul_(1 - lr * wd)
                    p.add_(update, alpha=-lr)
                else:
                    denom: torch.Tensor = v_hat.sqrt().add_(eps)
                    if wd != 0: p.mul_(1 - lr * wd)
                    p.addcdiv_(m_hat, denom, value=-lr)
