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
        if X.ndim > 2:
            B = b * A + c * A.matmul(A)
            X = a * X + B.matmul(X)
        else:
            B = torch.addmm(A, A, A, beta=b, alpha=c)
            X = torch.addmm(X, B, X, beta=a, alpha=1.0)

    if G.size(-2) > G.size(-1):
        X = X.mT

    return X
