import torch

def run_equivalence_test():
    P = 10
    K_INNER_STEPS = 5
    LR_INNER = 0.1
    
    print("--- G-CAFM diag(G) Equivalence Test ---")
    print(f"Matrix Size P={P}, Inner Steps K={K_INNER_STEPS}\n")

    A = torch.randn(P, P)
    G = A.T @ A + torch.eye(P) * 1e-3
    g = torch.randn(P)

    diag_true = torch.diag(G)
    print(f"1. True Diag(G):\n{diag_true.numpy()}\n")

    g_squared_approx = g**2
    mse_g_squared = torch.mean((g_squared_approx - diag_true)**2).item()
    print(f"2. g^2 Approximation:\n{g_squared_approx.numpy()}")
    print(f"   - MSE vs True: {mse_g_squared:.6f}\n")

    z = torch.randint(0, 2, (P,)).float() * 2 - 1
    v = G @ z
    print("3. Hutchinson-like Approximation Analysis:")
    sample_counts = [1, 10, 100, 1000]
    for num_samples in sample_counts:
        hutch_samples = []
        for _ in range(num_samples):
            z = torch.randint(0, 2, (P,)).float() * 2 - 1
            v = G @ z
            hutch_samples.append((v * z).unsqueeze(0))
        
        diag_hutch_avg_approx = torch.mean(torch.cat(hutch_samples, dim=0), dim=0)
        mse_hutch_avg = torch.mean((diag_hutch_avg_approx - diag_true)**2).item()
        print(f"\n   - With {num_samples} sample(s):")
        print(f"     - Result: {diag_hutch_avg_approx.numpy()}")
        print(f"     - MSE vs True: {mse_hutch_avg:.6f}")
    print("")

    d = torch.zeros(P, requires_grad=True)
    optimizer_d = torch.optim.SGD([d], lr=LR_INNER)
    
    accumulated_importance = torch.zeros(P)
    
    print("4. Inner Loop Accumulation Approximation:")
    for k in range(K_INNER_STEPS):
        optimizer_d.zero_grad()
        gnvp = G @ d
        quad_loss_grad = g + gnvp
        
        accumulated_importance += quad_loss_grad.detach()**2
        
        d.backward(quad_loss_grad)
        optimizer_d.step()
        
        print(f"   - Step {k+1}: d_norm={torch.norm(d.detach()).item():.4f}, grad_norm={torch.norm(quad_loss_grad.detach()).item():.4f}")

    accumulated_importance_norm = accumulated_importance / (torch.norm(accumulated_importance) + 1e-9)
    diag_true_norm = diag_true / (torch.norm(diag_true) + 1e-9)
    
    mse_inner_loop = torch.mean((accumulated_importance_norm - diag_true_norm)**2).item()
    print(f"\n   - Final Accumulated (Normalized):\n{accumulated_importance_norm.numpy()}")
    print(f"   - True Diag (Normalized for comparison):\n{diag_true_norm.numpy()}")
    print(f"   - MSE (Normalized): {mse_inner_loop:.6f}\n")

if __name__ == "__main__":
    run_equivalence_test()