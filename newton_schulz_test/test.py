import os
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm

os.makedirs("newton_schulz_test", exist_ok=True)

def zeropower_via_newtonschulz5(G, steps):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(0) > G.size(1):
        X = X.T
    # Ensure spectral norm is at most 1
    X = X / (X.norm() + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.T
        B = (
            b * A + c * A @ A
        )  # adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(0) > G.size(1):
        X = X.T
    return X

def newtonschulz_simple(G, steps):
    assert len(G.shape) == 2
    a, b = (3, -1)
    X = G.bfloat16()
    if G.size(0) > G.size(1):   
        X = X.T
    # Ensure spectral norm is at most 1
    X = X / (X.norm() + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.T
        X = a * X + b * A @ X
        
    if G.size(0) > G.size(1):
        X = X.T
    return X

def random_diagonal(n, device=None, dtype=None):
    diag = torch.randn(n, device=device, dtype=dtype).abs()
    return torch.diag(diag)

G = random_diagonal(8, device="cpu", dtype=torch.bfloat16)

traj1 = [G.clone()]
traj2 = [G.clone()]
func1 = lambda G: zeropower_via_newtonschulz5(G, steps=1)
func2 = lambda G: newtonschulz_simple(G, steps=1)

for step in tqdm(range(100)):
    G1 = func1(traj1[-1])
    G2 = func2(traj2[-1])
    traj1.append(G1.clone())
    traj2.append(G2.clone())
    
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
for i in range(1, G.shape[0]+1):
    axes[0].plot([t[i-1, i-1].cpu().item() for t in traj1], label=f"G_{i,i}")
    axes[1].plot([t[i-1, i-1].cpu().item() for t in traj2], label=f"G_{i,i}")
    
# plt.legend()
axes[0].set_title(r"$p_5(\Sigma)=3.4445\Sigma -4.7750\Sigma\Sigma^T \Sigma + 2.0315\Sigma \Sigma^T \Sigma \Sigma^T \Sigma$")
axes[0].grid(True)
axes[1].set_title(r"$p_3(\Sigma)=3\Sigma - \Sigma\Sigma^T \Sigma$")
axes[1].grid(True)

axes[0].set_ylabel("Diagonal Entries")
axes[0].set_xlabel("Iterations")
axes[1].set_xlabel("Iterations")

plt.tight_layout()
plt.savefig("newton_schulz_test/func1_convergence.pdf")
plt.close()

# # ---- Plot func2 iterations ----
# plt.figure()
# for i, y in enumerate(traj2):
#     plt.plot(x, y, label=f"iter {i}")
# plt.legend()
# plt.title("Newton-Schulz quintic iteration (func2)")
# plt.savefig("newton_schulz_test/func2_convergence.png")
# plt.close()