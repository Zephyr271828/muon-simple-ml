from abc import ABC, abstractmethod
import torch


from abc import ABC, abstractmethod
from typing import Iterable, Dict
import torch

class Task(ABC):
    @abstractmethod
    def loss(self) -> torch.Tensor:
        pass

    @abstractmethod
    def grad(self) -> Dict[str, torch.Tensor]:
        """
        Return a dict of gradients keyed by parameter name
        """
        pass

    @abstractmethod
    def params(self) -> Dict[str, torch.Tensor]:
        """
        Return a dict of parameters (oracle)
        """
        pass

    @abstractmethod
    def set_params(self, new_params: Dict[str, torch.Tensor]):
        pass

    def apply_update(self, delta: Dict[str, torch.Tensor]):
        params = self.params()
        for k in params:
            params[k] = params[k] + delta[k]
        self.set_params(params)
        

class QuadraticTask(Task):
    def __init__(self, d=128, cond=1e3, seed=0, device="cpu"):
        torch.manual_seed(seed)
        self.device = device

        # Create SPD matrix with controlled condition number
        U, _ = torch.linalg.qr(torch.randn(d, d))
        eigs = torch.logspace(0, torch.log10(torch.tensor(cond)), d)
        self.A = U @ torch.diag(eigs) @ U.T
        self.b = torch.randn(d, device=device)

        self.x = torch.zeros(d, device=device)

        self.x_star = torch.linalg.solve(self.A, self.b)
        self.c = 0.5 * self.x_star @ self.A @ self.x_star - self.b @ self.x_star

    def loss(self):
        return 0.5 * self.x @ self.A @ self.x - self.b @ self.x - self.c

    def grad(self):
        return {"x": self.A @ self.x - self.b}

    def params(self):
        return {"x": self.x}

    def set_params(self, new_params):
        self.x = new_params["x"]

class LinearSystemTask(Task):
    def __init__(self, n=512, d=64, seed=0, device="cpu"):
        torch.manual_seed(seed)
        self.device = device

        self.X = torch.randn(n, d, device=device) 
        self.y = torch.randn(n, device=device) 

        self.w = torch.zeros(d, device=device)

        # Closed-form solution (for evaluation)
        XTX = self.X.T @ self.X / n
        XTy = self.X.T @ self.y / n
        self.w_star = torch.linalg.solve(XTX + 1e-6 * torch.eye(d, device=device), XTy)

    def loss(self):
        r = self.X @ self.w - self.y
        return 0.5 * torch.mean(r ** 2)

    def grad(self):
        g = (self.X.T @ (self.X @ self.w - self.y)) / self.X.shape[0]
        return {"w": g}

    def params(self):
        return {"w": self.w}

    def set_params(self, new_params):
        self.w = new_params["w"]

import math
import torch
import torch.nn.functional as F

class MLPRegressionTask(Task):
    def __init__(self, n=1024, d=32, h=128, depth=2, seed=0, device="cpu"):
        torch.manual_seed(seed)
        self.device = device
        self.depth = depth

        # ----- Normalized input -----
        X = torch.randn(n, d, device=device)
        self.X = (X - X.mean(0)) / (X.std(0) + 1e-6)

        # ----- Teacher model -----
        W_true = torch.randn(d, 1, device=device)
        y = self.X @ W_true + 0.05 * torch.randn(n, 1, device=device)
        self.y = (y - y.mean()) / (y.std() + 1e-6)

        # ----- Weights only (no bias) -----
        self.weights = {}
        dims = [d] + [h] * depth + [1]

        for i in range(len(dims) - 1):
            self.weights[f"W{i}"] = (
                torch.randn(dims[i], dims[i+1], device=device)
                / math.sqrt(dims[i])
            )

    def forward(self):
        h = self.X
        L = len(self.weights)

        for i in range(L):
            W = self.weights[f"W{i}"]
            h = h @ W

            if i < L - 1:
                h = F.layer_norm(h, h.shape[1:])  # ✅ stabilization
                h = torch.relu(h)

        return h

    def loss(self):
        return torch.mean((self.forward() - self.y) ** 2)

    def grad(self):
        for p in self.weights.values():
            p.requires_grad_(True)

        loss = self.loss()
        loss.backward()

        grads = {k: v.grad.clone() for k, v in self.weights.items()}

        for p in self.weights.values():
            p.grad = None
            p.requires_grad_(False)

        return grads

    def params(self):
        return self.weights

    def set_params(self, new_params):
        self.weights = new_params
            
class ResNetTask(Task):
    """
    Bias-free residual network with LayerNorm:
        h₀ = X W_in
        h_{k+1} = h_k + ReLU(LN(h_k W_k))
        ŷ = h_L W_out
    """
    def __init__(self, n=1024, d=32, width=128, depth=4, seed=0, device="cpu"):
        assert torch.cuda.is_available(), "ResNetTask requires CUDA for reasonable speed."
        torch.manual_seed(seed)
        self.device = device
        self.depth = depth

        # ----- Normalized input -----
        X = torch.randn(n, d, device=device)
        self.X = (X - X.mean(0)) / (X.std(0) + 1e-6)

        # ----- Teacher mapping -----
        W_true = torch.randn(d, 1, device=device)
        y = self.X @ W_true + 0.05 * torch.randn(n, 1, device=device)
        self.y = (y - y.mean()) / (y.std() + 1e-6)

        self.params_dict = {}

        # Input projection
        self.params_dict["W_in"] = (
            torch.randn(d, width, device=device)
            / math.sqrt(d)
        )

        # Residual blocks
        for k in range(depth):
            self.params_dict[f"W{k}"] = (
                torch.randn(width, width, device=device)
                / math.sqrt(width)
            )

        # Output head
        self.params_dict["W_out"] = (
            torch.randn(width, 1, device=device)
            / math.sqrt(width)
        )

    def forward(self):
        h = self.X @ self.params_dict["W_in"]

        for k in range(self.depth):
            W = self.params_dict[f"W{k}"]
            z = F.layer_norm(h @ W, h.shape[1:])
            h = h + torch.relu(z)

        return h @ self.params_dict["W_out"]

    def loss(self):
        return torch.mean((self.forward() - self.y) ** 2)

    def grad(self):
        for p in self.params_dict.values():
            p.requires_grad_(True)

        loss = self.loss()
        loss.backward()

        grads = {k: v.grad.clone() for k, v in self.params_dict.items()}

        for p in self.params_dict.values():
            p.grad = None
            p.requires_grad_(False)

        return grads

    def params(self):
        return self.params_dict

    def set_params(self, new_params):
        self.params_dict = new_params