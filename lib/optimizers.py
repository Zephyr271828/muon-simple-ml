from abc import ABC, abstractmethod
import torch


class Optimizer(ABC):
    def __init__(self, lr: float):
        self.lr = lr
        self.t = 0  # step counter

    @abstractmethod
    def step(self, task):
        pass
    
class SGD(Optimizer):
    def __init__(self, lr=1e-2):
        super().__init__(lr)

    def step(self, task):
        self.t += 1

        params = task.params()
        grads = task.grad()

        delta = {}
        for k in params:
            delta[k] = -self.lr * grads[k]

        task.apply_update(delta)
            
class AdamW(Optimizer):
    def __init__(self, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        super().__init__(lr)
        self.b1, self.b2 = betas
        self.eps = eps
        self.wd = weight_decay
        self.m = {}
        self.v = {}

    def step(self, task):
        self.t += 1

        params = task.params()
        grads = task.grad()

        delta = {}

        for k in params:
            if k not in self.m:
                self.m[k] = torch.zeros_like(params[k])
                self.v[k] = torch.zeros_like(params[k])

            g = grads[k]

            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * (g * g)

            m_hat = self.m[k] / (1 - self.b1 ** self.t)
            v_hat = self.v[k] / (1 - self.b2 ** self.t)

            update = m_hat / (torch.sqrt(v_hat) + self.eps)

            # Decoupled weight decay
            w_decay = self.wd * params[k]

            delta[k] = -self.lr * (update + w_decay)

        task.apply_update(delta)
            
class Muon(Optimizer):
    """
    Muon = Momentum + Newton–Schulz Orthogonalization
    """
    def __init__(self, lr=1e-2, momentum=0.9, ns_steps=5, eps=1e-7):
        super().__init__(lr)
        self.momentum = momentum
        self.ns_steps = ns_steps
        self.eps = eps
        self.m = {}

        # NS coefficients (from paper / implementation)
        self.a = 3.4445
        self.b = -4.7750
        self.c = 2.0315

    def _newton_schulz(self, G):
        if G.ndim < 2:
            return G  # fallback for vectors/biases

        X = G.to(torch.float32)
        X = X / (X.norm() + self.eps)

        transposed = False
        if X.shape[0] > X.shape[1]:
            X = X.T
            transposed = True

        for _ in range(self.ns_steps):
            A = X @ X.T
            B = self.b * A + self.c * A @ A
            X = self.a * X + B @ X

        if transposed:
            X = X.T

        return X.to(G.dtype)

    def step(self, task):
        self.t += 1

        params = task.params()
        grads = task.grad()
        delta = {}

        for k in params:
            if k not in self.m:
                self.m[k] = torch.zeros_like(params[k])

            # Momentum accumulation
            self.m[k] = self.momentum * self.m[k] + grads[k]

            # Orthogonalized update
            update = self._newton_schulz(self.m[k])

            delta[k] = -self.lr * update

        task.apply_update(delta)
