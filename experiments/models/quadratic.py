# models/quadratic.py
import torch
from torch.utils.data import DataLoader, Dataset

class DummyDataset(Dataset):
    def __init__(self, dim, limit=1000):
        self.dim = dim
        self.limit = limit
    
    def __len__(self):
        return self.limit

    def __getitem__(self, idx):
        # Dummy input (not actually used)
        x = torch.zeros(self.dim)
        # Dummy label (not used either)
        y = torch.zeros(1)
        return x, y

# models/quadratic.py
import torch
from torch.utils.data import DataLoader, Dataset

class DummyDataset(Dataset):
    def __init__(self, dim, limit=1000):
        self.dim = dim
        self.limit = limit
    
    def __len__(self):
        return self.limit

    def __getitem__(self, idx):
        # Dummy input (not actually used)
        x = torch.zeros(self.dim)
        # Dummy label (not used either)
        y = torch.zeros(1)
        return x, y

class QuadraticModel(torch.nn.Module):
    def __init__(self, dim=128, device="cuda"):
        super().__init__()
        
        self.w = torch.nn.Parameter(torch.zeros(dim, device=device))
        # construct A = Q^T D Q, positive definite
        
        Q, _ = torch.linalg.qr(torch.randn(dim, dim, device=device))
        eigs = torch.linspace(0.1, 10.0, dim).to(device)  # condition number
        self.A = Q @ torch.diag(eigs) @ Q.T

    def forward(self, inputs=None, labels=None):
        # f(w) = 0.5 * w^T A w
        loss = 0.5 * (self.w @ (self.A @ self.w))

        # Return object with `.loss` for trainer compatibility
        return type("Out", (), {"loss": loss})

def get_model_and_dataloader(
    model_name="quadratic",
    dataset_name=None,
    hidden_size=128,
    limit=1000,
    device="cuda",
):
    assert model_name == "quadratic"
    model = QuadraticModel(dim=hidden_size, device=device)
    dataset = DummyDataset(dim=hidden_size, limit=limit)
    train_loader = DataLoader(dataset, batch_size=1, shuffle=True)
    return model, train_loader