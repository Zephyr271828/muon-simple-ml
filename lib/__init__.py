from .optimizers import (
    SGD,
    AdamW,
    Muon
)
from .tasks import (
    QuadraticTask,
    LinearSystemTask,
    MLPRegressionTask,
    ResNetTask
)
from .train_utils import run, plot_losses

optimizer_registry = {
    "SGD": SGD,
    "AdamW": AdamW,
    "Muon": Muon,
}
task_registry = {
    "Quadratic": QuadraticTask,
    "LinearSystem": LinearSystemTask,
    "MLPRegression": MLPRegressionTask,
    "ResNet": ResNetTask,
}
__all__ = [
    "optimizer_registry",
    "task_registry",
    "run",
    "plot_losses",
]