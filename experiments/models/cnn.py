import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet18

class CNNWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, inputs, labels=None):
        logits = self.model(inputs)
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels)
        return type("Out", (), {"loss": loss, "logits": logits})

class ImageNet(Dataset):
    """
    Drop-in replacement for torchvision.datasets.ImageNet
    Uses HuggingFace imagenet-1k instead.
    """

    def __init__(self, root=None, split="train", transform=None, limit=None):
        assert split in ["train", "validation", "test"]

        self.ds = load_dataset(
            "imagenet-1k",
            split=split,
            trust_remote_code=True,
        ).select(range(limit))

        self.transform = transform
        self.limit = limit

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        sample = self.ds[idx]
        img = sample["image"]     # PIL
        label = sample["label"]  # int

        if self.transform is not None:
            img = self.transform(img)

        return img, torch.tensor(label, dtype=torch.long)

def get_model_and_dataloader(
    model_name="resnet",
    dataset_name="cifar100",
    hidden_size=512,   # unused but kept for interface consistency
    limit=50_000,
):

    if dataset_name == "cifar100":
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize((0.5071, 0.4867, 0.4408),
                        (0.2675, 0.2565, 0.2761))
        ])
        train_dataset = torchvision.datasets.CIFAR100(
            root="./data",
            train=True,
            download=True,
            transform=transform,
        )
        num_classes = 100
    elif dataset_name == "imagenet":
        transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            )
        ])
        train_dataset = ImageNet(
            root="./data/imagenet",
            split="validation",
            transform=transform,
            limit=limit,
        )
        num_classes = 1000
    else:
        assert 0, f"dataset {dataset_name} not supported"

    if limit < len(train_dataset):
        train_dataset = torch.utils.data.Subset(
            train_dataset, range(limit)
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    if model_name == "resnet":
        model = resnet18(num_classes=num_classes)
    else:
        assert 0, f"model {model_name} not supported"

    return CNNWrapper(model), train_loader