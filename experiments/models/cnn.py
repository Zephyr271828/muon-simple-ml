import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from torchvision import models

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

class HFDatasetWrapper(Dataset):
    """Wrap a HuggingFace image dataset into a PyTorch Dataset."""

    def __init__(self, hf_dataset, transform=None):
        self.ds = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        sample = self.ds[idx]
        img = sample["image"]
        label = sample["label"]

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
        raw_dataset = torchvision.datasets.CIFAR100(
            root="./data",
            train=True,
            download=True,
            transform=transform,
        )
        if limit < len(raw_dataset):
            raw_dataset = torch.utils.data.Subset(
                raw_dataset, range(limit)
            )
        num_classes = 100
    elif dataset_name == "imagenet":
        transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.Lambda(lambda img: img.convert("RGB")),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)),
        ])
        
        hf_ds = load_dataset("timm/mini-imagenet", split="train")
        if limit is not None:
            hf_ds = hf_ds.select(range(min(limit, len(hf_ds))))
        raw_dataset = HFDatasetWrapper(hf_ds, transform=transform)
        
        num_classes = 100
    else:
        assert 0, f"dataset {dataset_name} not supported"

    test_size = int(len(raw_dataset) * 0.01)
    train_dataset, val_dataset = torch.utils.data.random_split(
        raw_dataset, [len(raw_dataset) - test_size, test_size]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    if model_name == "resnet18":
        model = models.resnet18(pretrained=False, num_classes=num_classes)
        
    elif model_name == "mobilenet_v2":
        model = models.mobilenet_v2(pretrained=False)
        model.classifier[1] = nn.Linear(
            model.classifier[1].in_features, num_classes
        )
        
    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(pretrained=False)
        model.classifier[1] = nn.Linear(
            model.classifier[1].in_features, num_classes
        )
        
    elif model_name == "convnext_tiny":
        model = models.convnext_tiny(pretrained=False)
        model.classifier[2] = nn.Linear(
            model.classifier[2].in_features, num_classes
        )
        
    elif model_name == "vit_b_16":
        model = models.vit_b_16(pretrained=False)
        model.heads[0] = nn.Linear(
            model.heads[0].in_features, num_classes
        )
        
    else:
        assert 0, f"model {model_name} not supported"

    return CNNWrapper(model), train_loader, val_loader