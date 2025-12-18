import pdb
import math
import torch
from loguru import logger
from optimizers import get_optimizer
from transformers import get_cosine_schedule_with_warmup

def train(model, train_loader, optimizer, lr_scheduler, device, epoch=1):
    model.train()
    for epoch in range(epoch):
        for step, (inputs, labels) in enumerate(train_loader):
            
            outputs = model(inputs=inputs.to(device), labels=labels.to(device))
            loss = outputs.loss
            
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            
            # if (step + 1)   % 50 == 0:
            #     torch.cuda.empty_cache()
            print("step", step, "allocated:", torch.cuda.memory_allocated() / 1024**2)
            
            logger.info(
                f"Epoch: {epoch} Step: {step} LR: {optimizer.param_groups[0]['lr']} Training loss: {loss.cpu().item()}"
            )

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="qwen",
        choices=[
            "qwen", "llama",
            "lstm", 
            "resnet", 
            "mlp", 
            "quadratic"
        ]
    )
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "muon", "sgd"])
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=0.1)
    parser.add_argument("--dataset", type=str, default="openwebtext-100k")
    parser.add_argument("--hidden_size", type=int, default=1024)
    parser.add_argument("--limit", type=int, default=100_000)
    args = parser.parse_args()
    log_path = f"logs/{args.model}_{args.dataset}_{args.optimizer}_lr{args.lr}_n{args.limit}.log"
    logger.add(log_path)
    
    if args.model in ["qwen", "llama"]:
        from models.lm import get_model_and_dataloader
    elif args.model in ["resnet"]:
        from models.cnn import get_model_and_dataloader
    elif args.model in ["lstm"]:
        from models.rnn import get_model_and_dataloader
    elif args.model in ["quadratic"]:
        from models.quadratic import get_model_and_dataloader
    else:
        assert 0, f"model {args.model} not supported"
    model, train_loader = get_model_and_dataloader(
        args.model, args.dataset, args.hidden_size, args.limit
    )
    optimizer = get_optimizer(
        args.optimizer, model, lr=args.lr
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    model.train()
    epoch = 1
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=100,
        num_training_steps=len(train_loader) * epoch,
        num_cycles=0.5,
    )
    
    with open(log_path, "w") as f:
        pass
    train(
        model,
        train_loader,
        optimizer,
        lr_scheduler,
        device,
        epoch=epoch,
    )