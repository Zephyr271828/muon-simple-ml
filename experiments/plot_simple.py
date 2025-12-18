import glob
import argparse
import matplotlib.pyplot as plt 

parser = argparse.ArgumentParser()
parser.add_argument("--log_pattern", type=str, required=True)
parser.add_argument("--save_path", type=str, default="plots/loss_plot.png")
args = parser.parse_args()

def mean(array):
    return sum(array) / len(array)

log_patterns = [
    "logs/resnet18*imagenet*",
    "logs/mobilenet_v2*imagenet*",
    "logs/efficientnet_b0*imagenet*",
    "logs/convnext_tiny*imagenet*",
]

models = [
    "ResNet18",
    "MobileNet V2",
    "EfficientNet B0",
    "ConvNeXt Tiny",
]

fig, axes = plt.subplots(1, len(log_patterns), figsize=(5 * len(log_patterns), 5))

colors = [
    "blue",
    "orange",
    "green",
]

for i, log_pattern in enumerate(log_patterns):
    results = {}
    val_results = {}
    lrs = {}
    for log_path in glob.glob(log_pattern):
        losses = []
        val_losses = []
        with open(log_path, "r") as f:
            for l in f:
                if "Validation Loss:" in l:
                    loss = float(l.split("Validation Loss:")[-1])
                    val_losses.append(loss)
                elif "Training loss:" in l:
                    loss = float(l.split("Training loss:")[-1])
                    losses.append(loss)
        if len(losses) == 0:
            continue
                
        optim = [o for o in ["sgd", "adamw", "muon"] if o in log_path][0]
        lr = log_path.split("_lr")[1].split("_")[0]
        
        if optim not in results or mean(val_losses) < mean(val_results[optim]):
            results[optim] = losses
            val_results[optim] = val_losses
            lrs[optim] = lr
        
    results = {o: l for o, l in sorted(results.items())}
    val_results = {o: l for o, l in sorted(val_results.items())}
    
    for j, (optim, losses) in enumerate(results.items()):
        # label = log_path.split("/")[-1].replace(".log", "")
        axes[i].plot(
            [i * 10 for i in range(1, len(val_results[optim])+1)], 
            val_results[optim], 
            linestyle="-", 
            label=f"{optim} (lr={lrs[optim]}) train",
            color=colors[j]
        )
        axes[i].plot(
            [i for i in range(1, len(losses)+1)],
            losses, 
            linestyle="--",
            label=f"{optim} (lr={lrs[optim]}) val",
            color=colors[j],
            alpha=0.5
        )
    if i == 0:
        axes[i].set_ylabel("Training Loss")
    axes[i].set_xlabel("Training Steps")
    axes[i].set_title(models[i])
    axes[i].grid(True)
    axes[i].legend()
    
plt.tight_layout()
plt.savefig(args.save_path)

    
    # plt.savefig(args.save_path)
    # print(f"Saved plot to {args.save_path}")