import glob
import argparse
import matplotlib.pyplot as plt 

parser = argparse.ArgumentParser()
parser.add_argument("--log_pattern", type=str, required=True)
parser.add_argument("--save_path", type=str, default="plots/loss_plot.png")
args = parser.parse_args()

def mean(array):
    return sum(array) / len(array)

results = {}
lrs = {}
for log_path in glob.glob(args.log_pattern):
    losses = []
    with open(log_path, "r") as f:
        for l in f:
            if "Training loss:" not in l:
                continue
            losses.append(float(l.split()[-1]))
            
    optim = [o for o in ["sgd", "adamw", "muon"] if o in log_path][0]
    lr = log_path.split("_lr")[1].split("_")[0]
    
    if optim not in results:
        results[optim] = losses
        lrs[optim] = lr
    elif mean(losses) < mean(results[optim]):
        results[optim] = losses
        lrs[optim] = lr
    
for optim, losses in results.items():
    # label = log_path.split("/")[-1].replace(".log", "")
    plt.plot(losses, label=f"{optim} (lr={lrs[optim]})")
plt.xlabel("Training Steps")
plt.ylabel("Training Loss")
plt.grid(True)
plt.tight_layout()
plt.legend()
plt.savefig(args.save_path)
print(f"Saved plot to {args.save_path}")