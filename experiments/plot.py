import glob
import argparse
import matplotlib.pyplot as plt 

parser = argparse.ArgumentParser()
parser.add_argument("--log_pattern", type=str, required=True)
parser.add_argument("--save_path", type=str, default="plots/loss_plot.png")
args = parser.parse_args()

for log_path in glob.glob(args.log_pattern):
    losses = []
    with open(log_path, "r") as f:
        for l in f:
            if "Training loss:" not in l:
                continue
            losses.append(float(l.split()[-1]))
    
    label = log_path.split("/")[-1].replace(".log", "")
    plt.plot(losses, label=label)
plt.grid(True)
plt.tight_layout()
plt.legend()
plt.savefig(args.save_path)
print(f"Saved plot to {args.save_path}")