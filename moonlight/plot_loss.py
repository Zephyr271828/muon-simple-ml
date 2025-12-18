import glob
import matplotlib.pyplot as plt 

results = {}
lrs = {}
for log_path in glob.glob("/work/nvme/bdhh/yxu21/ds-shu-200/logs/*10000*.log"):
    losses = []
    with open(log_path, "r") as f:
        for l in f:
            if "Training loss:" not in l:
                continue
            losses.append(float(l.split()[-1]))
    lr = log_path.split("_lr")[1].split("_")[0]
    optim = [o for o in ["sgd", "adamw", "muon"] if o in log_path][0]
    if optim not in results:
        results[optim] = losses
        lrs[optim] = lr
    elif losses[-1] < results[optim][-1]:
        results[optim] = losses
        lrs[optim] = lr
    
for optim, losses in results.items():
    # label = log_path.split("/")[-1].replace(".log", "")
    plt.plot(losses, label=f"{optim} (lr={lrs[optim]})")
plt.grid(True)
plt.tight_layout()
plt.legend()
plt.savefig("moonlight/loss_plot.png")