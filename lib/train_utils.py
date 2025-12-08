import time
import matplotlib.pyplot as plt
from tqdm import tqdm

def run(task, optimizer, steps=500, threshold=1e-5, task_name="task"):
    losses = []

    for t in tqdm(range(steps), desc="task={}|optim={}|lr={}".format(task_name, optimizer.__class__.__name__, optimizer.lr)):
        loss = task.loss()
        optimizer.step(task)
        losses.append(loss.item())
        # if len(losses) > 1 and abs(losses[-1] - losses[-2]) < threshold:
        #     break
        if loss.item() < threshold:
            break
       

    return task.params(), losses

def plot_losses(results, save_path="optimizer_comparison.png"):
    # --- Plot ---
    ymin = float('inf')
    ymax = float('-inf')
    fig, axes = plt.subplots(1, len(results), figsize=(len(results) * 5, 5))
    for i, (gname, group) in enumerate(results.items()):
        for label, losses in group.items():
            axes[i].plot(losses, label=f"{label}")
            ymin = min(ymin, min(losses))
            # ymax = max(ymax, max(losses))
            ymax = max(ymax, losses[0])
        
        axes[i].set_title(f"{gname}")
        axes[i].set_yscale("log")
        axes[i].set_xlabel("Step")
        axes[i].set_ylabel("Loss")
        axes[i].grid(True)
        axes[i].legend()
    
    # for i in range(len(results)):
    #     axes[i].set_ylim([ymin * 0.9, ymax * 1.1])
        
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved plot to {save_path}")