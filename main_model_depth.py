import torch
import matplotlib.pyplot as plt
from functools import partial
from tqdm import tqdm

from lib import optimizer_registry, task_registry, run, plot_losses

device = "cuda" if torch.cuda.is_available() else "cpu"

task_cls = {
    "ResNet-2": partial(task_registry["ResNet"], n=1024, d=32, width=128, depth=2, seed=42, device=device),
    "ResNet-4": partial(task_registry["ResNet"], n=1024, d=32, width=128, depth=4, seed=42, device=device),
    "ResNet-8": partial(task_registry["ResNet"], n=1024, d=32, width=128, depth=6, seed=42, device=device),
    "ResNet-16": partial(task_registry["ResNet"], n=1024, d=32, width=128, depth=16, seed=42, device=device),
}

task_name = "ResNet"  # Choose task here
steps = 5000

def main():
    
    results = {}
    
    for task_name, task in task_cls.items():
        results[task_name] = {}
        for opt_name in ["AdamW", "Muon"]:  # optimizer_registry.items():
            optimizer_cls = optimizer_registry[opt_name]
            best_losses = [float('inf')]
            for lr in [1e-5, 3e-5, 1e-4, 3e-4, 1e-3]:
                task = task_cls[task_name]()  # Reset task for each run
                opt_instance = optimizer_cls(lr=lr)
                w_final, losses = run(task, opt_instance, steps=steps, task_name=task_name)
                print(f"{opt_name} with lr={lr}: final loss = {losses[-1]:.3e}")
                if min(losses) < min(best_losses) \
                    or (min(losses) == min(best_losses) and len(losses) < len(best_losses)):
                    best_losses = losses
                    best_lr = lr
            results[task_name][f"{opt_name}_lr={best_lr}"] = best_losses  
            
    save_path = f"plots/model_depth_step_v_loss.png"
    plot_losses(results, save_path=save_path)
        
if __name__ == '__main__':
    main()