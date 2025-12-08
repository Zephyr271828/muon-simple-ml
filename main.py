import torch
import matplotlib.pyplot as plt
from functools import partial
from tqdm import tqdm

from lib import optimizer_registry, task_registry, run, plot_losses

device = "cuda" if torch.cuda.is_available() else "cpu"

task_cls = {
    "Quadratic": partial(task_registry["Quadratic"], d=32, seed=42, device=device),
    "LinearSystem": partial(task_registry["LinearSystem"], n=512, d=32, seed=42, device=device),
    "MLPRegression": partial(task_registry["MLPRegression"], n=1024, d=32, h=128, depth=2, seed=42, device=device),
    "ResNet": partial(task_registry["ResNet"], n=1024, d=32, width=128, depth=4, seed=42, device=device),
}

task_name = "ResNet"  # Choose task here
steps = 5000

def main():
    
    results = {}
    
    for opt_name, optimizer in optimizer_registry.items():
        results[opt_name] = {}
        for lr in [1e-5, 3e-5, 1e-4, 3e-4, 1e-3]:
            # task = QuadraticTask(
            task = task_cls[task_name]()  # Reset task for each run
            opt_instance = optimizer(lr=lr)
            w_final, losses = run(task, opt_instance, steps=steps)
            print(f"{opt_name} with lr={lr}: final loss = {losses[-1]:.3e}")
            results[opt_name][f"lr={lr}"] = losses  
            
    save_path = f"plots/demo_{task_name.lower()}_step_v_loss.png"
    plot_losses(results, save_path=save_path)
        
if __name__ == '__main__':
    main()