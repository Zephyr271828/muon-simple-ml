import os
import numpy as np
import matplotlib.pyplot as plt

os.makedirs("newton_schulz_test", exist_ok=True)

def func1(x):
    return 1.5 * x - 0.5 * x ** 3

def func2(x):
    return 3.4445 * x - 4.775 * x ** 3 + 2.0315 * x ** 5

x = np.linspace(-5, 5, 100)

x1 = x.copy()
x2 = x.copy()

traj1 = [x1.copy()]
traj2 = [x2.copy()]

for step in range(5):
    x1 = func1(x1)
    x2 = func2(x2)
    traj1.append(x1.copy())
    traj2.append(x2.copy())

# ---- Plot func1 iterations ----
plt.figure()
for i, y in enumerate(traj1):
    plt.plot(x, y, label=f"iter {i}")
plt.legend()
plt.title("Newton–Schulz cubic iteration (func1)")
plt.savefig("newton_schulz_test/func1_convergence.png")
plt.close()

# ---- Plot func2 iterations ----
plt.figure()
for i, y in enumerate(traj2):
    plt.plot(x, y, label=f"iter {i}")
plt.legend()
plt.title("Newton–Schulz quintic iteration (func2)")
plt.savefig("newton_schulz_test/func2_convergence.png")
plt.close()