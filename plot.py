import matplotlib.pyplot as plt
import numpy as np

# Use 'classic' style for exact paper-like appearance (white bg, light gray grid)
plt.style.use("classic")

# Set font details
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Helvetica", "Arial"]
plt.rcParams["font.size"] = 10
plt.rcParams["axes.labelsize"] = 10
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["xtick.labelsize"] = 8
plt.rcParams["ytick.labelsize"] = 8
plt.rcParams["legend.fontsize"] = 8
plt.rcParams["grid.color"] = "lightgray"
plt.rcParams["grid.linestyle"] = "-"
plt.rcParams["grid.alpha"] = 0.5

# Methods and exact matching colors from paper's figure
methods_upper = ["Vine", "Single Path", "Natural Gradient", "Max KL", "Empirical FIM", "CEM", "CMA", "RWR"]
colors_upper = [
  "#1f77b4",
  "#ff7f0e",
  "#2ca02c",
  "#e377c2",
  "#17becf",
  "#d62728",
  "#8c564b",
  "#7f7f7f",
]  # blue, orange, green, magenta/pink, cyan, red, brown, gray

methods_lower = ["Vine", "Single Path", "Natural Gradient", "CEM"]
colors_lower = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]  # blue, orange, green, red


# Function to generate approximate data (mean and std) for a task
def generate_data(num_iters, base_curves, scales, noise_level=0.2, start_shift=0):
  x = np.arange(num_iters)
  means = []
  stds = []
  for i, base in enumerate(base_curves):
    # Sigmoid-like growth to mimic learning curves
    mean = base * (1 - np.exp(-x / scales[i])) + np.random.normal(0, 0.05, num_iters) + start_shift
    std = np.abs(mean) * noise_level * (1 - x / num_iters) + 0.1  # Decreasing variance
    means.append(mean)
    stds.append(std)
  return x, means, stds


# Data for each task (approximated to match trends/scales)
# Cartpole: Quick rise to ~10 for good methods, lower for others
base_cart = [10, 9, 8, 3, 4, 2, 1, 0.5]
scales_cart = [5, 6, 8, 20, 15, 30, 35, 40]
x_cart, means_cart, stds_cart = generate_data(50, base_cart, scales_cart, 0.15)

# Swimmer: Good methods rise to ~0.15, poor drop to ~-0.05
base_swim = [0.15, 0.13, 0.10, 0.05, 0.02, -0.02, -0.04, -0.05]
scales_swim = [5, 6, 8, 10, 12, 15, 20, 25]
x_swim, means_swim, stds_swim = generate_data(50, base_swim, scales_swim, 0.1, start_shift=-0.05)

# Hopper: Good to ~2.5, poor ~0.5 or less
base_hop = [2.5, 2.2, 0.8, 0.4]
scales_hop = [40, 45, 100, 120]
x_hop, means_hop, stds_hop = generate_data(200, base_hop, scales_hop, 0.25)

# Walker: Good to ~3, poor ~0.5 or less
base_walk = [3.2, 2.8, 0.9, 0.5]
scales_walk = [50, 55, 110, 130]
x_walk, means_walk, stds_walk = generate_data(200, base_walk, scales_walk, 0.3)

# Create 2x2 subplots with explicit white figure background
fig, axs = plt.subplots(2, 2, figsize=(10, 8), sharex=False, sharey=False)
fig.patch.set_facecolor("white")  # Explicit white background for figure
axs = axs.flatten()


# Plot function
def plot_task(ax, x, means, stds, title, ylabel, methods, colors, step):
  ax.set_facecolor("white")  # Explicit white background for each axes
  for i, (method, color) in enumerate(zip(methods, colors)):
    ax.plot(x, means[i], color=color, label=method, linewidth=1.5)
    # Error bars spaced out for readability (every 'step' points)
    ax.errorbar(x[::step], means[i][::step], yerr=stds[i][::step], fmt="none", ecolor=color, elinewidth=1, capsize=0)
  ax.set_title(title)
  ax.set_xlabel("number of policy iterations")
  ax.set_ylabel(ylabel)
  ax.grid(True)
  ax.legend(loc="lower right", frameon=True, edgecolor="black")


# Plot each task with appropriate step for spacing error bars
plot_task(axs[0], x_cart, means_cart, stds_cart, "Cartpole", "Reward", methods_upper, colors_upper, step=5)
plot_task(axs[1], x_swim, means_swim, stds_swim, "Swimmer", "Cost (- velocity ctrl)", methods_upper, colors_upper, step=5)
plot_task(axs[2], x_hop, means_hop, stds_hop, "Hopper", "Reward", methods_lower, colors_lower, step=10)
plot_task(axs[3], x_walk, means_walk, stds_walk, "Walker", "Reward", methods_lower, colors_lower, step=10)

# Adjust layout and show/save
plt.tight_layout()


plt.savefig("output.png")
plt.show()  # Or plt.savefig('trpo_figure4_reproduction.png') to save
