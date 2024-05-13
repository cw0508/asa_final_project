import numpy as np
import random
import matplotlib.pyplot as plt
from collections import Counter
from scipy.optimize import curve_fit

plt.style.use('plot_style.txt')

# -----------------------------------------------------------------
# Variables:
# -----------------------------------------------------------------
# Parameters
Lx, Ly = 200, 200   # Lattice dimensions
t_nn = 1.0          # Nearest neighbor hopping constant
U = 2.0             # On-site Coulomb interaction strength
T = 0.1             # Temperature in units where kB = 1
kB = 1              # Boltzmann constant
mc_steps = 2000     # Number of Monte Carlo steps

# -----------------------------------------------------------------
# Function to compute the lattice energy
def compute_energy(lattice, t_nn, U):
    nn_energy = 0
    for i in range(Lx):
        for j in range(Ly):
            nn_energy += -t_nn * lattice[i, j] * (lattice[(i + 1) % Lx, j] + lattice[i, (j + 1) % Ly])
            nn_energy += U * (lattice[i, j] == 1)
    return nn_energy

# -----------------------------------------------------------------
# Function to compute energy change due to flipping a spin
def energy_change(lattice, i, j, t_nn, U):
    delta_E = 2 * t_nn * lattice[i, j] * (lattice[(i - 1) % Lx, j] + lattice[(i + 1) % Lx, j] +
                                         lattice[i, (j - 1) % Ly] + lattice[i, (j + 1) % Ly])
    if lattice[i, j] == 1:
        delta_E += 2 * U
    return delta_E

# -----------------------------------------------------------------
# Initialize the lattice
lattice = np.random.choice([-1, 1], size=(Lx, Ly))
lattices = {}
current_energy = compute_energy(lattice, t_nn, U)

# Compute intervals for snapshots
snapshot_intervals = [int(mc_steps * i / 4) for i in range(5)]

# Monte Carlo simulation using the Metropolis-Hastings algorithm
for step in range(mc_steps):
    i, j = np.random.randint(0, Lx), np.random.randint(0, Ly)
    delta_E = energy_change(lattice, i, j, t_nn, U)
    if random.random() < np.exp(-delta_E / (kB * T)):
        lattice[i, j] *= -1
        current_energy += delta_E

    if step in snapshot_intervals:
        lattices[step] = lattice.copy()

# -----------------------------------------------------------------
# Flood fill function for cluster analysis
def flood_fill(lattice, x, y, label, clusters, visited):
    stack = [(x, y)]
    Lx, Ly = lattice.shape
    size = 0
    while stack:
        cx, cy = stack.pop()
        if visited[cx, cy]:
            continue
        visited[cx, cy] = True
        clusters[cx, cy] = label
        size += 1
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = (cx + dx) % Lx, (cy + dy) % Ly
            if lattice[cx, cy] == lattice[nx, ny] and not visited[nx, ny]:
                stack.append((nx, ny))
    return size

# -----------------------------------------------------------------
def analyze_clusters(lattice):
    visited = np.zeros_like(lattice, dtype=bool)
    clusters = np.zeros_like(lattice, dtype=int)
    cluster_sizes = []
    label = 1
    for i in range(Lx):
        for j in range(Ly):
            if not visited[i, j]:
                size = flood_fill(lattice, i, j, label, clusters, visited)
                if size > 0:
                    cluster_sizes.append(size)
                label += 1
    return cluster_sizes, clusters

# -----------------------------------------------------------------
# Plotting the evolution of cluster sizes
cluster_sizes_over_time = []

# Initialize the figure
plt.figure(figsize=(10, 8))

# Plotting the evolution of cluster sizes
for key, lattice in lattices.items():
    cluster_sizes, cluster_map = analyze_clusters(lattice)
    cluster_sizes_over_time.append(cluster_sizes)

    # Calculate and plot cluster size distribution
    size_count = Counter(cluster_sizes)
    sizes, counts = zip(*sorted(size_count.items()))
    probabilities = np.array(counts) / np.sum(counts)

    plt.loglog(sizes, probabilities, '+', label=f'Step {key}')


steps = snapshot_intervals
average_sizes = [np.mean(s) if s else 0 for s in cluster_sizes_over_time]

# If the length of average_sizes is less than steps, pad it with zeros
if len(average_sizes) < len(steps):
    average_sizes.extend([0] * (len(steps) - len(average_sizes)))

plt.plot(steps, average_sizes, marker='x', linestyle='--', color='b', label='Average Cluster Size ⟨s⟩')

# Set titles and labels
plt.title('Cluster Size Distributions and Average Cluster Size Over Time')
plt.xlabel('Cluster size (s)')
plt.ylabel('P(s) / Average Cluster Size ⟨s⟩')
plt.grid(True)
plt.legend()
plt.show()


