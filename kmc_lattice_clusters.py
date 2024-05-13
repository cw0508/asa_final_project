import numpy as np
import random
import matplotlib.pyplot as plt
from collections import Counter

plt.style.use('plot_style.txt')
# plt.rcParams['figure.constrained_layout.use'] = True

# -----------------------------------------------------------------
# Variables:
# -----------------------------------------------------------------
# Parameters
Lx = 200        # Number of lattice sites along x-direction
Ly = 200        # Number of lattice sites along y-direction
t_nn = 1.0      # Nearest neighbor hopping constant
U = 2.0         # On-site Coulomb interaction strength
T = 0.1         # Temperature in units where kB = 1 (essentially scaling temperature)
kB = 1          # Boltzmann constant
mc_steps = 200000

# -----------------------------------------------------------------
# Compute intervals for snapshots, ensuring last step is included
snapshot_intervals = [int(mc_steps * i / 4) for i in range(3)] + [mc_steps - 1]

# Function to compute energy of the lattice
def compute_energy(lattice, t_nn, U):
    nn_energy = 0
    for i in range(Lx):
        for j in range(Ly):
            # Only sum over right and upper neighbor to avoid double-counting
            nn_energy += -t_nn * lattice[i, j] * (lattice[(i + 1) % Lx, j] + lattice[i, (j + 1) % Ly])
            # On-site interaction
            nn_energy += U * lattice[i, j] * lattice[i, j]
    return nn_energy


# -----------------------------------------------------------------
# Initialize lattice
initial_lattice = np.random.choice([-1, 1], size=(Lx, Ly))
lattices = {snapshot_intervals[0]: initial_lattice.copy()}  # Store initial state
current_energy = compute_energy(initial_lattice, t_nn, U)

# Function to compute energy change due to flipping an electron at site (i, j)
def energy_change(lattice, i, j, t_nn, U):
    delta_E = 2 * t_nn * lattice[i, j] * (lattice[(i - 1) % Lx, j] + lattice[(i + 1) % Lx, j] +
                                         lattice[i, (j - 1) % Ly] + lattice[i, (j + 1) % Ly])
    if lattice[i, j] == 1:
        delta_E += 2 * U
    return delta_E


# -----------------------------------------------------------------
# Initialize lattice
lattice = np.random.choice([-1, 1], size=(Lx, Ly))
lattices = {snapshot_intervals[0]: lattice.copy()}  # Store initial state

# Monte Carlo simulation using Metropolis-Hastings algorithm
energy_values = []
current_energy = compute_energy(lattice, t_nn, U)

for step in range(mc_steps):
    i, j = np.random.randint(0, Lx), np.random.randint(0, Ly)
    delta_E = energy_change(lattice, i, j, t_nn, U)
    if random.random() < np.exp(-delta_E / (kB * T)):
        lattice[i, j] *= -1
        current_energy += delta_E
    energy_values.append(current_energy)
    if step in snapshot_intervals:
        lattices[step] = lattice.copy()


# -----------------------------------------------------------------
# Analyze the clusters:
# -----------------------------------------------------------------
# Function to perform flood-fill algorithm for cluster detection
def flood_fill(lattice, x, y, label, clusters, visited):
    stack = [(x, y)]
    while stack:
        x, y = stack.pop()
        if visited[x, y]:
            continue
        visited[x, y] = True
        clusters[x, y] = label
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = (x + dx) % Lx, (y + dy) % Ly
            if lattice[x, y] == lattice[nx, ny] and not visited[nx, ny]:
                stack.append((nx, ny))

# -----------------------------------------------------------------
def analyze_clusters(lattice):
    visited = np.zeros_like(lattice, dtype=bool)
    clusters = np.zeros_like(lattice, dtype=int)
    label = 1
    for i in range(Lx):
        for j in range(Ly):
            if not visited[i, j]:
                flood_fill(lattice, i, j, label, clusters, visited)
                label += 1
    return clusters


# -----------------------------------------------------------------
# Plotting Combined:
# -----------------------------------------------------------------
# Analyze clusters for each snapshot
fig, axs = plt.subplots(2, len(snapshot_intervals), figsize=(12, 6))
for idx, key in enumerate(sorted(lattices.keys())):
    lattice = lattices[key]
    clusters = analyze_clusters(lattice)

    lattice_plot = axs[0, idx].imshow(lattice, cmap='viridis')
    axs[0, idx].set_title(f'Lattice @ Step {key}')
    axs[0, idx].set_xlabel('x')
    axs[0, idx].set_ylabel('y')


    cluster_plot = axs[1, idx].imshow(clusters, cmap='viridis')
    axs[1, idx].set_title(f'Cluster Map @ Step {key}')
    axs[1, idx].set_xlabel('x')
    axs[1, idx].set_ylabel('y')


fig.subplots_adjust(right=2.25)
cbar_ax1 = fig.add_axes([0.87, 0.55, 0.02, 0.35])  
cbar_ax2 = fig.add_axes([0.87, 0.10, 0.02, 0.35])  

cbar_lattice = fig.colorbar(lattice_plot, cax=cbar_ax1)
cbar_lattice.set_label('Spin Value')
cbar_cluster = fig.colorbar(cluster_plot, cax=cbar_ax2)
cbar_cluster.set_label('Cluster ID')

fig.suptitle(f'Kinetic Monte Carlo (kMC) Simulation of 2D Falicov-Kimball (FK) Model @ Temp = {T} [Kelvin (K)]')

plt.tight_layout()
plt.show()

