import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.signal import convolve2d
import os

desktop_path = os.path.join(os.path.expanduser('~'), 'Desktop')
plt.style.use('plot_style.txt')



# Parameters
Lx, Ly = 50, 50  # dimensions
T = 100        # Temperature
t_nn = 1.0       # Nearest neighbor hopping parameter
U = 2.0          # On-site interaction parameter
kB = 1.0         # Boltzmann constant

# Function to compute energy change due to flipping an electron at site (i, j)
def energy_change(lattice, i, j):
    delta_E = 2 * t_nn * lattice[i, j] * (lattice[(i - 1) % Lx, j] + lattice[(i + 1) % Lx, j] +
                                         lattice[i, (j - 1) % Ly] + lattice[i, (j + 1) % Ly])
    if lattice[i, j] == 1:
        delta_E += 2 * U
    return delta_E

# Initialize lattice
lattice = np.random.choice([-1, 1], size=(Lx, Ly))

# Define Monte Carlo update function with energy calculations
def mc_update(lattice, T):
    i, j = np.random.randint(0, Lx), np.random.randint(0, Ly)
    delta_E = energy_change(lattice, i, j)
    if delta_E <= 0 or random.random() < np.exp(-delta_E / (kB * T)):
        lattice[i, j] *= -1

# Function to calculate the potential
def calculate_potential(lattice):
    kernel = np.array([[0, 1, 0],
                       [1, 0, 1],
                       [0, 1, 0]])
    potential = convolve2d(lattice, kernel, mode='same', boundary='wrap')
    return potential

# Create figure and axes
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
im1 = ax1.imshow(lattice, cmap='viridis', animated=True)
im2 = ax2.imshow(calculate_potential(lattice), cmap='hot', animated=True)
ax1.set_title('Lattice Configuration')
ax2.set_title('Potential Distribution')

fig.suptitle(f'Dynamic Lattice State and Potential Visualization @ Temp = {T} [Kelvin (K)]', fontsize=14)

# Function to update the animation
def update(frame):
    for _ in range(100):  # Perform 100 MC updates per frame
        mc_update(lattice, T)
    im1.set_array(lattice)
    im2.set_array(calculate_potential(lattice))
    return im1, im2,

# Create animation
ani = FuncAnimation(fig, update, frames=250, interval=50, blit=True)
ani.save(os.path.join(desktop_path, 'kmc_temp1.mp4'), writer='ffmpeg')

# Show the animation
plt.show()



