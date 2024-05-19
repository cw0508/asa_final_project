# Kinetic Monte Carlo Simulations on the Falicov-Kimball Model

## Project Overview
This project implements Kinetic Monte Carlo (kMC) simulations to study the dynamics of phase separation in correlated electron systems, particularly using the Falicov-Kimball model. The simulations focus on understanding how temperature and effective potential affect the dynamics and stability of electron clusters, based on the paper: https://www.pnas.org/doi/epdf/10.1073/pnas.2119957119

## Features
- **Kinetic Monte Carlo Simulations**: Implements kMC to model electron dynamics.
- **Temperature Variations**: Analyzes how different temperatures impact the system behavior.
- **Cluster Formation Analysis**: Studies the sizes and probabilities of cluster formations at various stages of the simulation.

## Installation
To run the simulation scripts, you need Python installed on your machine along with some additional libraries.

### Prerequisites
- Python 3.8+
- NumPy
- random
- collections
- scipy
- Matplotlib (for generating visualizations)

### Setup
Clone the repository and install the required Python packages:
```bash
git clone https://github.com/cw0508/asa_final_project.git
cd asa_final_project
pip install -r requirements.txt
