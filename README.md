# Stochastic Reforestation Planning

An optimization framework for reforestation planning that integrates Mixed-Integer Linear Programming (MILP), Evolutionary Game Theory, and Genetic Algorithms to optimize tree species allocation while minimizing ecological competition.

## Project Overview

This project addresses the challenge of assigning tree species to forest grid locations in a way that:
- Minimizes intra-species competition through game-theoretic modeling
- Meets target species composition requirements
- Considers spatial adjacency effects between planted species
- Compares multiple optimization approaches (exact solvers vs. heuristic methods)

## Project Structure

```
stochastic-reforestation-planning/
├── README.md                          # This file
├── W_matrixGT/                        # Ecological Game Theory analysis
│   ├── eco_game.ipynb                # Game theory ecosystem modeling
│   ├── w_matrix.csv                  # Competition matrix between species
│   └── species_traits_base.csv       # Base species characteristics
├── solverfinal/                       # Exact solver implementation
│   ├── solverfinal.ipynb             # Pyomo-based MILP solver
│   ├── species_targets.csv           # Target species composition
│   ├── w_matrix.csv                  # Competition weights
│   └── {49,64,81,100,121,144,625}/   # Results for grids of different sizes
├── heuristicofinal/                   # Genetic algorithm approach
│   ├── hfinal.ipynb                  # GA-based optimization
│   ├── species_targets.csv           # Target species composition
│   ├── w_matrix.csv                  # Competition weights
│   └── {49,64,81,100,121,144,625}/   # GA results for different grid sizes
├── iteraciones_final/                 # Iterative refinement approach
│   ├── iterfinal.ipynb               # Iterative optimization solver
│   ├── solverTmatrix.ipynb           # T-matrix based solver
│   ├── heuristic_helper.py           # Helper functions
│   ├── sample_nodes.py               # Node sampling utilities
│   ├── T_matrix.csv                  # Transition matrix
│   ├── nodes_100.csv                 # Spatial node coordinates
│   ├── w_matrix.csv                  # Competition matrix
│   ├── species_targets.csv           # Target composition
│   └── results_*.csv                 # Analysis results
└── simuladorfinal/                    # Simulation and validation
    ├── sf.ipynb                       # Simulator framework
    ├── comousarelhelper.ipynb        # Visualization helpers
    ├── sample_nodes.py               # Node generation
    └── nodes_{size}.csv              # Grid node definitions
```

## Key Components

### 1. **Ecological Game Theory (W_matrixGT/)**
- Models species interactions as an evolutionary game
- Generates competition matrices using game-theoretic principles
- Analyzes ecosystem equilibrium and convergence properties

### 2. **Exact Solver (solverfinal/)**
- Implements a Mixed-Integer Linear Program using Pyomo
- Solves the species assignment problem optimally
- Tests multiple grid sizes: 49, 64, 81, 100, 121, 144, 625 nodes
- Outputs: optimal assignments, cost breakdowns, composition validation

### 3. **Genetic Algorithm (heuristicofinal/)**
- Provides a scalable heuristic approach for large instances
- Uses evolutionary optimization to approximate optimal solutions
- Comparative analysis against exact solver results
- Faster computation for larger grid sizes

### 4. **Iterative Refinement (iteraciones_final/)**
- Implements iterative improvement strategies
- T-matrix based constraint handling
- Helper utilities for solution refinement
- Statistical analysis of convergence properties

### 5. **Simulation Framework (simuladorfinal/)**
- Validates solutions through ecological simulation
- Visualizes spatial distribution of species
- Tests solution robustness and ecological stability

## Data Files

Each experiment includes:
- **nodes_*.csv**: Grid coordinates and spatial positions
- **adjacency.csv**: Adjacency matrix for spatial constraints
- **assignments_*.csv**: Optimal or GA-generated species assignments
- **w_matrix.csv**: Competition interaction matrix (species × species)
- **species_targets.csv**: Target abundance for each species
- **composition_*.csv**: Composition analysis (expected vs. actual)
- **results_*.csv**: Performance metrics and cost breakdown

## Usage

### Running the Exact Solver
1. Open `solverfinal/solverfinal.ipynb`
2. Define grid size and input parameters
3. Execute cells to run the MILP optimization
4. Review results in the corresponding size folder (e.g., `solverfinal/100/`)

### Running the Genetic Algorithm
1. Open `heuristicofinal/hfinal.ipynb`
2. Set GA parameters (population size, generations, mutation rate)
3. Execute to find near-optimal solutions
4. Compare results with exact solver outputs

### Iterative Optimization
1. Open `iteraciones_final/iterfinal.ipynb`
2. Use iterative refinement for improved solutions
3. Analyze convergence and solution quality

### Simulation & Visualization
1. Open `simuladorfinal/sf.ipynb`
2. Load validated assignments
3. Visualize and analyze ecological outcomes

## Dependencies

- Python 3.x
- Pyomo (for MILP solving)
- NumPy, Pandas (data manipulation)
- Matplotlib, Seaborn (visualization)
- CPLEX or open-source solver (CBC/Glpk) for Pyomo
- Genetic algorithm libraries (DEAP or similar)

## Key Metrics

- **Cost**: Total ecological competition cost
- **Composition Error**: Deviation from target species distribution
- **Spatial Balance**: Distribution uniformity across grid
- **Convergence**: Solution quality across iterations/generations

## Results

The framework enables comparison between:
- **Exact solutions** (computationally expensive, optimal)
- **Heuristic solutions** (fast, near-optimal)
- **Iterative refinements** (balanced approach)

Grid sizes tested: 7×7 (49), 8×8 (64), 9×9 (81), 10×10 (100), 11×11 (121), 12×12 (144), 25×25 (625)

## Authors

Project made by JorgeAndujoV, marielalvarez, anasponce and VivianaCL
