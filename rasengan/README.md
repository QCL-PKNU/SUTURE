# Rasengan: A Transition Hamiltonian-based Approximation Algorithm for Solving Constrained Binary Optimization Problems

## Overview

This repository consists of three main components:

* **`env_setup/`**: Conda environment configuration files
* **`rasengan/`**: Source code for the Rasengan algorithm
* **`reproduce/`**: Scripts and notebooks for reproducing the experiments in the paper

## Installation

### 1. Set up the Conda environment

Navigate to the environment setup directory:

```bash
cd env_setup
```

Install the CPU-compatible environment:

```bash
conda env create -f cpu-env.yml
```

Activate the environment:

```bash
conda activate rasengan
```

If the installation fails or is interrupted, remove the environment and reinstall:

```bash
conda remove -n rasengan --all
```

### 2. GPU Support (Optional)

Our method is accelerated using the DDSIM backend and already runs efficiently on CPU.  

GPU acceleration via `qiskit-aer-gpu` is optional and can speed up baseline simulations (e.g., HEA and QAOA), which involve many `RX` gates. However, the CPU environment alone is sufficient for reproducing all experimental results.

To enable GPU support on Linux, you must have CUDA 11.2 or later and a compatible GPU driver installed.

#### Option A: Preconfigured environment for CUDA 12.8

If CUDA 12.8 is available on your system, you can set up the environment directly with:

```bash
conda env create -f gpu-cuda12.8-env.yml
```

#### Option B: Manual installation for other CUDA versions

To add GPU support to an existing `rasengan` environment, simply install the GPU simulator with:

```bash
pip install qiskit-aer-gpu
```

## Verify Installation

Run the environment check script located in the `env_setup/` directory:

```bash
python env_check.py
```

If your environment is correctly configured, you will see:

```
✅ CPU environment configured successfully!
```

If GPU support is also correctly set up, you will additionally see:

```
✅ GPU environment configured successfully!
```

The programs in `reproduce/` will automatically detect whether GPU acceleration is available. No manual switching is required.

If the test fails, consider:

1. Ensuring the correct Conda environment is activated.
2. Making sure the Python environment is isolated from global site-packages. You may disable the user site by:

   ```bash
   export PYTHONNOUSERSITE=1
   ```

## Reproduce the Experiments

The original paper evaluates our algorithm on 20 benchmark problems, each with 100 cases. Some of these experiments require several days of continuous computation.

To reduce the cost of reproduction, the programs in the `reproduce/` directory have been appropriately scaled down by decreasing the number of benchmarks.
Additionally, the number of cases per benchmark is reduced from 100 to approximately 10, which may introduce some bias in the experimental results. The total runtime is approximately **40 hours on CPU**, or **25 hours with GPU acceleration**. Experiments were conducted on the following hardware:

- **OS**: Ubuntu 20.04 LTS
- **CPU**: Dual AMD EPYC 9554, 128 cores
- **GPU**: NVIDIA H100 (80 GB)
- **Memory**: 1.5 TB

### Directory Structure

Inside the `reproduce/` directory, you will find the following top-level items:

* **`figure_x/`** :
  Subdirectories corresponding to each figure in the paper. Each one represents a standalone experiment.

* **`table_2/`** :
  Subdirectories corresponding to Table 2 in the paper.

* **`paper_figs/`** :
  Screenshots of the original experimental figures from the published paper, provided for reference.

* **[`results_summary.ipynb`](./reproduce/results_summary.ipynb)** :
  Aggregates the reproduced results and compares them with the original figures. All plots have been pre-generated for quick inspection.

* **[`run_all_experiments.py`](./reproduce/run_all_experiments.py)** :
  A consolidated script that launches all reproduction experiments in sequence. It can be run in the background without requiring manual supervision, and automatically writes all output data and plots into the corresponding experiment subdirectories.

Each of the `figure_x/` and `table_2/` subdirectories includes the following files:

* `run_and_plot.ipynb` :
  The main notebook that performs the experiments and generates plots.

* `run_backend.py` :
  A standalone Python script that provides a non-interactive version of the notebook, containing the same experiment logic. Suitable for batch runs or headless execution without Jupyter.

* `only_plot.ipynb` :
  A lightweight notebook used solely for generating plots from existing result files.

### Recommended Reproduction Workflows

**Option 1:**
Run the consolidated script `run_all_experiments.py` to execute all experiments in one go. After completion, you may inspect the results using `only_plot.ipynb` in each subdirectory or check `results_summary.ipynb` for a global overview.

**Option 2:**
Navigate to each experiment subdirectory and run `run_and_plot.ipynb` or `run_backend.py` individually. Monitor CPU usage and adjust parallelism accordingly. For GPU-intensive experiments (e.g., QAOA or HEA), **we recommend running only one program at a time**, as sequential execution may be faster overall.

## Troubleshooting

This section summarizes known issues that may arise during execution and provides suggested workarounds.

### 1. `A process in the process pool was terminated abruptly while the future was running or pending.`

These errors may occur when **too many reproduction programs are executed simultaneously**, causing excessive load on the CPU or GPU.

**Solution:**
Run the reproduction programs **one at a time**, preferably in sequence. If the issue persists even with sequential execution, consider reducing parallelism by adjusting the `num_processes` variable in the script, or switching to a machine with more resources.

### 2. Output files are missing or not saved in the expected subdirectory

This usually happens when the script is executed from the wrong working directory, since output paths are **relative to the execution location**.

**Solution:**
Make sure to **run each script from within its corresponding subdirectory** (e.g., `figure_x/` or `table_2/`) so that output files are saved in the correct place.
