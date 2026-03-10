# 1D Inversion

This repo contains code for a toy problem that generates inversions of a simple 1D atmospheric transport model. This is a simplified version of the code base developed to support the theoretical framework described in:

> Nesser, H., Bowman, K. W., Thill, M. D., Varon, D. J., Randles, C. A., Tewari, A., Cardoso-Saldaña, F. J., Reidy, E., Maasakkers, J. D., and Jacob, D. J.: **Predicting and correcting the influence of boundary conditions in regional inverse analyses**, *Geoscientific Model Development*, 18, 9279–9291, https://doi.org/10.5194/gmd-18-9279-2025, 2025.

## Repository Structure

```
1D_inversion/
├── run.py          # Main entry point for running inversion experiments
├── config.yaml     # Configuration file for inversion parameters
└── src/            # Source code for the 1D transport model and inversion routines
```

---

## Getting Started

### Prerequisites

- Python 3.x
- Standard scientific Python libraries (e.g., NumPy, SciPy)

### Installation

Clone the repository:

```bash
git clone https://github.com/hannahnesser/1D_inversion.git
cd 1D_inversion
```

Create and activate the conda environment using the provided `environment.yml`:

```bash
conda env create -f environment.yml
conda activate 1D_inversion
```

### Running the Model

Inversion parameters (number of grid cells, wind speed, prior error standard deviation, observing system error, BC perturbation magnitude, etc.) are set in `config.yaml`. Once configured, run:

```bash
python run.py
```

---

## Citation

If you use this code in your research, please cite:

```
Nesser, H., Bowman, K. W., Thill, M. D., Varon, D. J., Randles, C. A., Tewari, A.,
Cardoso-Saldaña, F. J., Reidy, E., Maasakkers, J. D., and Jacob, D. J.:
Predicting and correcting the influence of boundary conditions in regional inverse analyses,
Geosci. Model Dev., 18, 9279–9291, https://doi.org/10.5194/gmd-18-9279-2025, 2025.
```

The full code and data archive for the paper's simulation experiments (including the two-dimensional Permian Basin demonstration) is also available on Zenodo:
> Nesser, H.: Boundary condition sensitivity simulation experiments (v2.0), Zenodo, https://doi.org/10.5281/zenodo.17417750, 2025.
