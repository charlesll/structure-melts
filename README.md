# Structural and dynamical properties of hydrous magmatic liquids at high pressure

This repository contains the data and Julia code associated with the publication:

> Lauréna Mourand, Clément Ferraina, Salomé Pannefieu, Cyril Aubaud, Charles Le Losq (2026).
> *Structural and dynamical properties of hydrous magmatic liquids at high pressure: insights from Raman spectroscopy observations during diamond anvil cell experiments.*
> Submitted to Chemical Geology.

## Contents

- **`dataset_spectra_T_P.h5`**: HDF5 dataset containing the 61 in-situ Raman spectra collected during the diamond anvil cell experiments, together with the associated pressure/temperature conditions and pressure marker calibrations. Datasets included:
  - `x_axis`: Raman shift axis (cm⁻¹, 400–1279 cm⁻¹)
  - `spectra`: Raman intensities, shape (61 spectra × 880 points)
  - `T`: temperature (°C)
  - `P`: pressure (GPa)
  - `Etat`: sample state for each spectrum (`Verre` = glass, `Melt`, `Melt+Ctx`)
  - `exp_nb`: experiment number
  - `Ruby_P_median`, `Ruby_P_std`: pressure from the ruby fluorescence marker (median, standard deviation)
  - `Borate_P_median`, `Borate_P_std`: pressure from the borate fluorescence marker (median, standard deviation)
  - `Ruby_position_median`, `Borate_position_median`: median peak positions of the pressure markers
- **`ANN_5prctH2O_LF.asc`**: reference low-frequency Raman spectrum (25 °C, 1 atm starting glass, 5 wt% H2O).
- **`Processing.jl`**: Julia script that reproduces the data processing, figures, and analyses presented in the paper (baseline removal, despiking, peak fitting, Qn speciation, comparison of glass/melt/quench spectra, etc.).

## Reproducing the results

The code was developed and tested with **Julia 1.11 and 1.12**.

Required Julia packages:

- [Spectra.jl](https://github.com/charlesll/Spectra.jl)
- HDF5
- Statistics
- Measurements
- MultivariateStats
- DataFrames
- CSV
- LaTeXStrings
- RecipesBase
- ColorSchemes
- LsqFit
- Plots
- CairoMakie
- Typst_jll

Install them from the Julia REPL:

```julia
using Pkg
Pkg.add(["Spectra", "HDF5", "Measurements", "MultivariateStats", "DataFrames",
         "CSV", "LaTeXStrings", "RecipesBase", "ColorSchemes", "LsqFit",
         "Plots", "CairoMakie", "Typst_jll"])
```

Then run the script from this folder:

```julia
julia Processing.jl
```

Figures are saved in the `figures/` subfolder created at runtime.

## License

The data and code in this repository are released under the [MIT](LICENSE).

## Citation

If you use this dataset or code, please cite the associated publication (see above) and the Zenodo archive DOI for this repository.
