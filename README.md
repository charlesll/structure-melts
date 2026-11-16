# Structural and dynamical properties of hydrous magmatic liquids at high pressure

[![DOI](https://zenodo.org/badge/1352435143.svg)](https://doi.org/10.5281/zenodo.22209627)

This repository contains the data and Python code associated with the publication:

> Lauréna Mourand, Clément Ferraina, Salomé Pannefieu, Kai-Uwe Hess, Cyril Aubaud, Charles Le Losq (2026).
> *Structural and dynamical properties of hydrous magmatic liquids at high pressure: insights from Raman spectroscopy observations during diamond anvil cell experiments.*
> Submitted to Chemical Geology.

## Installation and reproduction

Use Python 3.12 or newer and GNU Make. From this directory:

```sh
python -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
make test PYTHON=.venv/bin/python
```

requirements.txt lists direct dependencies; requirements-lock.txt records the source environment. GPvisc 0.3.5 uses pretrained model 1 on CPU; model assets are supplied through that package. Installation may require network access. NumPy/SciPy handle numerics, pandas/openpyxl read metadata, h5py stores arrays, Matplotlib/cmcrameri render figures, rampy processes Raman spectra, ruptures detects splits, and gvar/lsqfit propagate and fit uncertainties.

## Layout and commands

Raw ASC acquisitions are in data/, workbooks in metadata/, numerical outputs in results/, and PDF/PNG figures and numerical sidecars in figures/manuscript/. INPUTS.sha256 records immutable input checksums. No original results are required.

make figures runs all upstream stages. Generated main figures are 2–4 and 6–10; the inserted DSC Figure 5 is supplied separately in the manuscript. Supplementary figures S1–S11 are generated, including the crystallised spectra in S5. make test also runs centroid diagnostics and IR processing before the tests. Individual targets: extract, pressure, peaks, temperature, transitions, viscosity, centroid, ir. Override the interpreter with PYTHON. make clean deletes generated results/, figures/ and Python caches, preserving sources and inputs. Direct `python -m src.figures` uses existing results; figure names may be supplied as arguments. Figure 7 also generates S6 from the same fit.

## Source inventory

| Module | Purpose |
| --- | --- |
| `src/__init__.py` | DAC glass-transition pipeline: pressure markers, pressure, Tg and Raman analysis. |
| `src/ann_viscosity.py` | ANN viscosity calculations and predictions for Supplementary Figure 11. |
| `src/raman_treatment.py` | Prétraitement Raman, centroïdes, rapports d’aires et tests GLS du NBO/T ; fonctions documentées en français. |
| `src/centroid_lf.py` | Raman centroids versus temperature and Melt 830-1270 cm-1 centroid versus P. |
| `src/config.py` | Run registry and paths. |
| `src/extract_markers.py` | Stage 1: fit Ruby / SrB4O7:Sm2+ lines and clean the glass Raman spectra. |
| `src/figure_data.py` | Shared data assembly for the manuscript figures (Python port of Processing.jl). |
| `src/figures/__init__.py` | Manuscript figures, regenerated in Python from `results/`. |
| `src/figures/__main__.py` | Regenerate all main and supplementary manuscript figures. |
| `src/figures/fig2_pt.py` | Figure 2 : diagnostics des deux jauges à la température du thermocouple. |
| `src/figures/fig3_raman.py` | Figure 3: corrected silicate and O-H spectra during manuscript Run 5. |
| `src/figures/fig4_comparison.py` | Figure 4: starting glass, cold compression, melt and quenched HP glass. |
| `src/figures/fig6_ruptures.py` | Figure 6: common-break analysis of Run 4, with crystals shown for context. |
| `src/figures/fig7_transitions.py` | Figure 7: P–T paths, split brackets and conditional ambient extrapolation. |
| `src/figures/fig8_melt_areas.py` | Figure8 : fractions d'aires vs P et NBO/T vs T/P, sélection exacte Melt. |
| `src/figures/fig9_delta_tg.py` | Figure 9: pressure-induced Tg changes relative to each series' reference. |
| `src/figures/fig10_raman.py` | Figure 10: LF centroid versus pressure and fitted T2S/Q4 area ratio. |
| `src/figures/supplementary_fig1.py` | Two existing Raman deconvolutions closest to 3 and 12 GPa. |
| `src/figures/supplementary_fig2.py` | Publication version of fitted Raman band positions versus borate pressure. |
| `src/figures/supplementary_fig3.py` | Publication version of fitted Raman band areas versus borate pressure. |
| `src/manuscript/supplementary_fig_5.py` | Supplementary Figure 5: selected crystallised spectra, cubic baseline and maximum normalisation. |
| `src/figures/supplementary_fig4.py` | Supplementary Figure 4: fitted band area fractions. |
| `src/figures/supplementary_fig7.py`–`supplementary_fig10.py` | Common-break analyses for Runs 2, 3, 5 and 6 (S7–S10), sharing Figure 6 code. |
| `src/figures/supplementary_fig6.py` | Supplementary Figure 6: absolute temperature–pressure interval regression. |
| `src/functions.py` | Shared implementation. |
| `src/ir_treatment.py` | Near-infrared water speciation; adopted coefficients read from the supplied CSV table. |
| `src/line_fit.py` | Pseudo-Voigt fit of a fluorescence line on the raw spectrum. |
| `src/manuscript/Supplementary_Figure_11_viscosity.py` | Supplementary Figure 11: pressure dependence of ANN melt viscosity. |
| `src/manuscript/__init__.py` | Additional manuscript figure scripts. |
| `src/manuscript_data.py` | Données et incertitudes communes aux figures 2 et 7–10 et la figure supplémentaire 4. |
| `src/peak_fitting.py` | Gaussian-prior Raman peak fitting for the silicate region. |
| `src/peak_uncertainty.py` | Propagation gvar des aires corrélées (ordre TOH,Q2,T2S,Q3,Q4). |
| `src/pressure.py` | Pressure from ruby and borate grains at the sample temperature. |
| `src/pressure_sensors.py` | Ruby R1 and SrB4O7:Sm2+ 0-0 optical pressure gauges. |
| `src/raman_observables.py` | Aligned Raman observables for manuscript Figure 10 and its model checks. |
| `src/temperature.py` | Offsets équivalents par acquisition et statistiques de la figure 2d. |
| `src/tg_pressure_regression.py` | Descriptive interval regression of spectral split brackets against P. |
| `src/tg_spectra_ruptures.py` | Exploratory common-break detection for Raman PCA, Qn, and sensor series. |

## License

The data and code in this repository are released under the [MIT](LICENSE).

## Citation

If you use this dataset or code, please cite the associated publication (see above) and the Zenodo archive DOI for this repository.
