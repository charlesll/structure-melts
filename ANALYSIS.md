# Current analysis and numerical conventions

This guide describes the analysis methods and numerical conventions used in this publication package.

## Inputs and configuration

The six runs are defined in `src/config.py`; their raw acquisitions are in
`data/` and their workbooks in `metadata/`. Workbook sheets are aligned to the
Glass sheet by temperature, membrane pressure and repeated-key occurrence,
not simply by row number. Experimental states come from the workbooks.
`INPUTS.sha256` records 939 input files, including the eight IR files
supplied under `data/IR/`.

The current Raman dataset contains 62 acquisitions: five room-temperature
compression points in Run 1 and heating/cooling acquisitions in Runs 2–6.
There are 22 acquisitions labelled exactly `Melt`, and 52 labelled `Verre`
or `Melt`; crystallised `Melt+Ctx` acquisitions are distinct.

The confirmed composition is SiO2/Al2O3/Na2O/H2O =
58.425/11.875/24.70/5.00 wt%, summing to 100. It is passed to gpvisc as weight
percentages, then converted to oxide mole fractions by that package.
The hydrous starting-glass NBO/T reference line is **0.59**, stored as
`ANN_HYDROUS_GLASS_NBO_T`; it is a comparison line, not a constraint on the
Raman-band fits or pressure tests.

The independent limiting fictive-temperature reference is **219 ±4 °C at
25 K/min**, for the as-received glass after piston-cylinder synthesis and
subsequently loaded into the DAC. `ANN_FICTIVE_REFERENCE` stores its value and
provenance. Figure 7 shows one point at ambient pressure (plotted at 0 GPa),
labelled `T_f' at 25 K/min`. The ±4 °C half-width is drawn as supplied; no
confidence level is assumed.

## Reproduction

Use Python ≥3.12 and the pinned dependencies in `requirements-lock.txt`.
From the repository root, `make test PYTHON=.venv/bin/python` runs the entire
production pipeline, centroid diagnostics, IR processing and tests. No stored
results are required. `make figures` rebuilds the main figure chain but does
not itself run the tests, centroid diagnostics or IR analysis.

Individual entry points are `python -m src.extract_markers --no-fig`,
`src.pressure`, `src.peak_fitting`, `src.temperature`, `src.manuscript_data`,
`src.ann_viscosity`, `src.figures`, `src.centroid_lf` and `src.ir_treatment`.
For a single extraction run, use e.g.
`python -m src.extract_markers run02_20250220 --no-fig`.
The Makefile has no RUNS variable. Direct figure commands use existing results;
refresh upstream calculations when changing inputs or methods.

## Pressure markers and uncertainty

Both gauges use Wei2024. Wavelengths are in nm, temperatures in K for the
calibration functions, and pressures in GPa. The adopted ambient references
are ruby R1 **694.31223 nm** and borate **685.4505001123697 nm**, recorded in
`src/config.py`. The 23 °C reference room is approximated as 296 K; filenames containing 25C and workbook temperatures retain their recorded labels.

Ruby positions use the detected R1 maximum; borate positions use the centre
of a pseudo-Voigt plus linear-background fit. The detection-width threshold is
separate from the measured FWHM. FWHM uses interpolated crossings at half the
background-corrected peak height. It sets the initial width, fit window and
bounds for the borate fit. A non-converged fit is warned about, counted and
left missing, without replacing its fitted centre by a detected maximum.

Pressure is evaluated at the recorded thermocouple temperature by default.
`P_std = sqrt(sample variance between grains + mean(propagated grain error²))`.
It is a descriptive spread, **not the standard error of the median**.
Calibration terms are shared between grains and are not divided by grain count.
The default production pressure calculation sets thermocouple and reference
wavelength uncertainties to zero; reported grain errors include line-position
errors when available and the implemented calibration-coefficient errors.
Full systematic and cross-acquisition covariance is not characterised.

The plotted temperature half-width is **±2 K**, from `SIGMA_T`. It excludes
the unresolved sample–thermocouple offset. Neither `T_cor` in dataset.h5 nor
the exact offsets reported by `src.temperature` are applied to the pressures.

## Raman treatment and band fitting

LF spectra are despiked, corrected using the measured background and placed
on the common **400–1279 cm^-1** grid. The Long correction is applied at the
recorded temperature. A missing HF acquisition does not remove a valid LF
spectrum. The room-pressure run01 LF file is explicitly identified as already
being in Raman-shift units. The starting-glass reference for Figure 4 is
explicitly `ANN_5prctH2O_LF.asc`, not the first file in a directory containing
several compositions.

Band fitting uses **820–1270 cm^-1**, a linear endpoint baseline and maximum
normalisation. Four Gaussian bands (T–OH, Q2, T2S, Q4) and one Q3 pseudo-Voigt
are fitted with the Gaussian priors listed in `src/peak_fitting.py`. Noise is
estimated from the residual to Gaussian smoothing (sigma 12 channels), with
floor 1e-4. Priors influence the fitted values; they are not just initial guesses.
Nonphysical fitted means are rejected. Parameter, area and area-fraction
covariances are saved. A failed band fit prevents replacement of peak_fits.h5.

These errors are local, conditional on the five-band model, baseline and noise.
Band intensities are not direct measurements of species populations.
Area fractions use all five fitted areas as their denominator.
NBO/T = 2 c2 f_Q2 + c3 f_Q3, with coefficients **0.84 ±0.07** and
**1.06 ±0.04**, attributed to Mysen (2007), Table 4 in `src/config.py`.
Their SDs are treated as independent and shared across acquisitions; fit
covariances are propagated with gvar. No extra renormalisation is applied.
Figure 8 pressure tests use the resulting full covariance but treat pressure
as fixed, without separating temperature/run effects; the 7 GPa threshold is fixed.

For Figure 10 and the rupture analysis, `prepare_glass` applies a two-part
linear baseline and area normalisation, retaining signed residual intensities.
Figure 10 uses a discrete signed LF centroid for x <830 cm^-1. Centroid
diagnostics instead use trapezoid integration on inclusive windows; their
pressure plot always uses **830–1270 cm^-1**. These definitions are explicit
and intentionally distinct.

## Common spectral-change search

Each individual variable is centred/scaled on the fitted observations.
Within each family, squared line-fit residuals are averaged across variables;
the five family costs are then added with equal weights:

| Family | Variables |
| --- | --- |
| General Raman PCA | Four scores from the prepared full Raman spectra |
| Raman area ratio | A(400–830)/A(830–1270) |
| Fitted-band area ratios | Three log-ratios: T2S/Q4, TOH/T2S, Q2/Q3 |
| Fitted-band positions | T–OH, Q2, T2S, Q3, Q4 |
| Sensors | Nine variables described below |

For each gauge, the sensor family contains two PCA scores of mean recentered
line shapes, RMS between-grain shape dispersion, and P_std. The ninth variable
is ruby median pressure minus borate median pressure. FWHM summaries are saved
in the dataset but are not added to this production rupture method.

Global Raman PCA learns from all six runs and all 62 acquisitions, including
crystallised and cooling observations, before the split-fit selection. Sensor
PCA similarly learns globally from valid mean grain shapes. Both use
`random_state=0`. Sensor shapes are recentered without rescaling their widths;
windows are ±0.8 nm for ruby and ±0.35 nm for borate, with 121 points each.
The stored borate grids cover every selected window: 686–690 nm for Run 5
and 685.5–689 nm for Run 6 (upper bounds excluded). Alignment rejects any finite window extending outside the
stored grid.
Shape dispersion is the root mean squared deviation from the mean shape,
over all valid grains and channels, using population division. Fewer than
two valid grains gives missing dispersion.

Split fitting retains non-crystallised heating observations at **T ≥150 °C**,
with at least three observations on each side. A single common split is
searched, with independent straight lines on its two sides; continuity is
not imposed. A partition alone does not demonstrate a transition.

Linear and quadratic nulls each receive 999 residual-row permutations,
with seeds 0 and 1. Rows are permuted jointly within a family, independently
between families. The null is refitted and the split re-searched each time.
This preserves within-family dependence, not dependence between families.
Residual exchangeability is assumed; there is no correction across runs.
A split is labelled supported only if **both p-values are strictly below 0.05**.
The five observed brackets are **262–286, 270–277, 256–262, 313–332,
300–306 °C** for Runs 2–6. Run 3 remains a candidate. Brackets are adjacent
sampled acquisitions, not confidence intervals or independently established Tg.

Figure 6 and S7–S10 show individual input channels in grey, a descriptive
channel mean in blue and crystallised observations as open red diamonds.
The mean is not used by the split cost. Vertical display limits use only the
fitted glass/melt channels; crystallised points outside the range are clipped. This display choice does not alter the analysis.

## Pressure dependence of the spectral brackets

`src/tg_pressure_regression.py` fits T = a + bP to the five brackets, including
the Run 3 candidate. The likelihood integrates a normal latent residual over
each observed bracket. Pressure spread is represented by an explicit Gaussian
sensitivity proxy combining maximum endpoint P_std and half endpoint drift.
The thermocouple half-width enters separately. The independent limiting fictive
temperature reference does not constrain the fit.

Current values are approximately **a =239.7 °C**, **b =6.93 °C/GPa**,
with intercept profile support **[210.9, 270.6] °C**. The profile threshold
is Δ(-2 log L)=3.841458820694124. This is approximate asymptotic 95% support,
not calibrated for exact coverage with five runs or for prediction of a new
acquisition. `src/figure_data.py` retains the nominal reference and profile;
a consistency test checks them against `results/fig7_interval_regression.json`.

Figure 7 and S6 share exactly the same fit. Figure 9 subtracts the fixed
nominal intercept from absolute mean-fit profiles. Its band can therefore have
finite width at 0 GPa; it is not the within-line random difference bP, which
would be exactly zero there. Albite/anorthite values are supplied comparison
series transcribed in the source; their shared-reference covariance is retained.
The peridotite curve is a supplied descriptive model.

## Viscosity scenarios

GPvisc **0.3.5**, model **1**, CPU, predicts with the full GP including
likelihood. The model mean is inverted at log10(eta/Pa s)=12 on 53 pressures
from 0 to 13 GPa. A 250–2000 K scan finds one descending crossing, followed
by bisection with a 0.002 K tolerance and a residual check of 0.002 log units.
The resulting mean isoviscous temperatures are about **250.73 °C** and
**405.14 °C** at the endpoints. Small numerical differences can occur between
executions. Predictive viscosity SD is not converted into a Tg confidence band.

S11 compares direct GP predictions with MYEGA scenarios at 1400, 1500 and
1600 °C. MYEGA uses the current spectral T(P) regression, **Ae=-1.48** and
**m=34.5 ±1.2**, with the latter interpreted as a Gaussian SD. Its shaded
95% interval transforms fragility quantiles only: Ae, T(P) and model error
are not varied. Neither set of curves is a viscosity measurement. Optional
`src.ann_viscosity --check-models` compares other supplied GP models; it is
not part of the default figure chain.

## Near-infrared speciation

`src.ir_treatment` reads the three samples in `data/IR/ANN_IR_list.csv`.
It uses a sixth-order baseline, a 2 cm^-1 resampling grid, isolated-spike
removal and Savitzky–Golay smoothing (31 points, order 3). It measures heights
near 4500 and 5200 cm^-1; integrated areas are diagnostic outputs only.

The adopted NS4 row is selected by the composition labels in
`IR_EXTINCTION_REFERENCE_COMPOSITION`. Coefficients are read directly from
`data/IR/IR_extinction_coeffs.csv`: **epsilon4500=0.44 ±0.03** and
**epsilon5200=0.64 ±0.03 L mol^-1 cm^-1**, with standard uncertainties.
Absorbance is decadic; thickness is converted from micrometres to cm;
density is in kg/m^3 and concentrations are wt% H2O equivalent.

Gvar propagates height, thickness, density and coefficient errors. Thickness
and density are shared between bands, and coefficients are shared across
samples. Baseline uncertainty, spectral-noise correlation and transfer of the
NS4 coefficients to the samples are not fully characterised. Outputs are
`results/IR/IR_results.csv`, `results/IR/IR_metadata.json` and two diagnostic
PNG/PDF pairs under `figures/IR/`.

## Figure numbers and scope

Main figures 2–4 and 6–10 and supplementary figures S1–S11, including S5,
are generated by this package. See [FIGURES.md](FIGURES.md).

S5 reads three selected raw Ctx spectra from sessions 20250220, 20250303 and
20250408. It applies a cubic baseline over 500–520, 800–840 and 1200–1300 cm⁻¹,
then normalises each corrected spectrum to its maximum. Peak markers and
temperature/pressure annotations are retained from the supplied script; they
are not updated by the pressure measurements code. Its JSON records the
input files and processing parameters. Run it alone with `python -m src.manuscript.supplementary_fig_5`
or `python -m src.figures Supplementary_Fig_5`.