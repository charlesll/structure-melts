"""Run registry and paths.

One `Run` = one heating series at (nearly) constant pressure, described by an
Excel table in `metadata/` (sheets Ruby, Borate, Glass, Tg) and a folder of
raw spectra in `data/`. Run numbers follow the manuscript (run01 ... run06).
"""
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
METADATA = ROOT / "metadata"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

# ANN melt composition supplied in wt%, summing to 100 including 5 wt% H2O.
# gpvisc converts these weight percentages to oxide mole fractions.
ANN_COMPOSITION_WT = dict(sio2=58.425, al2o3=11.875, na2o=24.70, h2o=5.00)
# Confirmed limiting fictive temperature of the as-received hydrous ANN glass
# after piston-cylinder synthesis, i.e. the material loaded into the DAC.
# The supplied ±4 °C is an error-bar half-width; no confidence level was given.
ANN_FICTIVE_REFERENCE = dict(
    temperature_C=219.0,
    error_half_width_C=4.0,
    heating_rate_K_min=25.0,
    material="ANN + 5 wt% H2O",
    thermal_history="As received after piston-cylinder synthesis; loaded into the DAC",
    uncertainty_note="Supplied ±4 °C; confidence level not specified",
)

# Confirmed reference line for the hydrous starting glass in Figure 8.
ANN_HYDROUS_GLASS_NBO_T = 0.59

# Composition labels identifying the adopted NS4 row in the supplied IR table.
# Extinction coefficients and their SDs are read from that row, not duplicated.
IR_EXTINCTION_REFERENCE_COMPOSITION = dict(SiO2=80.0, Al2O3=0.0, Na2O=20.0)

# i-Melt values supplied by user; fragility uncertainty interpreted as 1 sigma.
ANN_MYEGA_A = -1.48
ANN_MYEGA_FRAGILITY = 34.5
ANN_MYEGA_FRAGILITY_SD = 1.2

# Common Raman-shift axis (cm-1) of the silicate region used for all runs
RAMAN_AXIS = np.arange(400.0, 1280.0, 1.0)

@dataclass
class Run:
    """Description of one experimental series.

    Parameters
    ----------
    run_id : unique name, `runNN_YYYYMMDD[_tag]`; also the table file stem.
    session : sub-folder of `data/` holding the raw .asc spectra.
    is_tg_series : False for isothermal series (no Tg to detect).
    ruby, borate : keyword arguments passed to the line-fitting routine
        (cut windows, peak width...), see `src.functions.estimate_peak_parameters`.
    x_ruby, x_borate : (start, stop) of the common wavelength grids (nm)
        on which the cleaned lines are stored (step 0.01 nm).
    ambient_reference : optional T_C/Pmem plus one Ruby/Borate reference stem;
        fills only a missing ambient sensor row, without pseudo-replication.
    glass_axis_units : per-LF-file explicit unit overrides; defaults wavelength_nm.
        Use raman_cm1 only for a spectrum already exported as Raman shift.
    """
    run_id: str
    session: str
    is_tg_series: bool = True
    ruby: dict = field(default_factory=dict)
    borate: dict = field(default_factory=dict)
    x_ruby: tuple = (689.1, 705.0)
    x_borate: tuple = (685.5, 688.0)
    ambient_reference: dict = field(default_factory=dict)
    glass_axis_units: dict = field(default_factory=dict)

    @property
    def table(self):
        return METADATA / f"{self.run_id}.xlsx"

    @property
    def spectra_dir(self):
        return DATA / self.session

    @property
    def markers_file(self):
        return RESULTS / "markers" / f"{self.run_id}.h5"


# Isolate Ruby R1 at ambient pressure; otherwise the first detected line is R2.
REFERENCE_RUBY_PARAMS = dict(cut_lb=693.4, cut_hb=695.5,
                              find_peaks_width=4., smooth=False)

RUNS = [
    Run("run01_20250220_25C", "session_20250220_ANN", is_tg_series=False,
        ruby=dict(find_peaks_rel_height=0.9, find_peaks_width=100.0),
        x_borate=(685.0, 688.0),
        ambient_reference=dict(T_C=25., Pmem=1., Ruby='1bar_25C_Rub_ref',
                                Borate='1bar_25C_Bor_ref'),
        glass_axis_units={'1bar_25C_Glass_LF': 'raman_cm1'}),
    Run("run02_20250220", "session_20250220_ANN"),
    Run("run03_20250303_32bar", "session_20250303_ANN"),
    Run("run04_20250303_34bar", "session_20250303_ANN"),
    Run("run05_20250408", "session_20250408_ANN_5H2O",
        ruby=dict(cut_lb=691.0, cut_hb=720.0),
        borate=dict(cut_hb=692.0),
        # Cover every observed borate centre plus the ±0.35 nm PCA window.
        x_ruby=(690.0, 705.0), x_borate=(686.0, 690.0)),
    Run("run06_20250430", "session_20250430_ANN_5H2O",
        ruby=dict(cut_lb=691.0, cut_hb=715.0),
        x_borate=(685.5, 689.0)),
]

# Default line-fitting parameters (overridden per run through Run.ruby / Run.borate)
RUBY_DEFAULTS = dict(cut_lb=689.0, cut_hb=720.0, find_peaks_width=20.0,
                     bkg_width=0.5, find_peaks_rel_height=0.8,
                     Lambda_whittaker=10**2.5, polynomial_order=1)
BORATE_DEFAULTS = dict(cut_lb=684.0, cut_hb=690.0, find_peaks_width=4.5,
                       bkg_width=0.6, polynomial_order=1, smooth=False)

# Production uses Wei et al. (2024) for both gauges, with thermal shifts.
# Grain-level calibration uncertainty propagation is implemented for this
# family only; other supported nominal scales are not used by the pipeline.
# Keep one calibration family for the whole study.
RUBY_SCALE = "wei2024"
BORATE_SCALE = "wei2024"

# Ambient fluorescence references: Raman room air-conditioned at 23 °C. 
# The "25C" filenames/workbook labels
# are nominal historical labels, not the reference acquisition temperature.
# Adopt the measured wavelengths directly as the Wei 296 K anchors: 23 °C
# is approximated as 296 K. This does not relabel workbook T_C.
# Raw files: data/session_20250220_ANN/1bar_25C_{Rub,Bor}_ref.asc.
LBD_REFERENCE_ROOM_T_C = 23.0
LBD_REFERENCE_T_K = 296.0
LBD_REFERENCE_RUBY_NM = 694.31223       # Ruby R1 maximum
LBD_REFERENCE_BORATE_NM = 685.4505001123697  # Borate pseudo-Voigt center
LBD0_RUBY_NM = LBD_REFERENCE_RUBY_NM
LBD0_BORATE_NM = LBD_REFERENCE_BORATE_NM

# Adopted thermocouple temperature error-bar half-width (K or deg C).
# Excludes the unresolved sample–thermocouple offset.
SIGMA_T = 2.0

def get_run(run_id):
    """Return the `Run` called `run_id` (or raise KeyError)."""
    for r in RUNS:
        if r.run_id == run_id:
            return r
    raise KeyError(run_id)

# Coefficients de conversion Raman venant de Mysen (2007),
# GCA 71, 1820–1834, tableau 4. SD  à 1 sigma.
MYSEN_Q2_COEFFICIENT = (0.84, 0.07)
MYSEN_Q3_COEFFICIENT = (1.06, 0.04)
MYSEN_REFERENCE = 'Mysen (2007), Geochim. Cosmochim. Acta 71, 1820–1834, Table 4; doi:10.1016/j.gca.2007.01.007'
