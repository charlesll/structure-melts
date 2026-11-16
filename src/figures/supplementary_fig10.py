"""Supplementary Figure 10: common-break analysis of run06_20250430.

Same analysis, selection and layout as Figure 6; crystals are display-only.
"""
from .fig6_ruptures import main as plot_ruptures

NAME = "Supplementary_Figure_10"
RUN_ID = "run06_20250430"


def main(d=None, outdir="manuscript", n_permutations=999):
    """Generate the PNG, PDF and numerical JSON for this heating run."""
    return plot_ruptures(d, outdir, n_permutations,
                         run_id=RUN_ID, output_name=NAME)


if __name__ == "__main__":
    main()
