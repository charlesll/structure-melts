"""Supplementary Figure 8: common-break analysis of run03_20250303_32bar.

Same analysis, selection and layout as Figure 6; crystals are display-only.
"""
from .fig6_ruptures import main as plot_ruptures

NAME = "Supplementary_Figure_8"
RUN_ID = "run03_20250303_32bar"


def main(d=None, outdir="manuscript", n_permutations=999):
    """Generate the PNG, PDF and numerical JSON for this heating run."""
    return plot_ruptures(d, outdir, n_permutations,
                         run_id=RUN_ID, output_name=NAME)


if __name__ == "__main__":
    main()
