"""Supplementary Figure 9: common-break analysis of run05_20250408.

Same analysis, selection and layout as Figure 6; crystals are display-only.
"""
from .fig6_ruptures import main as plot_ruptures

NAME = "Supplementary_Figure_9"
RUN_ID = "run05_20250408"


def main(d=None, outdir="manuscript", n_permutations=999):
    """Generate the PNG, PDF and numerical JSON for this heating run."""
    return plot_ruptures(d, outdir, n_permutations,
                         run_id=RUN_ID, output_name=NAME)


if __name__ == "__main__":
    main()
