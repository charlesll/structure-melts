import numpy as np
import matplotlib.pyplot as plt
import rampy as rp
from scipy.signal import find_peaks
import os


def estimate_peak_parameters(
    x, y, 
    cut_lb=684.0, 
    cut_hb=696.0,
    bkg_width=0.5,
    smooth = True,
    find_peaks_height = 0.5,
    find_peaks_width = 10.0,
    find_peaks_rel_height = 0.7,
    Lambda_whittaker=10**2.5,
    polynomial_order=1,
    figure=True,
    save_figure=False,
    save_name="test.pdf",
    close_figure=False
):
    """
    Estimates the parameters of a peak in a given signal.

    Parameters:
    ----------
    x : array-like
        Array of x-values (e.g., wavelengths or time).
    y : array-like
        Array of corresponding y-values (signal intensity).
    cut_lb : float, optional, default=684.0
        Lower bound of the x-region to consider for the peak.
    cut_hb : float, optional, default=696.0
        Upper bound of the x-region to consider for the peak.
    bkg_width : float, optional, default=0.5
        Width of the region near the edges for background estimation.
    smooth : bool, True or False
        Activate Whittaker smoothing for the spectrum
    find_peaks_height : float, optional; default = 0.5
    find_peaks_width : float, optional; default = 10.0
    find_peaks_rel_height : float, optional; default = 0.7
        Detection width threshold only; returned FWHM uses half peak height.
    Lambda_whittaker : float, optional, default=10**2.5
        Regularization parameter for Whittaker smoothing.
    polynomial_order : int, optional, default=1
        Order of the polynomial used for background subtraction.
    figure : bool, optional, default=True
        Whether to generate and display plots for visualization.
    save_figure : bool, optional, default=False
        Whether to save the generated figure to a file.
    save_name : str, optional, default="test.pdf"
        Filename to save the figure if `save_figure` is True.
    close_figure : bool, optional, default=False
        Whether to close the figure after saving or displaying.

    Returns:
    -------
    tuple
        Returns a tuple containing:
        - centroid : float
            Centroid of the peak.
        - position : float
            Position of the peak maximum.
        - fwhm : float
            Full width at half maximum (FWHM) of the peak, in units of x.
        - x : array-like
            Selected wavelength axis.
        - y_bas : array-like
            Normalized signal after smoothing and background subtraction.

    Notes:
    -----
    - Uses Whittaker smoothing for signal refinement.
    - Fits a polynomial background based on regions near the peak edges.
    - Automatically normalizes the signal for peak detection.
    - If enabled, generates and optionally saves a two-panel plot:
        - Left: Normalized signal and smoothed signal.
        - Right: Signal, background, final processed signal, and peak.

    Raises:
    ------
    ValueError
        If input lengths differ, no peak is detected, or half-height crossings are absent.

    Example:
    -------
    >>> x = np.linspace(680, 700, 1000)
    >>> y = np.exp(-0.5 * ((x - 690) / 1)**2)
    >>> estimate_peak_parameters(x, y)
    (689.99, 690.0, 2.35, array([...]), array([...]))
    """
    # Check if inputs have the same length
    if len(x) != len(y):
        raise ValueError("Input arrays x and y must have the same length.")
    
    # First selection: Cut the signal around the peak region
    mask = (x > cut_lb) & (x < cut_hb)
    x, y = x[mask], y[mask]


    y /= np.max(y)  # Normalize the signal

    # Smooth the signal using Whittaker smoothing
    if smooth == True:
        y_smo = rp.smooth(x, y, method="whittaker", Lambda=Lambda_whittaker)
    else:
        y_smo = np.copy(y)

    # Subtract background using a polynomial fit
    roi = np.array([[np.min(x), np.min(x) + bkg_width],
                    [np.max(x) - bkg_width, np.max(x)]])
    y_bas, bkg = rp.baseline(x, y_smo, roi, method="poly", polynomial_order=polynomial_order)

    # normalize the signal
    y_bas = rp.normalise(y_bas.ravel(), method="intensity")

    # Calculate peak parameters
    centroid = rp.centroid(x.reshape(-1, 1), y_bas.reshape(-1, 1))

    # find peak position
    peaks, _ = find_peaks(y_bas, height=find_peaks_height,
                          width=find_peaks_width, rel_height=find_peaks_rel_height)
    if not len(peaks):
        raise ValueError("No peak satisfies the detection criteria")
    position = x[peaks[0]]
    # Le seuil de détection ne définit pas la largeur à mi-hauteur.
    fwhm = _fwhm_at_peak(x, y_bas, peaks[0])

    # Generate visualization if requested
    if figure:
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.plot(x, y_smo, "r-", label="Smoothed signal")
        plt.plot(x, y, label="Normalized signal")
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Intensity (a.u.)")
        plt.legend(loc="best")

        plt.subplot(1, 2, 2)
        plt.plot(x, bkg, "r-", label="Background")
        plt.plot(x, y_smo, label="Smoothed signal")
        plt.plot(x, y_bas, "b-", label="Final signal")
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Intensity (a.u.)")
        plt.axvline(position, color="k", linestyle="--", linewidth=0.5, label="Peak position")
        plt.legend(loc="best")

        plt.tight_layout()

        if save_figure:
            plt.savefig(save_name, dpi=180)
        if close_figure:
            plt.close()

    return centroid[0], position, fwhm, x, y_bas

def estimate_position(x, y):
    """
    Estimate the position of the maximum of a peak.

    Parameters:
    - x (array-like): The x-values (e.g., Raman shift or wavelength).
    - y (array-like): The y-values (e.g., intensity or counts).

    Returns:
    - position (float): Estimated position of the maximum value of a peak
    """
    return x[np.argmax(y)]

def _fwhm_at_peak(x, y, peak_index):
    """Largeur à mi-hauteur du pic choisi, avec interpolation sur l’axe x.

    Le signal doit être corrigé de son fond. Les premiers croisements de
    y[pic]/2 de part et d’autre du pic délimitent la largeur, en unités de x.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    if (x.ndim != 1 or y.shape != x.shape or not np.all(np.isfinite(x))
            or not np.all(np.isfinite(y)) or not np.all(np.diff(x) > 0)):
        raise ValueError("Expected finite signal on a strictly increasing axis")
    half = y[peak_index] / 2
    left = np.flatnonzero(y[:peak_index] <= half)
    right = np.flatnonzero(y[peak_index + 1:] <= half)
    if half <= 0 or not len(left) or not len(right):
        raise ValueError("Peak lacks positive height or two half-height crossings")
    i, j = left[-1], peak_index + 1 + right[0]
    lower = x[i] + (half-y[i])*(x[i+1]-x[i])/(y[i+1]-y[i])
    upper = x[j-1] + (half-y[j-1])*(x[j]-x[j-1])/(y[j]-y[j-1])
    return float(upper-lower)


def estimate_fwhm(x, y):
    """Return interpolated FWHM of the signal maximum, in x units.

    Raises ValueError when two half-height crossings cannot be measured.
    """
    return _fwhm_at_peak(x, y, int(np.argmax(y)))

def create_dir(dirName):
    """search and, if necessary, create a folder

    Parameters
    ----------
    dirName : str
        path of the new directory
    """
    try:
        # Create target Directory
        os.mkdir(dirName)
        print("Directory ", dirName, " Created ")
    except FileExistsError:
        print("Directory ", dirName, " already exists")

