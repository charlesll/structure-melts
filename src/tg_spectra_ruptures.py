"""Exploratory common-break detection for Raman PCA, Qn, and sensor series.

Uses ruptures dynamic programming with a multivariate piecewise-linear segment
cost. One breakpoint is fitted jointly across the requested feature families.
Residual permutations calibrate the best split against linear and quadratic
nulls. These p-values are conditional on exchangeability and are not adjusted
for comparisons across runs. Segment lines need not meet at the breakpoint.

Usage: python -m src.tg_spectra_ruptures
Output: results/tg_spectra_ruptures.csv/.h5,
        figures/tg_spectra_ruptures/<run>_ruptures.png
"""
import argparse
import csv

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
try:
    from ruptures.base import BaseCost
    from ruptures.detection import Dynp
except ImportError:
    BaseCost = object
    Dynp = None

from .config import FIGURES, RESULTS
from .pressure import load_dataset

from sklearn.decomposition import PCA
from .raman_treatment import prepare_glass, area_ratio

DATASET = RESULTS / "dataset.h5"
PEAK_FITS = RESULTS / "peak_fits.h5"
PCA_RANDOM_STATE = 0  # Reproductibilité des solveurs ACP aléatoires.
N_PC = 4  # Nombre de composantes des spectres verre/liquide.
N_PC_SENSOR = 2  # Nombre de composantes des formes de raies des jauges.
MIN_SIDE = 3  # Nombre minimal de paliers de chaque côté de la rupture.
MODES = ("global", "loro", "within", "run")
QN_SET = "basic"
QN_RATIO_PAIRS = tuple((i, j) for i in range(5) for j in range(i + 1, 5))
FEATURE_QN_PAIRS = ((2, 4), (0, 2), (1, 3))

def _make_decomp(k):
    """Construit l’ACP utilisée pour préparer les observables de rupture."""
    return PCA(n_components=k, random_state=PCA_RANDOM_STATE)

def align_sensor(x, spectra, positions, half_width, n_pts=121):
    """Recentre les raies de chaque grain sur une grille relative commune.

    spectra : (paliers, canaux, grains), positions : (paliers, grains).
    Retour : décalages (n_pts,) et intensités (paliers, grains, n_pts).
    half_width est exprimé dans les unités de x, ici en nm."""
    x = np.asarray(x, float)
    positions = np.asarray(positions, float)
    valid = np.isfinite(positions)
    # np.interp would silently extend the edge value outside the stored grid,
    # producing artificial flat shapes. Reject incomplete observed windows.
    if (np.any(positions[valid]-half_width < x[0]-1e-9)
            or np.any(positions[valid]+half_width > x[-1]+1e-9)):
        raise ValueError(
            "Sensor alignment window exceeds stored wavelength grid; "
            "extend Run.x_ruby/x_borate and rerun extraction")
    off = np.linspace(-half_width, half_width, n_pts)
    out = np.empty((spectra.shape[0], spectra.shape[2], n_pts))
    for i in range(spectra.shape[0]):
        for g in range(spectra.shape[2]):
            out[i, g] = np.interp(positions[i, g] + off, x, spectra[i, :, g])
    return off, out

def mean_sensor_shapes(aligned):
    """Moyenne des formes de raies sur les grains entièrement mesurés.

    Entrée : (paliers, grains, canaux). Un grain contenant une valeur
    manquante est exclu en entier. Sans grain valide, le palier reste NaN."""
    aligned = np.asarray(aligned, float)
    valid = np.isfinite(aligned).all(axis=2)
    count = valid.sum(axis=1)
    total = np.where(valid[:, :, None], aligned, 0.).sum(axis=1)
    mean = np.full((len(aligned), aligned.shape[2]), np.nan)
    np.divide(total, count[:, None], out=mean, where=count[:, None] > 0)
    return mean

def shape_dispersion(aligned):
    """Dispersion RMS des formes sur les grains mesurés.

    Entrée : (paliers, grains, canaux). Retour : (paliers,).
    Moins de deux grains valides : NaN ; variance de population."""
    aligned = np.asarray(aligned, float)
    valid = np.isfinite(aligned).all(axis=2)
    count = valid.sum(axis=1)
    mean = mean_sensor_shapes(aligned)
    squared = np.where(valid[:, :, None], (aligned-mean[:, None])**2, 0.).sum(axis=(1, 2))
    result = np.full(len(aligned), np.nan)
    np.divide(squared, count*aligned.shape[2], out=result, where=count >= 2)
    return np.sqrt(result)

def _clean_ladder(T, labels, *arrays, drop_crystal=True):
    """Sélectionne les paliers de chauffage et trie les températures croissantes.

    Exclut les retours à une température inférieure à un maximum antérieur.
    drop_crystal=True exclut aussi les étiquettes finissant par Ctx.
    Applique les mêmes indices aux étiquettes et aux tableaux supplémentaires."""
    T = np.asarray(T, float)
    keep = np.array([np.isfinite(t) and t >= np.nanmax(T[:i + 1]) for i, t in enumerate(T)])
    if drop_crystal:
        keep &= ~np.array([str(s).endswith("Ctx") for s in labels])
    order = np.argsort(T[keep], kind="stable")
    return (T[keep][order], np.asarray(labels)[keep][order]) + tuple(np.asarray(a)[keep][order] for a in arrays)

def prepare_channels(Y, standardize):
    """Centre chaque colonne de la matrice (paliers, observables).

    Si standardize=True, divise par son écart-type de population.
    Une colonne constante reste nulle sans division par zéro."""
    Y = np.asarray(Y, float)
    Y = Y - Y.mean(axis=0)
    if standardize:
        sd = Y.std(axis=0)
        Y = Y / np.where(sd > 0, sd, 1.0)
    return Y

def fit_glass_pca(spectra_by_run, mode, run_id, n_pc=N_PC):
    """Calcule les scores ACP des spectres préparés du run demandé.

    spectra_by_run : dictionnaire de matrices (paliers, canaux).
    global : apprentissage sur tous les runs ; loro : autres runs seulement ;
    within : retrait préalable de la moyenne de chaque run ; run : run seul.
    Retour : scores (paliers, composantes) et objet ACP ajusté."""
    if mode == "within":
        centred = {k: v - v.mean(axis=0) for k, v in spectra_by_run.items()}
        train = np.vstack(list(centred.values()))
        k = max(1, min(n_pc, train.shape[0] - 1))
        pca = _make_decomp(k).fit(train)
        return pca.transform(centred[run_id]), pca
    if mode == "global":
        train = np.vstack(list(spectra_by_run.values()))
    elif mode == "loro":
        train = np.vstack([s for k, s in spectra_by_run.items() if k != run_id])
    elif mode == "run":
        train = spectra_by_run[run_id]
    else:
        raise ValueError(f"unknown mode {mode!r}")
    k = max(1, min(n_pc, train.shape[0] - 1))
    pca = _make_decomp(k).fit(train)
    return pca.transform(spectra_by_run[run_id]), pca

def sensor_channels(markers_file, dataset_run, sensor_pcas):
    """Lit les raies des jauges et recentre chaque grain sur sa position.

    Retour : dictionnaire Ruby/Borate de tableaux (paliers, grains, canaux).
    dataset_run et sensor_pcas sont conservés dans la signature historique."""
    out = {}
    with h5py.File(markers_file, "r") as f:
        for name, hw in (("Ruby", 0.8), ("Borate", 0.35)):
            g = f[name]
            _, al = align_sensor(g["x"][()], g["spectra"][()], g["positions"][()], hw)
            out[name] = al
    return out

def build_sensor_matrix(aligned, pcas, run, dataset_run, T_idx):
    """Assemble les observables des jauges aux indices de paliers T_idx.

    Colonnes : scores ACP des raies, dispersions des formes, dispersions
    de pression et différence des pressions médianes rubis–borate.
    Retour : matrice (paliers sélectionnés, observables)."""
    cols = []
    for name in ("Ruby", "Borate"):
        mean = mean_sensor_shapes(aligned[name])
        valid = np.isfinite(mean).all(axis=1)
        scores = np.full((len(mean), pcas[name].n_components_), np.nan)
        scores[valid] = pcas[name].transform(mean[valid])
        cols.append(scores[T_idx])
        cols.append(shape_dispersion(aligned[name])[T_idx, None])
        cols.append(dataset_run[f"{name.lower()}_P_std"][T_idx, None])
    cols.append((dataset_run["ruby_P_median"] - dataset_run["borate_P_median"])[T_idx, None])
    return np.hstack(cols)

def qn_channels(run_id, path=PEAK_FITS, kind=None):
    """Lit les observables des bandes dans peak_fits.h5, dans l’ordre des paliers.

    Ordre des bandes : TOH, Q2, T2S, Q3, Q4. basic : fractions et positions ;
    ratios : dix log-rapports d’aires ; extended : positions, largeurs,
    trois log-rapports basés sur Q3. Retourne None si le run est absent."""
    kind = QN_SET if kind is None else kind
    if not path.exists():
        return None
    with h5py.File(path, "r") as f:
        if run_id not in f:
            return None
        g = f[run_id]
        areas, pos, wid = g["areas"][()], g["positions"][()], g["widths"][()]
    la = np.log(np.clip(areas, 1e-12, None))
    if kind == "extended":
        ratios = np.column_stack([la[:, 3] - la[:, 2], la[:, 3] - la[:, 1], la[:, 3] - la[:, 4]])
        return np.hstack([pos, wid, ratios])
    if kind == "ratios":
        return np.column_stack([la[:, i] - la[:, j] for i, j in QN_RATIO_PAIRS])
    if kind == "basic":
        return np.hstack([areas / areas.sum(axis=1, keepdims=True), pos])
    raise ValueError(f"unknown Qn channel set {kind!r}")

def ratio_test_families(x, spectra, run_id, sensors, idx, path=PEAK_FITS):
    """Prépare les familles rapports Raman, rapports de bandes, positions et jauges.

    idx conserve la correspondance avec les acquisitions sources.
    Retour : dictionnaire nom → (matrice, standardisation), ou None
    si les ajustements de bandes nécessaires sont absents."""

    qn = qn_channels(run_id, path=path, kind="basic")
    ratios = qn_channels(run_id, path=path, kind="ratios")
    if qn is None or ratios is None:
        return None
    ratio_indices = [QN_RATIO_PAIRS.index(pair) for pair in FEATURE_QN_PAIRS]
    raman_ratio = area_ratio(x, spectra)
    return {
        "raman_area_ratio": (raman_ratio[idx, None], True),
        "qn_ratios": (ratios[idx][:, ratio_indices], True),
        "peak_positions": (qn[idx, 5:10], True),
        "sensors": (sensors, True),
    }

def _sensor_pcas(aligned_by_run, n_pc=N_PC_SENSOR):
    """Ajuste deux ACP globales sur les formes moyennes des jauges.

    Les paliers sans forme entièrement finie sont exclus de l’apprentissage.
    Une population insuffisante provoque une erreur explicite."""
    pcas = {}
    for name in ("Ruby", "Borate"):
        train = np.vstack([mean_sensor_shapes(a[name]) for a in aligned_by_run.values()])
        train = train[np.isfinite(train).all(axis=1)]
        if len(train) < n_pc:
            raise ValueError(f'{name}: insufficient observed line shapes for global PCA')
        pcas[name] = _make_decomp(n_pc).fit(train)
    return pcas

OUT_CSV = RESULTS / "tg_spectra_ruptures.csv"
OUT_H5 = RESULTS / "tg_spectra_ruptures.h5"
FIG_DIR = FIGURES / "tg_spectra_ruptures"
FAMILY_NAMES = ("glass_pca", "raman_area_ratio", "qn_ratios", "peak_positions", "sensors")
# Une « famille » rassemble des observables de même origine physique.
# Une colonne = un canal ; une ligne = un palier de température.
# L'ordre des familles doit rester identique dans les matrices et les coû
FAMILY_LABELS = {
    "glass_pca": ("PC1", "PC2", "PC3", "PC4"),
    "raman_area_ratio": ("Raman area ratio",),
    "qn_ratios": ("T2S/Q4", "TOH/T2S", "Q2/Q3"),
    "peak_positions": ("TOH", "Q2", "T2S", "Q3", "Q4"),
    "sensors": ("Ruby PC1", "Ruby PC2", "Ruby shape", "Ruby P scatter",
                "Borate PC1", "Borate PC2", "Borate shape", "Borate P scatter",
                "Ruby-Borate offset"),
}


# Seuil propre à la segmentation commune, indépendant des autres méthodes Tg.
T_MIN = 150.0  # exclure les paliers de chauffage strictement sous 150 °C


class FamilyLinearCost(BaseCost):
    """Equal-family-weighted linear regression residual cost for segmentation."""

    model = "family_linear"
    min_size = 2

    def __init__(self, family_sizes):
        # Le signal contient [T, canaux famille 1, canaux famille 2, ...].
        # Les slices repèrent les colonnes de chaque famille ; la colonne 0
        # est réservée à la température, d'où start = 1.
        self.family_sizes = tuple(family_sizes)
        self.signal = None
        self._slices = []
        start = 1
        for size in self.family_sizes:
            self._slices.append(slice(start, start + size))
            start += size

    def fit(self, signal):
        if signal.ndim != 2 or signal.shape[1] != 1 + sum(self.family_sizes):
            raise ValueError("signal must contain temperature and all family channels")
        self.signal = np.asarray(signal, float)
        return self

    def family_errors(self, start, end):
        # On ajuste uniquement les lignes [start:end] (borne end exclue).
        if end - start < self.min_size:
            raise ValueError("segment is shorter than the minimum size")
        segment = self.signal[start:end]
        t = segment[:, 0]
        tc = (t - t.mean()) / max(float(t.std()), 1.0)
        # Recentrer/réduire T améliore le conditionnement numérique sans
        # changer les droites possibles. Colonnes du modèle : constante, T.
        design = np.column_stack((np.ones(len(t)), tc))
        out = {}
        for name, columns in zip(FAMILY_NAMES, self._slices):
            y = segment[:, columns]
            residual = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
            # RSS = somme des carrés des résidus, sur les paliers et canaux.
            # Diviser par le nombre de canaux évite que les 9 canaux capteurs
            # pèsent mécaniquement neuf fois plus que le rapport Raman unique.
            out[name] = float(np.sum(residual ** 2) / y.shape[1])
        return out

    def error(self, start, end):
        return float(sum(self.family_errors(start, end).values()))


def find_common_break(T, families, min_side=MIN_SIDE):
    """Find one common break using ruptures' exact dynamic-programming search."""
    T = np.asarray(T, float)
    if len(T) < 2 * min_side:
        raise ValueError(f"at least {2 * min_side} points are needed for one break")
    if tuple(families) != FAMILY_NAMES:
        raise ValueError(f"families must be ordered as {FAMILY_NAMES}")
    channels = []
    sizes = []
    for name in FAMILY_NAMES:
        Y, standardize = families[name]
        Z = prepare_channels(Y, standardize)
        # Centrage et, si demandé, division par l'écart-type de chaque canal
        # sur les seuls points fournis à l'ajustement (pas les cristallisés).
        if Z.ndim != 2 or Z.shape[0] != len(T) or not np.isfinite(Z).all():
            raise ValueError(f"invalid or non-finite channels in {name}")
        channels.append(Z)
        sizes.append(Z.shape[1])

    signal = np.column_stack((T, *channels))
    # Un seul indice de séparation pour TOUTES les familles. Chaque segment
    # possède sa propre pente et son intercept : un saut est donc autorisé.
    cost = FamilyLinearCost(sizes)
    cost.min_size = min_side
    if Dynp is None:
        # For exactly one break, exhaustive search is identical to Dynp.
        cost.fit(signal)
        split = min(range(min_side, len(T) - min_side + 1),
                    key=lambda k: cost.error(0, k) + cost.error(k, len(T)))
    else:
        # jump=1 examine toutes les séparations entre observations ; ce n'est
        # pas un pas de 1 °C. min_side impose au moins 3 points par côté.
        detector = Dynp(custom_cost=cost, min_size=min_side, jump=1).fit(signal)
        split = detector.predict(n_bkps=1)[0]
    before = cost.family_errors(0, len(T))
    # Comparer une droite sur toute la série aux deux droites séparées.
    # Ce gain par famille décrit sa contribution ; ce n'est pas une p-value.
    after = {
        name: left + right
        for name, left, right in zip(
            FAMILY_NAMES,
            [cost.family_errors(0, split)[name] for name in FAMILY_NAMES],
            [cost.family_errors(split, len(T))[name] for name in FAMILY_NAMES],
        )
    }
    improvement = {name: before[name] - after[name] for name in FAMILY_NAMES}
    return {
        "split_index": split,
        # Le milieu entre deux températures est une convention de restitution.
        # [T_left, T_right] n'est PAS un intervalle de confiance statistique.
        "T_break": float((T[split - 1] + T[split]) / 2),
        "T_left": float(T[split - 1]),
        "T_right": float(T[split]),
        "improvement": improvement,
        "total_improvement": float(sum(improvement.values())),
    }


def permutation_test(T, families, n_permutations=999, seed=0, min_side=MIN_SIDE,
                     null_degree=1):
    """Calibrate the best segmented RSS improvement against a polynomial null.

    Residual rows are permuted independently by family, preserving within-family
    channel covariance. The best break is re-searched for every permutation.
    """
    if n_permutations < 1:
        raise ValueError("n_permutations must be positive")
    if null_degree not in (1, 2):
        raise ValueError("null_degree must be 1 or 2")
    T = np.asarray(T, float)
    if len(T) < 2 * min_side:
        raise ValueError(f"at least {2 * min_side} points are needed for one break")
    if tuple(families) != FAMILY_NAMES:
        raise ValueError(f"families must be ordered as {FAMILY_NAMES}")

    Ys = {}
    sizes = []
    for name in FAMILY_NAMES:
        Y, standardize = families[name]
        Z = prepare_channels(Y, standardize)
        if Z.ndim != 2 or Z.shape[0] != len(T) or not np.isfinite(Z).all():
            raise ValueError(f"invalid or non-finite channels in {name}")
        Ys[name] = Z
        sizes.append(Z.shape[1])

    cost = FamilyLinearCost(sizes)
    cost.min_size = min_side

    def total_segmented_rss(samples):
        # Rechercher le meilleur split à chaque appel, y compris pour les
        # données simulées : on tient ainsi compte de la recherche de rupture.
        cost.fit(np.column_stack((T, *(samples[name] for name in FAMILY_NAMES))))
        candidate_costs = [
            cost.error(0, split) + cost.error(split, len(T))
            for split in range(min_side, len(T) - min_side + 1)
        ]
        return float(min(candidate_costs))

    def null_fit(Y, degree):
        # Hypothèse sans rupture : droite (degree=1) ou parabole (degree=2).
        # Toutes les colonnes sont ajustées simultanément par moindres carrés.
        tc = (T - T.mean()) / max(float(T.std()), 1.0)
        design = np.column_stack([tc ** power for power in range(degree + 1)])
        coefficients = np.linalg.lstsq(design, Y, rcond=None)[0]
        fitted = design @ coefficients
        return fitted, Y - fitted

    observed_null_rss = 0.0
    fitted_residuals = {}
    for name, Y in Ys.items():
        fitted, residual = null_fit(Y, null_degree)
        observed_null_rss += float(np.sum(residual ** 2) / Y.shape[1])
        fitted_residuals[name] = (fitted, residual)
    observed_gain = observed_null_rss - total_segmented_rss(Ys)
    # Gain observé = RSS du modèle nul - meilleur RSS à deux segmen
    # Sous le nul quadratique, ce gain peut être négatif : les deux modèles
    # ne sont pas emboîtés et la parabole peut mieux expliquer les données.

    rng = np.random.default_rng(seed)
    null_gains = np.empty(n_permutations)
    for iteration in range(n_permutations):
        synthetic = {}
        null_rss = 0.0
        for name in FAMILY_NAMES:
            fitted, residual = fitted_residuals[name]
            permuted = residual[rng.permutation(len(T))]
            # Une seule permutation des LIGNES pour tous les canaux d'une
            # famille préserve leur covariance. Chaque famille est mélangée
            # séparément : la dépendance entre familles n'est pas préservée.
            # Cette procédure suppose des résidus échangeables entre paliers.
            Y = fitted + permuted
            synthetic[name] = Y
            _, remaining = null_fit(Y, null_degree)
            # Réajuster aussi le modèle nul sur chaque série simulée, afin de
            # comparer exactement la même statistique que sur les observations.
            null_rss += float(np.sum(remaining ** 2) / Y.shape[1])
        null_gains[iteration] = null_rss - total_segmented_rss(synthetic)

    p_value = float((1 + np.sum(null_gains >= observed_gain)) / (n_permutations + 1))
    # Correction « +1 » : aucune p-value nulle ; avec 999 permutations,
    # la plus petite valeur possible est 1/1000 = 0.001.
    return {
        "p_value": p_value,
        "observed_gain": float(observed_gain),
        "null_gains": null_gains,
        "null_degree": null_degree,
        "n_permutations": n_permutations,
    }


def _run_families(D, x, markers_dir, n_min=T_MIN, drop_crystal=True):
    """Build families; retain crystals only for an explicitly requested display."""
    # Les raies sont réalignées autour de leur position pour décrire leur
    # forme indépendamment de leur déplacement spectral avec P et T.
    aligned = {
        run_id: sensor_channels(markers_dir / f"{run_id}.h5", d, None)
        for run_id, d in D.items()
    }
    pcas = _sensor_pcas(aligned)
    # Prétraitement Raman : fond linéaire en deux parties + normalisation.
    # Les PCA globales apprennent sur TOUS les runs et tous leurs états,
    # avant la sélection des paliers utilisés pour rechercher la rupture.
    glass_spectra = {
        run_id: prepare_glass(x, d["spectra"])
        for run_id, d in D.items()
    }
    glass_pca_scores = {}
    for run_id in D:
        scores, _ = fit_glass_pca(glass_spectra, "global", run_id)
        glass_pca_scores[run_id] = scores
    output = {}
    for run_id, d in D.items():
        if not d["is_tg_series"]:
            # Le run de compression à 25 °C n'est pas une série de chauffage.
            continue
        T, labels, idx = _clean_ladder(
            d["T_C"], d["Etat"], np.arange(len(d["T_C"])),
            drop_crystal=drop_crystal)
        hot = T >= n_min
        # idx conserve la correspondance avec les lignes d'origine, pour
        # sélectionner les mêmes paliers dans les capteurs et les bandes Qn.
        T, labels, idx = T[hot], labels[hot], idx[hot].astype(int)
        if len(T) < 2 * MIN_SIDE:
            continue
        sensors = build_sensor_matrix(aligned[run_id], pcas, run_id, d, idx)
        # Attention : les canaux historiques appelés « P scatter » utilisent
        # P_std, qui comprend aujourd'hui aussi des incertitudes propagées.
        ratio_families = ratio_test_families(
            x, glass_spectra[run_id], run_id, sensors, idx)
        if ratio_families is None:
            continue
        families = {
            "glass_pca": (glass_pca_scores[run_id][idx], True),
            **ratio_families,
        }
        families = {name: families[name] for name in FAMILY_NAMES}
        output[run_id] = (T, labels, families)
    return output


def plot_run(run_id, T, labels, families, result, path):
    """Plot standardized feature channels, family means, and the detected break."""
    # Diagnostic détaillé historique. Il ne peut afficher que les points
    # reçus ; main() lui transmet par défaut la série sans cristallisés.
    # La figure du manuscrit est tracée séparément dans fig6_ruptures.py.
    fig, axes = plt.subplots(3, 2, figsize=(11, 10), constrained_layout=True)
    for ax, name in zip(axes.flat, FAMILY_NAMES):
        values, standardize = families[name]
        Z = prepare_channels(values, standardize)
        for j, label in enumerate(FAMILY_LABELS[name]):
            ax.plot(T, Z[:, j], "o-", ms=3, lw=0.8, alpha=0.45, label=label)
        ax.plot(T, Z.mean(axis=1), "k-", lw=2, label="family mean")
        # Cette moyenne est descriptive : des canaux de signes opposés peuvent
        # s'annuler. Le coût de détection utilise chaque canal séparément.
        ax.axvline(result["T_break"], color="C3", ls="--",
                   label=f"break {result['T_break']:.1f} °C")
        glass = np.array([str(s).startswith("Verre") for s in labels])
        if glass.any() and not glass.all():
            last = np.flatnonzero(glass)[-1]
            ax.axvspan(T[last], T[last + 1], color="orange", alpha=0.2,
                       label="Verre→Melt bracket")
        ax.set(title=name.replace("_", " "), xlabel="Temperature, °C",
               ylabel="Standardized channel")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7, loc="best")
    axes.flat[-1].set_visible(False)
    fig.suptitle(f"{run_id}: exploratory ruptures single-break fit")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main(argv=None):
    # Pipeline complet : charger -> construire les familles -> ajuster chaque
    # run -> tester les deux modèles nuls -> enregistrer chiffres et tracés.
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--t-min", type=float, default=T_MIN,
                        help="exclude points colder than this temperature (°C; default: 150)")
    parser.add_argument("--n-permutations", type=int, default=999,
                        help="permutations per run and null model (default: 999)")
    args = parser.parse_args(argv)
    D = load_dataset()
    with h5py.File(DATASET, "r") as f:
        x = f["x_axis"][()]
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    runs = _run_families(D, x, RESULTS / "markers", args.t_min)
    rows = []
    with h5py.File(OUT_H5, "w") as output:
        output.attrs["method"] = "ruptures Dynp, one common breakpoint, family-linear cost"
        output.attrs["permutation_method"] = (
            "family-wise residual row permutation; best break re-searched per draw")
        output.attrs["n_permutations"] = args.n_permutations
        output.attrs["t_min_C"] = args.t_min
        for run_id, (T, labels, families) in runs.items():
            result = find_common_break(T, families)
            p_linear = permutation_test(
                T, families, args.n_permutations, seed=0, null_degree=1)
            p_quadratic = permutation_test(
                T, families, args.n_permutations, seed=1, null_degree=2)
            p_conservative = max(p_linear["p_value"], p_quadratic["p_value"])
            # Prendre le maximum impose que les deux tests passent le seuil.
            # Cela ne corrige PAS les comparaisons multiples entre runs.
            rows.append({
                "run_id": run_id,
                "n_points": len(T),
                "split_index": result["split_index"],
                "T_break_C": result["T_break"],
                "T_left_C": result["T_left"],
                "T_right_C": result["T_right"],
                "total_RSS_improvement": result["total_improvement"],
                "p_linear": p_linear["p_value"],
                "p_quadratic": p_quadratic["p_value"],
                "p_conservative": p_conservative,
                **{f"RSS_improvement_{k}": v for k, v in result["improvement"].items()},
            })
            group = output.create_group(run_id)
            # HDF5 conserve matrices, gains simulés et paramètres pour audit ;
            # CSV fournit un résumé compact avec une ligne par run.
            group.create_dataset("T_C", data=T)
            group.create_dataset("labels", data=np.asarray(labels, dtype="S"))
            group.attrs.update({k: v for k, v in result.items() if k != "improvement"})
            group.attrs["p_linear"] = p_linear["p_value"]
            group.attrs["p_quadratic"] = p_quadratic["p_value"]
            group.attrs["p_conservative"] = p_conservative
            for name, value in result["improvement"].items():
                group.attrs[f"RSS_improvement_{name}"] = value
            group.create_dataset("null_gains_linear", data=p_linear["null_gains"])
            group.create_dataset("null_gains_quadratic", data=p_quadratic["null_gains"])
            for name, (Y, standardize) in families.items():
                group.create_dataset(name, data=Y)
                group.attrs[f"{name}_standardized"] = standardize
            plot_run(run_id, T, labels, families, result,
                     FIG_DIR / f"{run_id}_ruptures.png")
    with open(OUT_CSV, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: f"{value:.4f}" if isinstance(value, float) else value
                for key, value in row.items()
            })
    print("[tg_spectra_ruptures] ->", OUT_CSV)


if __name__ == "__main__":
    main()
