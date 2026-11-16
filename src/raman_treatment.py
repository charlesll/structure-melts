"""Traitements Raman du verre et du liquide.

Aucune détection de rupture ni écriture de résultats dans ce module.
Les deux définitions de centroïde sont conservées explicitement.
Les tests GLS du NBO/T utilisent la covariance complète des observables.
"""
import numpy as np
import rampy as rp
from scipy.optimize import curve_fit
from scipy.stats import chi2

def prepare_glass(x, spectra):
    """Corrige la ligne de base puis normalise chaque spectre à une aire unité.

    x : axe Raman (cm⁻¹), de forme (n_canaux,).
    spectra : intensités de forme (n_spectres, n_canaux).
    Le minimum entre 800 et 950 cm⁻¹ sépare deux lignes de base linéaires.
    La normalisation utilise une intégration par trapèzes, sans écrêtage."""
    x = np.asarray(x, float)
    out = np.empty_like(np.asarray(spectra, float))
    win = (x > 800) & (x < 950)
    for i, y in enumerate(np.asarray(spectra, float)):
        xm = x[win][np.argmin(y[win])]
        lo = (x < 410) | (np.abs(x - xm) < 5)
        hi = (x > 1220) | (np.abs(x - xm) < 5)
        base = np.where(x < xm, np.polyval(np.polyfit(x[lo], y[lo], 1), x),
                        np.polyval(np.polyfit(x[hi], y[hi], 1), x))
        s = y - base
        out[i] = s / np.trapezoid(s, x)
    return out

def centroid(x, spectra, window):
    """Centroïde intégré sur une fenêtre inclusive en cm⁻¹.

    Entrée : axe x (n_canaux,), spectres (n_spectres, n_canaux).
    Retour : un centroïde par spectre, calculé par trapèzes. Les intensités
    restent signées ; aucune correction de ligne de base implicite."""
    m = (x >= window[0]) & (x <= window[1])
    y = spectra[:, m]
    return np.trapezoid(y * x[m], x[m], axis=1) / np.trapezoid(y, x[m], axis=1)

def area_ratio(x, spectra, low_window=(400.0, 830.0), high_window=(830.0, 1270.0)):
    """Rapport des aires LF/HF intégrées par trapèzes.

    Fenêtres inclusives en cm⁻¹ ; spectres de forme (n_spectres, len(x)).
    Une aire au dénominateur nulle ou non finie donne NaN."""
    x = np.asarray(x, float)
    spectra = np.asarray(spectra, float)
    low = (x >= low_window[0]) & (x <= low_window[1])
    high = (x >= high_window[0]) & (x <= high_window[1])
    low_area = np.trapezoid(spectra[:, low], x[low], axis=1)
    high_area = np.trapezoid(spectra[:, high], x[high], axis=1)
    return np.divide(low_area, high_area, out=np.full_like(low_area, np.nan),
                     where=np.isfinite(high_area) & (high_area != 0))

def lf_centroid(x, spectra):
    """Moment discret LF historique de la figure 10, en cm⁻¹.

    Utilise les sommes signées sur x < 830 cm⁻¹, sur la grille régulière.
    Cette définition diffère du centroïde intégré par trapèzes : la conserver
    pour reproduire exactement la figure. Dénominateur nul : NaN."""
    x = np.asarray(x, float)
    spectra = np.asarray(spectra, float)
    if spectra.ndim != 2 or spectra.shape[1] != len(x):
        raise ValueError('Expected spectra shape (n_spectra, len(x))')
    mask = x < 830
    numerator = (spectra[:, mask] * x[mask]).sum(axis=1)
    denominator = spectra[:, mask].sum(axis=1)
    return np.divide(numerator, denominator, out=np.full(len(spectra), np.nan),
                     where=denominator != 0)

def treat_glass(sp, sp_hf, bkg,
                convert_in_cm = True,
                bkg_cutoff = 1220.0,
                roi_hf = np.array([[3770, 4000]]),
                long_correction = False,
                T_long = 23.0,
                laser_wavelength = 532.0):
    """Traite indépendamment les acquisitions LF et HF du verre ou du liquide.

    sp, sp_hf et bkg : tableaux (n_canaux, 2), axe puis intensité.
    Axe en nm si convert_in_cm=True, sinon en cm⁻¹. Les tableaux
    sont modifiés par le despiking. Le fond mesuré est ajusté puis soustrait.
    HF absent : retourne deux tableaux vides pour cette région.
    Retour : x_LF, y_LF, x_HF, y_HF ; normalisation historique conservée."""
    sp[:,1] = rp.despiking(sp[:,0], sp[:,1])
    if sp_hf is not None:
        sp_hf[:,1] = rp.despiking(sp_hf[:,0], sp_hf[:,1])
    bkg[:,1] = rp.despiking(bkg[:,0], bkg[:,1])
    
    # Conversion de l’axe en déplacement Raman (cm⁻¹).
    if convert_in_cm == True:
        x_ = nm_to_invcm(sp[:,0])
        x_hf = nm_to_invcm(sp_hf[:,0]) if sp_hf is not None else np.array([])
    else:
        x_ = np.copy(sp[:,0])
        x_hf = np.copy(sp_hf[:,0]) if sp_hf is not None else np.array([])

    # Région utilisée pour ajuster le fond mesuré.
    sp_b = sp[x_>bkg_cutoff, 1]
    bkg_b = bkg[x_>bkg_cutoff, 1]

    # Ajustement du décalage et du facteur multiplicatif du fond.
    popt, _ = curve_fit(bkg_fit, bkg_b, sp_b)

    # Soustraction du fond ajusté.
    bkg_new = popt[0] + popt[1]*bkg[:,1]
    y_ = sp[:,1] - bkg_new

    # Correction thermique de Long, uniquement si demandée.
    if long_correction == True:
        _, y_, _ = rp.tlcorrection(x_, y_.copy(), T_long, laser_wavelength, correction="long")

    # Correction constante résiduelle de la ligne de base.
    y_, bas_ = rp.baseline(x_, y_.copy(), roi = np.array([[1220, 1400.0],]), method="poly", polynomial_order = 0)

    # Normalisation historique de la somme à 500.
    y_ = y_ / np.sum(y_) * 500

    # Retrait des dimensions unitaires.
    y_ = np.squeeze(y_)

    # Une acquisition HF absente ne doit pas supprimer la mesure LF valide.
    # L’extraction représentera la région HF manquante par des NaN.
    if sp_hf is None:
        return x_, y_, np.array([]), np.array([])

    # Traitement indépendant de la région haute fréquence.
    # Aucune correction de Long dans cette région.
    y_corr, _ = rp.baseline(x_hf, sp_hf[:,1].ravel(), roi_hf, "poly", polynomial_order=1)
    y_hf = (y_corr-np.min(y_corr))/(np.max(y_corr)-np.min(y_corr))

    return x_, y_, x_hf, y_hf

def nm_to_invcm(x, laser_nm=532.0):
    """Convertit une longueur d’onde en nm en déplacement Raman en cm⁻¹."""
    x_inv_cm = 1.0/(x*1.0e-7)
    laser_inv_cm = 1.0/(laser_nm*1.0e-7)
    shift_inv_cm = laser_inv_cm-x_inv_cm
    return shift_inv_cm

def invcm_to_nm(shift_inv_cm, laser_nm=532.0):
    """Convertit un déplacement Raman en cm⁻¹ en longueur d’onde en nm."""
    laser_inv_cm = 1.0 / (laser_nm * 1.0e-7)
    x_inv_cm = laser_inv_cm - shift_inv_cm
    x_nm = 1.0 / (x_inv_cm * 1.0e-7)
    return x_nm

def bkg_fit(bkg, a, b):
    """Fond affine a + b × bkg utilisé pour ajuster le fond mesuré."""
    return a + bkg*b


def compare_pressure_gls(p, y, covariance):
    """Compare constante, droite et deux plateaux à 7 GPa par moindres carrés généralisés.

    p : pressions en GPa, vecteur (n,).
    y : valeurs de NBO/T, vecteur (n,).
    covariance : covariance complète de y, matrice (n, n), symétrique et
    définie positive. Elle inclut les erreurs communes de conversion.

    La covariance reste fixée à sa valeur propagée ; les pressions sont
    traitées comme exactes. Le test suppose y approximativement gaussien.
    Le blanchiment de Cholesky transforme le problème en régression ordinaire.
    Les erreurs des paramètres ne sont pas rééchelonnées par le chi² résiduel.

    Retour : chi² de la constante et dictionnaires linear / fixed_7GPa :
    effect (pente par GPa ou contraste haut−bas), se, CI95 (±1,96 se),
    chi2, df, delta_chi2 et p (gain de chi² à un degré de liberté).
    Le seuil de 7 GPa est fixé ; aucune correction de sa sélection historique,
    ni séparation des effets température/run, n’est effectuée.
    """
    from scipy.linalg import solve_triangular

    p, y, c = np.asarray(p, float), np.asarray(y, float), np.asarray(covariance, float)
    if p.shape != y.shape or y.ndim != 1 or c.shape != (len(y), len(y)):
        raise ValueError("Dimensions incompatibles pour le GLS")
    if not all(np.isfinite(v).all() for v in [p, y, c]) or not np.allclose(c, c.T):
        raise ValueError("Données GLS non finies ou covariance non symétrique")

    # Blanchir les valeurs et les matrices de régression avec le même facteur.
    L = np.linalg.cholesky(c)
    wy = solve_triangular(L, y, lower=True)

    def fit(design):
        """Ajuste une matrice de régression avec la covariance fixée."""
        wx = solve_triangular(L, design, lower=True)
        if np.linalg.matrix_rank(wx) != design.shape[1]:
            raise ValueError("Modèle de pression non identifiable (rang insuffisant)")
        beta = np.linalg.lstsq(wx, wy, rcond=None)[0]
        bc = np.linalg.inv(wx.T @ wx)
        resid = wy - wx @ beta
        return beta, bc, float(resid @ resid)

    _, _, null = fit(np.ones((len(y), 1)))
    out = {"constant_chi2": null, "df_constant": len(y)-1}
    # Chaque alternative ajoute un seul paramètre à la constante.
    for name, x in [("linear", p-p.mean()), ("fixed_7GPa", (p >= 7).astype(float))]:
        beta, bc, rss = fit(np.column_stack([np.ones(len(y)), x]))
        se = float(np.sqrt(bc[1, 1]))
        gain = max(0., null-rss)
        out[name] = {
            "effect": float(beta[1]), "se": se,
            "CI95": [float(beta[1]-1.96*se), float(beta[1]+1.96*se)],
            "chi2": rss, "df": len(y)-2, "delta_chi2": gain,
            "p": float(chi2.sf(gain, 1)),
        }
    return out
