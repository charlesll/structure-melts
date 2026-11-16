"""Propagation gvar des aires corrélées (ordre TOH,Q2,T2S,Q3,Q4).

Les matrices HDF5 conservent les covariances, les GVar sont reconstruits par
acquisition. Aucune corrélation entre acquisitions n'est inventée.
"""
import numpy as np


def area_gvars(areas, covariance):
    """Construit cinq variables corrélées; covariance en unités d'aire²."""
    import gvar as gv
    a, c = np.asarray(areas, float), np.asarray(covariance, float)
    if a.shape != (5,) or c.shape != (5,5):
        raise ValueError('Expected five areas and a 5x5 covariance')
    if not np.isfinite(a).all() or not np.isfinite(c).all():
        raise ValueError('Nonfinite areas/covariance')
    if not np.allclose(c, c.T, rtol=1e-8, atol=1e-12):
        raise ValueError('Asymmetric area covariance')
    if np.linalg.eigvalsh(c).min() < -1e-10*max(np.max(np.abs(c)), 1e-20):
        raise ValueError('Area covariance is not positive semidefinite')
    return gv.gvar(a, (c+c.T)/2)


def propagate_area_ratio(areas, covariance, numerator=1, denominator=3):
    """Rapport par acquisition et SD; entrées (...,5), (...,5,5)."""
    import gvar as gv
    a, c = np.asarray(areas,float), np.asarray(covariance,float)
    if a.shape[-1:] != (5,) or c.shape != a.shape[:-1]+(5,5):
        raise ValueError('Incompatible area/covariance shapes')
    mean=np.full(a.shape[:-1],np.nan); sd=mean.copy()
    for idx in np.ndindex(mean.shape):
        if not np.isfinite(a[idx]).all() or not np.isfinite(c[idx]).all():
            continue  # Fit manquant : sortie NaN, sans fabriquer de covariance.
        if a[idx][denominator] == 0: continue
        v=area_gvars(a[idx],c[idx]); ratio=v[numerator]/v[denominator]
        mean[idx]=gv.mean(ratio); sd[idx]=gv.sdev(ratio)
    return mean,sd


def propagate_area_fractions(areas, covariance):
    """Fractions et covariance complète via GVar; mêmes dimensions que l'entrée."""
    import gvar as gv
    a,c=np.asarray(areas,float),np.asarray(covariance,float)
    if a.shape[-1:] != (5,) or c.shape != a.shape[:-1]+(5,5):
        raise ValueError('Incompatible area/covariance shapes')
    mean=np.full_like(a,np.nan); cov=np.full_like(c,np.nan)
    for idx in np.ndindex(a.shape[:-1]):
        if not np.isfinite(a[idx]).all() or not np.isfinite(c[idx]).all(): continue
        if a[idx].sum() == 0: continue
        v=area_gvars(a[idx],c[idx]); f=v/sum(v)
        mean[idx]=gv.mean(f); cov[idx]=gv.evalcov(f)
    return mean,cov


def nbot_from_fractions(fractions, covariance):
    """NBO/T=2*c2*f_Q2+c3*f_Q3, sans renormalisation supplémentaire.

    Entrées (n,5) et (n,5,5). Fractions des CINQ bandes, même dénominateur
    que Figure8a. Fits indépendants entre acquisitions; deux coefficients
    communs, supposés indépendants entre eux et des fractions.
    Renvoie moyenne (n,), covariance totale (n,n), covariance fit seule (n,n).
    Propagation gvar au premier ordre, coefficients interprétés à1sigma.
    """
    import gvar as gv
    from .config import MYSEN_Q2_COEFFICIENT, MYSEN_Q3_COEFFICIENT
    f,c=np.asarray(fractions,float),np.asarray(covariance,float)
    if f.ndim != 2 or f.shape[1] != 5 or c.shape != (len(f),5,5):
        raise ValueError('Expected fractions (n,5) and covariance (n,5,5)')
    if not np.allclose(f.sum(axis=1),1.,atol=1e-8):
        raise ValueError('Fractions must sum to one across the five bands')
    c2=gv.gvar(*MYSEN_Q2_COEFFICIENT); c3=gv.gvar(*MYSEN_Q3_COEFFICIENT)
    total=[]; fit_only=[]
    for mean,cov in zip(f,c):
        v=area_gvars(mean,cov)
        total.append(2*c2*v[1]+c3*v[3])
        fit_only.append(2*gv.mean(c2)*v[1]+gv.mean(c3)*v[3])
    return np.asarray(gv.mean(total)), np.asarray(gv.evalcov(total)), np.asarray(gv.evalcov(fit_only))
