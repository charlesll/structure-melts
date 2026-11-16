"""Figure8 : fractions d'aires vs P et NBO/T vs T/P, sélection exacte Melt."""
import json
import numpy as np
import matplotlib.pyplot as plt
from ..pressure import load_dataset
from ..figure_data import save, style, PEAK_ORDER, PEAK_LABELS, RUN_LABEL
from ..manuscript_data import WIDTH, BAND_COLORS, BAND_MARKERS, RUN_COLORS, fits_for
from ..peak_uncertainty import nbot_from_fractions
from ..config import MYSEN_REFERENCE, MYSEN_Q2_COEFFICIENT, MYSEN_Q3_COEFFICIENT, SIGMA_T, ANN_HYDROUS_GLASS_NBO_T
NAME = 'fig8_melt_areas'


def main(d=None):
    style(); D=load_dataset()
    # Panneau gauche sur toute la hauteur; deux diagnostics alignés à droite.
    fig=plt.figure(figsize=(WIDTH,4.4),layout='constrained')
    grid=fig.add_gridspec(2,2)
    a=fig.add_subplot(grid[:,0])
    b=fig.add_subplot(grid[0,1])
    c=fig.add_subplot(grid[1,1],sharey=b)
    fractions=[]; covariances=[]; rows=[]
    annotation_pos_ = [(4,0.06),(4,0.16),(4,0.20),(4,0.32),(8.5,0.20)]
    for k,(run,r) in enumerate(D.items()):
        m=r['Etat']=='Melt'
        f,cov,_,_=fits_for(run,r)
        p=r['borate_P_median'][m]; ps=r['borate_P_std'][m]
        sd=np.sqrt(np.maximum(np.diagonal(cov,axis1=-2,axis2=-1),0))
        for j,(color,marker) in enumerate(zip(BAND_COLORS,BAND_MARKERS)):
            a.errorbar(p,f[m,j],xerr=ps,yerr=sd[m,j],fmt=marker,
                       color=color,ms=3,lw=.5,label=PEAK_ORDER[j] if k==0 else None)
            a.annotate(PEAK_LABELS[PEAK_ORDER[j]],
                       xy=annotation_pos_[j],
                       color=color,
                       xycoords='data',ha='center',fontsize=9)
        for i in np.flatnonzero(m):
            fractions.append(f[i]); covariances.append(cov[i])
            rows.append({'run_id':run,'step':int(i),'T_C':float(r['T_C'][i]),
                         'P_GPa':float(r['borate_P_median'][i]),
                         'P_std_GPa':float(r['borate_P_std'][i])})
    # Une seule paire de coefficients gvar pour TOUS les runs : la covariance
    # hors diagonale décrit l'incertitude de conversion commune aux points.
    nbot,total_cov,fit_cov=nbot_from_fractions(fractions,covariances)
    total_sd=np.sqrt(np.diag(total_cov));fit_sd=np.sqrt(np.diag(fit_cov))
    for run in D:
        m=np.array([row['run_id']==run for row in rows])
        if not m.any(): continue
        T=np.array([row['T_C'] for row in rows])[m]
        b.errorbar(T,nbot[m],xerr=SIGMA_T,yerr=total_sd[m],fmt='o',
                   color=RUN_COLORS[run],ms=4,lw=.6,mec='black',mew=.3,label=RUN_LABEL[run])
        P=np.array([row['P_GPa'] for row in rows])[m]
        P_sd=np.array([row['P_std_GPa'] for row in rows])[m]
        c.errorbar(P,nbot[m],xerr=P_sd,yerr=total_sd[m],fmt='o',
                   color=RUN_COLORS[run],ms=4,lw=.6,mec='black',mew=.3)
    for i,row in enumerate(rows):
        row.update(NBO_T=float(nbot[i]),NBO_T_sd=float(total_sd[i]),NBO_T_fit_sd=float(fit_sd[i]))
    for ax in (b, c):
        ax.axhline(ANN_HYDROUS_GLASS_NBO_T, color='0.35', ls=':', lw=1, zorder=1)
        ax.annotate(f'Hydrous glass: NBO/T = {ANN_HYDROUS_GLASS_NBO_T:g}',
                    xy=(.02, ANN_HYDROUS_GLASS_NBO_T),
                    xycoords=('axes fraction', 'data'), xytext=(0, 4),
                    textcoords='offset points', ha='left', fontsize=6, color='0.35')
    b.legend(loc="upper left",fontsize=6,ncol=3)
    a.set(xlabel='SrB$_4$O$_7$:Sm$^{2+}$ median P (GPa)',
          ylabel='Peak area fraction',
          title='a)')
    b.set(xlabel='Thermocouple Temperature (°C)',
          ylabel='NBO/T',
          title='b)')
    c.set(xlabel='SrB$_4$O$_7$:Sm$^{2+}$ median P (GPa)',ylabel='NBO/T',
          title='c)')
   
    from ..raman_treatment import compare_pressure_gls
    p=np.array([row['P_GPa'] for row in rows])
    tests=compare_pressure_gls(p,nbot,total_cov)
    print('\n[Figure8] NBO/T = 2*(0.84±0.07)*f_Q2 + (1.06±0.04)*f_Q3')
    print('f_Qn = band area / sum of all FIVE band areas; no additional renormalization.')
    print(MYSEN_REFERENCE)
    print(f'N={len(rows)} Melt acquisitions; shared Gaussian conversion coefficients (assumed 1 SD).')
    print('GLS pressure tests include full covariance: spectral fits + shared conversion errors.')
    print('Fixed pressures; P_std and T/run confounding not modeled. Local Gaussian approximation.')
    for name,label,units in [('linear','Linear pressure slope','GPa^-1'),
                              ('fixed_7GPa','High-minus-low contrast at fixed 7 GPa','')]:
        t=tests[name]
        print(f'{label}: {t["effect"]:+.5f} ± {t["se"]:.5f} {units}; '
              f'95% CI [{t["CI95"][0]:+.5f}, {t["CI95"][1]:+.5f}]; p={t["p"]:.5f}')
    print('Fixed-threshold comparison does not correct for historical threshold selection.')
    png,_=save(fig,NAME,'manuscript')
    png.with_suffix('.json').write_text(json.dumps({
        'points':rows,'formula':'NBO/T = 2*c_Q2*f_Q2 + c_Q3*f_Q3',
        'hydrous_glass_reference_NBO_T':ANN_HYDROUS_GLASS_NBO_T,
        'normalization':'f_i=A_i/sum(A_TOH,A_Q2,A_T2S,A_Q3,A_Q4)',
        'reference':MYSEN_REFERENCE,'coefficients':{'Q2':MYSEN_Q2_COEFFICIENT,'Q3':MYSEN_Q3_COEFFICIENT},
        'source_status':'Coefficients/Table4 attribution supplied by user; original table not independently accessed',
        'coefficient_assumptions':'1sigma; independent Q2/Q3 coefficients, shared across acquisitions; independent from fits',
        'NBO_T_covariance':total_cov.tolist(),'NBO_T_fit_covariance':fit_cov.tolist(),
        'pressure_tests_GLS':tests,'panel_b':'NBO/T vs thermocouple temperature; horizontal errors: type K thermocouple ±2 K; vertical errors: total 1sigma',
        'thermocouple_type':'K','thermocouple_error_half_width_K':float(SIGMA_T),
        'panel_c':'NBO/T vs borate pressure, P_std horizontal errors; shared NBO/T scale with panel b',
        'limitations':'Conditional on five-band model; no extra normalization, species or hydroxy contributions; fixed P; no T/run adjustment'
    },indent=2)+'\n')
    return png


if __name__=='__main__':
    main()
