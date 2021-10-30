import pandas as pd
import numpy as np
from statsmodels.stats.proportion import proportions_ztest
from scipy.stats import norm, ttest_ind,ttest_ind_from_stats,t

def ztest(cA,cB,nA,nB,sig_level=0.5, alternative='two-sided'):
    z,p = proportions_ztest([cA,cB],[nA,nB],alternative=alternative, prop_var=False)
    if p <= sig_level:
        t = '***'
    else:
        t = ''
    diff = (cA/nA - cB/nB)*100
    up = ((cA/nA)/(cB/nB)-1)*100
    return (diff,up,t)

def ttest(pA,pB,nA,nB,sA,sB,sig_level=0.05, alternative = 'two.sided'):          
    n = nA+nB
    p_diff = pA-pB
    p_up = (pA/pB-1)*100
    stderr = np.sqrt(sA**2/nA+sB**2/nB)
    df = (sA**2/nA + sB**2/nB)**2 / ((sA**2/nA)**2 / (nA-1) + (sB**2/nB)**2 / (nB-1))
    t,p = ttest_ind_from_stats(pA,sA,nA,
                               pB,sB,nB,
                               equal_var=False)  
    if (alternative == 'two.sided' and p <= sig_level) or (alternative == 'greater' and p/2 <= sig_level and t>0) or (alternative == 'less' and p/2 <= sig_level and t<0):
        txt = '***'
    else:
        txt = ''
    return (p_diff,p_up,txt)
    