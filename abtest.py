import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from statsmodels.stats.proportion import proportions_ztest
from scipy.stats import norm, ttest_ind,ttest_ind_from_stats,t
from statsmodels.stats import weightstats
from statsmodels.stats import proportion
import math
from IPython.core.display import HTML

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

def z_test_ci(mu,std,n=1,sig_level=0.05,tail='two.sided'):
    if tail=='two.sided':
        sig_level = sig_level/2
    left,right = norm.interval(1-sig_level, loc=mu, scale=std)
    return (left,right)

def t_test_ci(mu,std,df,sig_level=0.05,tail='two.sided'):
    if tail=='two.sided':
        sig_level = sig_level/2
    left,right = t.interval(1-sig_level,df,loc=mu,scale=std)
    return (left,right)