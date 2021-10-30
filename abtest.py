from statsmodels.stats.proportion import proportions_ztest

def ztest(cA,cB,nA,nB,sig_level=0.5, alternative='two-sided'):
    z,p = proportions_ztest([cA,cB],[nA,nB],alternative=alternative, prop_var=False)
    if p <= sig_level:
        t = '***'
    else:
        t = ''
    diff = (cA/nA - cB/nB)*100
    up = ((cA/nA)/(cB/nB)-1)*100
    return (diff,up,t)