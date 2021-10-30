import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from statsmodels.stats.proportion import proportions_ztest
from scipy.stats import norm,t
from statsmodels.stats import weightstats
from statsmodels.stats import proportion
import math
from IPython.core.display import HTML

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

def plot_CI(ax,mu,s,df=1,tail='two.sided',sig_level=0.05,color='grey',test_type='z-test'):
    if test_type=='t-test':
        left,right = t_test_ci(mu,s,df,sig_level,tail=tail)  
    else:
        left,right = z_test_ci(mu,s,sig_level,tail=tail)
    ax.axvline(left, c=color, linestyle='--', alpha=0.5)
    ax.axvline(right, c=color, linestyle='--', alpha=0.5)

def plot_norm_dist(ax,mu,std,n=1,with_CI=False,sig_level=0.05,label=None,tail='two.sided',color=None):
    x = np.linspace(mu-6*std,mu+6*std,1000)
    y = norm(mu,std).pdf(x)
    ax.plot(x,y,label=label,c=color)
    ax.axvline(mu, c=color, linestyle='--', alpha=0.5)
    if with_CI:
        plot_CI(ax,mu,std,sig_level=sig_level,tail=tail,test_type='z-test')

def plot_vbar_result(ax,data,mu_dict,delta=0.4):
    yspan = len(data)
    color = ['#c1daf0','#cccccc']
    linecolor = ['#428bca','#555555']  
    yplaces = [.5+i for i in range(yspan)]
    ylabels = list(data.keys())
    ax.set_yticks(yplaces,)
    ax.set_yticklabels(ylabels,fontsize='xx-large')
    ax.set_ylim(0,yspan)
    ax.set_facecolor('white')
    low,hi = data[ylabels[0]]
    for pos,label in zip(yplaces,ylabels):
        start,end = data[label]
        ax.add_patch(patches.Rectangle((start,pos-delta/2.0),end-start,delta))
        if start<low: low = start #loop for min of range
        if end>hi: hi= end #loop for max of range
    t=0
    for i,rec in zip(ylabels,ax.patches):
        mu = mu_dict[i]
        ax.plot([mu,mu],[rec.get_y(),rec.get_y()+rec.get_height()],
                 linewidth=3,color= linecolor[t])
        rec.set(facecolor =color[t])
        ax.text(rec.get_x()+ 0.7*rec.get_width(),
                rec.get_y()+ 0.2*rec.get_height(),'{}'.format(round(mu,5)),
               ha='center',va='bottom',fontsize = 'x-large')
        t+=1
    ax.plot((low,hi),(0,0))
    return ax

def plot_show_alpha(ax,mu,std,df,sig_level=0.05,tail='two.sided',color='green',test_type='z-test'):
    x = np.linspace(mu-8*std,mu+8*std,1000)
    if test_type =='t-test':
        left,right = t_test_ci(mu,std,df,tail=tail,sig_level=sig_level) 
        null = t.pdf(x,df,loc=mu,scale=std)
    else:
        left,right = z_test_ci(mu,std,sig_level=sig_level,tail=tail)
        null = norm(mu,std).pdf(x)
        
    if tail=='two.sided':
        ax.fill_between(x,0,null,color=color,alpha=0.25,where= (x<left) | (x>right))
    elif tail == 'greater':
        ax.fill_between(x,0,null,color=color,alpha=0.25,where= (x>right))
    elif tail == 'less':
        ax.fill_between(x,0,null,color=color,alpha=0.25,where= (x<left))

def setting_table(sig_level,alternative,labelA,labelB,diff_value):
    alternative_dict = {'two.sided':'two-sided',
                        'less':'smaller',
                        'greater':'larger'}
    if diff_value != 0 :
        hypothesis_dict = {'two.sided':'μA ≠ μB',
                           'two-sided':'μA ≠ μB',
                       'greater': 'μA > μB + {}'.format(diff_value),
                       'larger' : 'μA > μB + {}'.format(diff_value),
                       'smaller': 'μA < μB + {}'.format(diff_value),
                       'less': 'μA < μB + {}'.format(diff_value)
                       }
    else:
        hypothesis_dict = {'two.sided':'μA ≠ μB',
                           'two-sided':'μA ≠ μB',
                       'greater': 'μA > μB',
                       'larger' : 'μA > μB',
                       'smaller': 'μA < μB',
                       'less': 'μA < μB'
                       }
        
    setting_table = """<tr><th>Control Group</th><th>Variation Group</th><th>Alternative Hypothesis</th><th>Significance Level</th></tr>
                <tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>""".format(labelA,labelB,hypothesis_dict[alternative],sig_level)
    main_table_html = """
    <style>table {width: 100%;}
    td {text-align: center;}
    tr:hover {background-color: #f5f5f5;}
    </style>
    <center><h3>Test Setting</h3></center>
    <table>""" +setting_table+ "</table>"
    display(HTML(main_table_html), metadata=dict(isolated=True))

def result_table(pA=1,pB=1,power=1,p=1,tscore=1,sA=1,sB=1,stderr=1,nA=1,nB=1,df=1,sig_level=0.05):
    sig = ""
    if p<=sig_level: 
        sig = " (<{})".format(sig_level)
        star = "***"
    else:  
        sig = " (>{})".format(sig_level)
        star = ""
    kpi_table = """<tr><th>μA</th><th>μB</th><th>Uplift</th></tr>
                <tr><td>{}</td><td>{}</td><td>{}%{}</td></tr>""".format(round(pA,5),round(pB,5),round((pB-pA)/pA*100,2),star)
    n_table = """<tr><th>Sample Size A</th><th>Sample Size B</th></tr>
                <tr><td>{}</td><td>{}</td></tr>""".format(nA,nB, round(df,5) if df!=None else 'NaN')
    result_table = """<tr><th>p value</h5></th><th>t-Score/z-score</th></tr>
                <tr><td>{}</td><td>{}</td></tr>""".format(str(round(p,5)) + sig +star,round(tscore,5))
    
    main_table_html = """
    <style>table {width: 100%;}
    td {text-align: center;}
    tr:hover {background-color: #f5f5f5;}
    </style>
    <hr>
    <center><h3>Test Statistics</h3></center>
    <table>""" + kpi_table + result_table + n_table + "</table>"
    display(HTML(main_table_html), metadata=dict(isolated=True))
    
def z_test(data_frame,variant_column,control_label,variant_label,kpi_name,
                alternative,diff_value = 0, sig_level=0.05,show_plot=False,show_p_value=False,show_alpha=False
          ):
    """
    Input:
    =========
    #data_frame: Data frame for your test with each row is an observation.
    Example: You want to test average WEIGHT between group "A" and group "control"
    the data frame should look like:
        +----------+----------------+----------------------+----------------------+
        |   index  |  variant_type  |  weight_gram         |  height_cm           |
        |----------+----------------+----------------------+----------------------|
        |  1       |   A            |  100.5               |  50                  |
        |----------+----------------+----------------------+----------------------+
        |  2       |   control      |  112.9               |  50                  |
        |----------+----------------+----------------------+----------------------+
        |  3       |   control      |  102.5               |  50                  |
        |----------+----------------+----------------------+----------------------+
        |  4       |   A            |  132.7               |  50                  |
        |----------+----------------+----------------------+----------------------+
    
    #variant_column: The column contains the info of which variant the observation falls into
    In the above example it should be "variant_type"

    #control_label: Value of the control group.
    In the above example it should be "control"

    #variant_label: Value of the Variant lable.
    In the above example it should be "A"

    #kpi_name: The column name of the easurement you want to evaluate.
    In the above example it should be "weight_gram"

    
    Method:
    =========

    """
    
    control = data_frame.query('{} == "{}"'.format(variant_column,control_label))[[kpi_name]]

    variant = data_frame.query('{} == "{}"'.format(variant_column,variant_label))[[kpi_name]]
    labelA = control_label
    labelB = variant_label
    meanA = control.mean()[0]
    meanB = variant.mean()[0]
    nA = control.count()[0]
    nB = variant.count()[0]
    sA = (control.std()/math.sqrt(nA))[0]
    sB = (variant.std()/math.sqrt(nB))[0]
    
    color =  ['#428bca','#555555']  
    stderr = np.sqrt(sA**2+sB**2)
    
    ### Report Reading ###
    alternative_dict = {'two.sided':'two-sided',
                        'less':'smaller',
                        'greater':'larger'}
    
    z,p = [k[0] for k in weightstats.ztest(control, variant, value=diff_value, alternative=alternative)]

    #No Binomial Correction
    power = 'NaN'   
    ### Plot Title ###
    display(HTML("<center><h2>Two Mean Z-Test</h2> <br> for {} and {}</center>".format(labelA,labelB)))
    display(HTML("""<hr style="border-top: 2px double #8c8b8b";/> """))
    setting_table(sig_level,alternative,labelA,labelB,diff_value)
    if show_plot:
        ## Prepare ploting ###
        fig = plt.figure(figsize=(16,4),constrained_layout=True)
        spec = fig.add_gridspec(ncols=2,nrows=1,width_ratios=[4,10],height_ratios=[4])
        ax1 = fig.add_subplot(spec[0,0])
        ax1.set_title("KPI Value and Confidence Interval",fontsize='xx-large')
        ax2 = fig.add_subplot(spec[0,1],facecolor='white')
        ax2.set_title("Expected Distribution",fontsize='xx-large')
        ### Plot result ###
        mu_dict =   {labelA:meanA,labelB:meanB}
        data_dict = {labelA:z_test_ci(meanA,sA,nA,sig_level,tail=alternative),
                     labelB:(z_test_ci(meanB,sB,nB,sig_level,tail=alternative))}
        plot_vbar_result(ax1,data_dict,mu_dict)
        ### Plot Hypothesis ####
        display(HTML("<hr><center><h3>Distribution Plot</h3></center>"))
        plot_norm_dist(ax2,meanA,sA,nA,label=labelA,with_CI=True,tail=alternative,color=color[0])
        plot_norm_dist(ax2,meanB,sB,nB,label=labelB,tail=alternative,color=color[1])
        if show_alpha:
            plot_show_alpha(ax2,meanA,sA,nA,sig_level=sig_level,tail=alternative,color=color[0],test_type='z-test')
        plt.legend(loc='best')
        plt.show()
    ### Plot Table ###
    result_table(pA=meanA,pB=meanB,power=power,p=p,tscore=z,sA=sA,sB=sB,
                 stderr=stderr,nA=nA,nB=nB,df=None,sig_level=sig_level)
    display(HTML("<center><i>*** : statistical significant</i></center>"))
    display(HTML("""<hr style="border-top: 2px double #8c8b8b";/> """))

def z_proportion_test(data_frame,variant_column,control_label,variant_label,kpi_name,
                alternative,diff_value = 0, sig_level=0.05,show_plot=False,show_p_value=False,show_alpha=False
                #labelA='Control',labelB='Variation'
          ):
    
    """
    Input:
    =========
    #data_frame: Data frame for your test with each row is an observation.
    Example: You want to test converstion rate between group "A" and group "control"
    the data frame should look like:
        +----------+----------------+----------------------+----------------------+
        |   index  |  variant_type  |  converted_1_0       |  height_cm           |
        |----------+----------------+----------------------+----------------------|
        |  1       |   A            |  1                   |  50                  |
        |----------+----------------+----------------------+----------------------+
        |  2       |   control      |  0                   |  50                  |
        |----------+----------------+----------------------+----------------------+
        |  3       |   control      |  0                   |  50                  |
        |----------+----------------+----------------------+----------------------+
        |  4       |   A            |  1                   |  50                  |
        |----------+----------------+----------------------+----------------------+
    
    #variant_column: The column contains the info of which variant the observation falls into
    In the above example it should be "variant_type"

    #control_label: Value of the control group.
    In the above example it should be "control"

    #variant_label: Value of the Variant lable.
    In the above example it should be "A"

    #kpi_name: The column name of the easurement you want to evaluate.
    In the above example it should be "converted_1_0"


    """
    
    control = data_frame.query('{} == "{}"'.format(variant_column,control_label))[[kpi_name]]

    variant = data_frame.query('{} == "{}"'.format(variant_column,variant_label))[[kpi_name]]
    labelA = control_label
    labelB = variant_label
    
    cA = control.sum()[0]
    cB = variant.sum()[0]
    nA = control.count()[0]
    nB = variant.count()[0]
    pA = (cA/nA)
    pB = (cB/nB)
    count = [cA,cB]
    nobs = [nA,nB]
    sA = math.sqrt(pA*(1-pA)/nA)
    sB = math.sqrt(pB*(1-pB)/nB)
    
    color =  ['#428bca','#555555']  
    stderr = np.sqrt(sA**2+sB**2)
    ### Report Reading ###
    alternative_dict = {'two.sided':'two-sided',
                        'less':'smaller',
                        'greater':'larger'}
    
    z,p = proportion.proportions_ztest(count, nobs, value=diff_value, alternative=alternative)
    
    #No Binomial Correction
    power = 'NaN'   
    ### Plot Title ###
    display(HTML("<center><h2>Two Proportion Z-Test</h2> <br> for {} and {}</center>".format(labelA,labelB)))
    display(HTML("""<hr style="border-top: 2px double #8c8b8b";/> """))
    setting_table(sig_level,alternative,labelA,labelB,diff_value)
    if show_plot:
        ## Prepare ploting ###
        fig = plt.figure(figsize=(16,4),constrained_layout=True)
        spec = fig.add_gridspec(ncols=2,nrows=1,width_ratios=[4,10],height_ratios=[4])
        ax1 = fig.add_subplot(spec[0,0])
        ax1.set_title("KPI Value and Confidence Interval",fontsize='xx-large')
        ax2 = fig.add_subplot(spec[0,1],facecolor='white')
        ax2.set_title("Expected Distribution",fontsize='xx-large')
        ### Plot result ###
        mu_dict =   {labelA:pA,labelB:pB}
        data_dict = {labelA:z_test_ci(pA,sA,nA,sig_level,tail=alternative),
                     labelB:(z_test_ci(pB,sB,nB,sig_level,tail=alternative))}
        plot_vbar_result(ax1,data_dict,mu_dict)
        ### Plot Hypothesis ####
        display(HTML("<hr><center><h3>Distribution Plot</h3></center>"))
        plot_norm_dist(ax2,pA,sA,nA,label=labelA,with_CI=True,tail=alternative,color=color[0])
        plot_norm_dist(ax2,pB,sB,nB,label=labelB,tail=alternative,color=color[1])
        if show_alpha:
            plot_show_alpha(ax2,pA,sA,nA,sig_level=sig_level,tail=alternative,color=color[0],test_type='z-test')
        plt.legend(loc='best')
        plt.show()
    ### Plot Table ###
    result_table(pA=pA,pB=pB,power=power,p=p,tscore=z,sA=sA,sB=sB,
                 stderr=stderr,nA=nA,nB=nB,df=None,sig_level=sig_level)
    display(HTML("<center><i>*** : statistical significant</i></center>"))
    display(HTML("""<hr style="border-top: 2px double #8c8b8b";/> """))
