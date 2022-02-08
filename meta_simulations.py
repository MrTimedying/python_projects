# List of function for the on-going p-value simulation #
import pandas as pd
import random
from matplotlib import pyplot as plt
import seaborn as sn
import numpy as np
import scipy.stats as stats
from scipy.stats import norm
from numpy import std, mean, sqrt
from scipy.stats import t


def cohen_d(x,y):
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    return (mean(x) - mean(y)) / sqrt(((nx-1)*std(x, ddof=1) ** 2 + (ny-1)*std(y, ddof=1) ** 2) / dof)

def simulation_p(numerosity1, numerosity2, mean1, mean2, sd1, sd2, n1, n2):
    pop = []
    pop2 = []
    for i in range(numerosity1):
        pop.append(random.gauss(mean1, sd1))
        pop2.append(random.gauss(mean2, sd2))

    dof1 = len(pop) - 1
    dof2 = len(pop2) - 1
    confidence = 0.95
    t_crit = np.abs(t.ppf((1 - confidence) / 2, dof1))
    results = pd.DataFrame({'Effect Size': [], 'p-value': [], '.95 Gap': [], 'Upper Bound': [], 'Lower Bound': []})
    for i in range(numerosity2):
        sample1 = random.sample(pop, n1)
        sample2 = random.sample(pop2, n2)
        mean1 = np.mean(sample1)
        std1 = np.std(sample1)
        mean2 = np.mean(sample2)
        std2 = np.std(sample2)
        variance1 = np.square(std1)
        variance2 = np.square(std2)
        variance_pop = ((n1 - 1) * variance1 + (n2 - 1) * variance2) / (n1 + n2 - 2)
        standard_error = np.sqrt((variance_pop / n1) + (variance_pop / n2))
        effect_size, pvalue = stats.ttest_ind(b=sample1, a=sample2, equal_var=True)
        effect_size = cohen_d(sample2, sample1)
        upper_bound = np.absolute(mean1 - mean2) + (t_crit * standard_error)
        lower_bound = np.absolute(mean1 - mean2) - (t_crit * standard_error)
        gap = np.absolute(upper_bound - lower_bound)
        results.loc[i, :] = [effect_size, pvalue, gap, upper_bound, lower_bound]
    df = pd.DataFrame(results, columns=['Effect Size', 'p-value', '.95 Gap', 'Upper Bound', 'Lower Bound'])
    df["log10"] = np.nan
    df["log10"] = np.log10(df["p-value"])

    return (df)

def counting_p(input_value,significance):
    tensamples = input_value['p-value'].to_list()
    a = sum(i < significance for i in tensamples)
    return(a)

def histogram_distribution(input_data):
    plimit = np.log10(.05)
    plimit1 = np.log10(.01)
    plimit2 = np.log10(.001)
    g3 = sn.histplot(data=input_data, x="log10", color="#FFF", edgecolor="black", bins=58)
    plt.axvline(plimit, 0,.9, color="Crimson", lw=.6)
    plt.axvline(plimit1, 0,.9, color="black", ls='dotted', lw=.6)
    plt.axvline(plimit2, 0,.9, color="black", ls='dashed', lw=.6)
    plt.text(plimit,-5.2,'.05',rotation=0, color="Crimson")
    plt.text(plimit1,-5.2,'.01',rotation=0, color="black")
    plt.text(plimit2,-5.2,'.001',rotation=0, color="black")
    g3.axes.yaxis.set_visible(False)
    g3.axes.xaxis.set_visible(False)

    for rectangle in g3.patches:
        if rectangle.get_x() >= plimit:
            rectangle.set_facecolor('GhostWhite')

    for rectangle in g3.patches:
        if rectangle.get_x() <= plimit:
            rectangle.set_facecolor('Crimson')
    return(g3)