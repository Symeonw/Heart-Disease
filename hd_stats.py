from scipy.stats import chi2_contingency
from scipy.stats import mannwhitneyu
from scipy.stats import shapiro
from scipy.stats import chi2
from math import sqrt
import pandas as pd
import numpy as np



def check_normal_dist(df):
    if len(df) >= 5000:
        raise ValueError("Shaprio-Wilks test should be used on data less than 5000 values")
    try:
        df = df.loc[:, df.dtypes != "category"]
    except:
        df = pd.DataFrame(df)
        df = df.loc[:, df.dtypes != "category"]
    cols = df.columns.tolist()
    n = 0
    for col in cols:
        stat, p = shapiro(df[col])
        if p > 0.05:
            n += 1
            print(f"Not normally distributed: {col}")
    if n > 0:
        print(f"finished with {n} variables being not normally distributed")
    else:
        print("All variables are normally distributed.")


def create_chi_table():
    p = np.array([0.995, 0.99, 0.975, 0.95, 0.90, 0.10, 0.05, 0.025, 0.01, 0.005])
    df = np.array(list(range(1, 30)) + list(range(30, 101, 10))).reshape(-1, 1)
    np.set_printoptions(linewidth=130, formatter=dict(float=lambda x: "%7.3f" % x))
    table = chi2.isf(p, df)
    return table


def check_chi(var1,var2):
    chi_inp = pd.crosstab(var1, var2)
    chi2, p, dof, expected = chi2_contingency(chi_inp.values)
    if np.array(i >= 5 for i in expected).all() == False:
        raise ValueError("""Expected frequency did not render expected value beyond 5,
        please gather additional data or use different variables.""")
    if create_chi_table()[dof-1][6] > chi2 or p > .05 :
        print("These variables are independent, failed to reject H0.")
    else:
        print("These variables are dependent, H0 rejected.")