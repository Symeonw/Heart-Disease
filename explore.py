from prep import prep_df, train_test_split, split_train_and_test
from statsmodels.graphics.gofplots import qqplot
from sklearn.preprocessing import StandardScaler
from scipy.stats import chi2_contingency
from baseline import baseline_metrics
from matplotlib import pyplot as plt
from scipy.stats import mannwhitneyu
from scipy.stats import ttest_ind
from scipy.stats import shapiro
from scipy import stats
import seaborn as sns
import pandas as pd 
import numpy as np
import warnings



warnings.filterwarnings("ignore")
df = prep_df()
df.describe()
# Average Age: 54, STD: 9
# Average max heart rate: 149. STD: 22
# Average blood pressure: 131, STD: 17
# Average cholesterol: 246, STD:51
sns.countplot(df.sex)
# Sex counts skewed towards Males. 
sns.countplot(df.target)
# Target about equal split between those with and without heart disease: 54/46

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
    print(f"finished {n} variables are not normally distributed")

check_normal_dist(df)# All continuous variable have a normal dist. 

# General Exploration


# Break Down by Age Group 


# Comparing Male vs Female
# Male
dft = df[df.sex == 1]
dft.describe() 
# Average age 53, STD: 8.8 - below combined average.
# Average max heart rate: 148, STD: 24 - below combined average
# Average blood pressure: 130, STD: 16 - below combined averge
# Average cholesterol: 239, STD: 42.7 - below combined average
dft.blood_sugar.astype(int).sum()/len(dft.blood_sugar)# 15.9% has high blood sugar

#Female
df2 = df[df.sex == 0]
df2.describe() 
# Average age 55, STD: 9.4 - Above combined average. 
# Average max heart rate: 151 , STD: 20, Above combined average.
# Average blood pressure: 133, STD: 19, Above combined average.
# Average cholesterol: 261, STD: 65, Above combined average
df2.blood_sugar.astype(int).sum()/len(df2.blood_sugar)# 12.5% has high blood sugar

# Statistical Testing on Males and Females without heart disease-------------------

# H0: Both Males and Females (without heart disease) have no
#    significant difference in blood pressure

bpf = df[(df.sex == 0) & (df.target == 0)]["blood_pressure"]
check_normal_dist(bpf)# Not Normally Distributed
bpm = df[(df.sex == 1) & (df.target == 0)]["blood_pressure"]
check_normal_dist(bpm)# Normally Distributed
mannwhitneyu(bpm, bpf)
#Conclusion: P-Value: 0.0009, H0 Rejected. 


# H0: Both Males and Females (without heart disease) have no
#   significant differences in cholesterol levels
chol_f = df[(df.sex == 0) & (df.target == 0)]["chol"]
check_normal_dist(chol_f)# Not Normally Distributed
chol_m = df[(df.sex == 1) & (df.target == 0)]["chol"]
check_normal_dist(chol_m) # Not Normally Distributed
mannwhitneyu(chol_m, chol_f)
#Conclusion: P-Value: 0.02, H0 is rejected. 


# Testing based on Heart Disease
df[df.target == 0].describe()
# Average blood pressure: 134
# Average cholesterol: 251
# Average max heart rate: 139
df[df.target == 1].describe()
# Average blood pressure: 129
# Average cholesterol: 242
# Average max heart rate: 158

#Assumptions to test:
# 1. Patients with heart disease have higher blood pressure
# 2. Patients with heart disease have higher cholesterol
# 3. Patients with heart disease have hight max heart rates


# H0: Patients with heart disease have no significant difference 
#   in blood pressure.

clear_bp = df[df.target == 0]["blood_pressure"]
check_normal_dist(clear_bp)# Normally Distributed
has_bp = df[df.target == 1]["blood_pressure"]
check_normal_dist(has_bp) # Normally Distributed
ttest_ind(clear_bp, has_bp)
# P-Value: 0.01; H0 rejected.

# H0: Patients with heart disease have no significant difference in cholesterol levels
clear_chol = df[df.target == 0]["chol"]
check_normal_dist(clear_chol)# Not Normally Distributed
has_chol = df[df.target == 1]["chol"]
check_normal_dist(has_chol) # Normally Distributed
mannwhitneyu(clear_chol, has_chol)
# P-Value: 0.01, H0 Rejected

clear_heart = df[df.target == 0]["max_heart_rate"]
check_normal_dist(clear_heart)# Not Normally Distributed
has_heart = df[df.target == 1]["max_heart_rate"]
check_normal_dist(has_heart) # Normally Distributed
mannwhitneyu(clear_heart, has_heart)
# P-Value: 4.89e-14, H0 Rejected


chi_cp = pd.crosstab(df.target,df.cp)
chi2, p, dof, expected = chi2_contingency(chi_cp.values)
print(f"Chi2: {chi2}, P-value: {p}, Degrees of Freedom: {dof}, Expected Values: {expected} ")




#NOTES: 
# The t-test assumes that the means of the different samples are 
# normally distributed; it does not assume that the population is normally distributed.
# By the central limit theorem, means of samples from a population with 
# finite variance approach a normal distribution regardless of the distribution 
# of the population. Rules of thumb say that the sample means are basically normally 
# distributed as long as the sample size is at least 20 or 30


chi_thal = pd.crosstab(df.sex,df.thal)
chi2, p, dof, expected = chi2_contingency(chi_thal.values)
print(f"Chi2: {chi2}, P-value: {p}, Degrees of Freedom: {dof}, Expected Values: {expected} ")

chi_blood_pressure = pd.crosstab(df.sex,df.blood_pressure)
chi2, p, dof, expected = chi2_contingency(chi_blood_pressure.values)
print(f"Chi2: {chi2}, P-value: {p}, Degrees of Freedom: {dof}, Expected Values: {expected} ")

chi_cp = pd.crosstab(df.sex,df.cp)
chi2, p, dof, expected = chi2_contingency(chi_cp.values)
print(f"Chi2: {chi2}, P-value: {p}, Degrees of Freedom: {dof}, Expected Values: {expected} ")



degrees of freedom: (rows - 1) * (cols - 1)



# Comparing Age
dfa_bin = pd.cut(df.age, 4, labels=["28-41","41-53","53-65", "65-77"])
dfa = df.copy()
dfa.age = dfa_bin
# Analysis for ages 28 to 41 years old
dft = dfa[dfa.age == "28-41"]
check_normal_dist(df)
dft.sex.value_counts()# 20 Males, 9 Females - 29 total subjects
dft.target.value_counts()# 22 Positive for Heart Disease; 7 Negetive
dft.blood_pressure.mean()# 124.13
dft.chol.mean()# 215.58
dft.blood_sugar
# Statistical Testing






# Principal Component Analysis (PCA)
from sklearn.decomposition import PCA
scaler = StandardScaler()
df_cont = df.loc[:, df.dtypes != "category"]
df_cont = scaler.fit_transform(df_cont)
pca = PCA(n_components=2)
pcs = pca.fit_transform(df_cont)
pcdf = pd.DataFrame(data = pcs, columns = ["principal component 1", "principal component 2"])
pcdf = pd.concat([PCDF, df[["target"]]], axis = 1)

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel("Principal Component 1", fontsize = 15)
ax.set_ylabel("Principal Component 2", fontsize = 15)
ax.set_title("2 component PCA", fontsize = 20)
targets = [0, 1]
colors = ["r", "b"]
for target, color in zip(targets,colors):
    indicesToKeep = pcdf["target"] == target
    ax.scatter(pcdf.loc[indicesToKeep, "principal component 1"]
               , pcdf.loc[indicesToKeep, "principal component 2"]
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()

pca.explained_variance_ratio_
# Principal Component 1 accounts for 36% of the variance 
# Principal Component 2 accounts for 21% of the variance
# The 2 Principal Components explain 57% of the variance of the data.
# Conclusion: Explained variance is not enough to be useful for further modeling or exploration. 


