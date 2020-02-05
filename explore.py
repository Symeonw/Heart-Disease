from prep import prep_df, train_test_split, split_train_and_test
from hd_stats import check_normal_dist, check_chi
from statsmodels.graphics.gofplots import qqplot
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from scipy.stats import ttest_ind
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
check_normal_dist(df)# All continuous variable have a normal dist. 




# General Exploration---------------------------------------------------------------------


# Break Down by Age Group 


# Comparing Male vs Female
# Male
dft = df[df.sex == 1]
dft.describe() 
# Average age 53, STD: 8.8 - below combined average.
# Average max heart rate: 148, STD: 24 - below combined average
# Average blood pressure: 130, STD: 16 - below combined averge
# Average cholesterol: 239, STD: 42.7 - below combined average
dft.blood_sugar.astype(int).sum()/len(dft.blood_sugar)# 15.9% have high blood sugar

#Female
df2 = df[df.sex == 0]
df2.describe() 
# Average age 55, STD: 9.4 - Above combined average. 
# Average max heart rate: 151 , STD: 20, Above combined average.
# Average blood pressure: 133, STD: 19, Above combined average.
# Average cholesterol: 261, STD: 65, Above combined average
df2.blood_sugar.astype(int).sum()/len(df2.blood_sugar)# 12.5% have high blood sugar


# Testing based on Heart Disease-----------------------------------
df[df.target == 0].describe()
len(df[df.target == 0])# 138 Samples
# Average blood pressure: 134
# Average cholesterol: 251
# Average max heart rate: 139
df[df.target == 1].describe()
len(df[df.target == 1])# 165 Samples 
# Average blood pressure: 129
# Average cholesterol: 242
# Average max heart rate: 158

#Assumptions to test:
# 1. Patients with heart disease have higher blood pressure
# 2. Patients with heart disease have higher cholesterol
# 3. Patients with heart disease have hight max heart rates


# H0: Patients with heart disease have no significant difference 
# H0 Patients with heart disease have no significant difference in blood pressure.
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

# H0: Patients with heart disease have no significant difference in max heart rate
clear_heart = df[df.target == 0]["max_heart_rate"]
check_normal_dist(clear_heart)# Not Normally Distributed
has_heart = df[df.target == 1]["max_heart_rate"]
check_normal_dist(has_heart) # Normally Distributed
mannwhitneyu(clear_heart, has_heart)
# P-Value: 4.89e-14, H0 Rejected

clear_peak = df[df.target == 0]["oldpeak"]
check_normal_dist(clear_peak)
has_peak = df[df.target == 1]["oldpeak"]
check_normal_dist(has_peak)
ttest_ind(clear_peak, has_peak)
# P-Value: 4.08e-15, H0 Rejected

#Chi tests on categorical variables
check_chi(df.target, df.cp)# Variables are dependent of Target
check_chi(df.target, df.ca)# Variables are dependent of Target
check_chi(df.target, df.blood_sugar) # Variables are independent of Target
check_chi(df.target, df.restecg) # Variables are dependent of Target
check_chi(df.target, df.exang) # Variables are dependent of Target
check_chi(df.target, df.slope) # Variables are dependent of Target
check_chi(df.target, df.thal) # Variables are dependent of Target


#Results:
# blood_sugar column was determined to be statistically insignificant while 
#   all other category columns were determined to be significant. 
# All three assuptions where statistically verified:
    # 1. Patients with heart disease have higher blood pressure
    # 2. Patients with heart disease have higher cholesterol
    # 3. Patients with heart disease have hight max heart rates



