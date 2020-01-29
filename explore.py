from prep import prep_df, train_test_split, split_train_and_test
from statsmodels.graphics.gofplots import qqplot
from sklearn.preprocessing import StandardScaler
from baseline import baseline_metrics
from matplotlib import pyplot as plt
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
# Average cholestrol: 246, STD:51
sns.countplot(df.sex)
# Sex counts skewed towards Males. 
sns.countplot(df.target)
# Target about equal split between those with and without heart disease: 54/46

def check_normal_dist(df):
    if len(df) >= 5000:
        raise ValueError("Shaprio-Wilks test should be used on data less than 5000 values")
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
dft.blood_sugar.astype(int).sum()/len(dft.blood_sugar)# 15.9% has high blood sugar


# Statistical Testing-------------------
dft.

scipy.stats.chisquare()
degrees of freedom: (rows - 1) * (cols - 1)


#Female
df2 = df[df.sex == 0]
df2.describe() 
# Average age 55, STD: 9.4 - Above combined average. 
# Average max heart rate: 151 , STD: 20, Above combined average.
# Average blood pressure: 133, STD: 19, Above combined average.
df2.blood_sugar.astype(int).sum()/len(df2.blood_sugar)# 12.5% has high blood sugar

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


