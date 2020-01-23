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
sns.countplot(df.sex)
# Sex counts skewed towards Males. 
sns.countplot(df.target)
# Target about equal split between those with and without heart disease: 54/46

def check_normal_dist(df):
    df = df.loc[:, df.dtypes != "category"]
    cols = df.columns.tolist()
    for col in cols:
        stat, p = shapiro(df[col])
        if p > 0.05:
            print(col)
    print("finished")

check_normal_dist(df)# All continuous variable have a normal dist. 

# Principal Component Analysis (PCA)
from sklearn.decomposition import PCA
scaler = StandardScaler()
df_cont = df.loc[:, df.dtypes != "category"]
df_cont = scaler.fit_transform(df_cont)
pca = PCA(n_components=2)
pcs = pca.fit_transform(df_cont)
pcdf = pd.DataFrame(data = pcs, columns = ["principal component 1", "principal component 2"])
pcdf = pd.concat([PCDF, df[["target"]]], axis = 1)
pcdf

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

# Multiple Corrospondence Analysis 
# import prince
# dfc = df.loc[:, df.dtypes == "category"]
# mca = prince.MCA(n_components=2,n_iter=3,copy=True,check_input=True,\
#     engine="auto",random_state=123)
# mca = mca.fit(dfc)

# ax = mca.plot_coordinates(dfc,ax=None,figsize=(6, 6),show_row_points=True,\
#     row_points_size=10,show_row_labels=False,show_column_points=True,\
#         column_points_size=30,show_column_labels=False,legend_n_cols=1)

# Factor Analysis of Mixed Data
df_cont = df.loc[:, df.dtypes != "category"]
dfc = df.loc[:, df.dtypes == "category"]
dfs = pd.DataFrame(scaler.fit_transform(df_cont))
dfs = pd.concat([dfs,dfc], axis = 1)
famd = prince.FAMD(n_components=3,n_iter=3,copy=True,check_input=True,\
    engine="auto",random_state=123)
famd = famd.fit(dfs.drop("target", axis="columns"))
ax = famd.plot_row_coordinates(dfs,ax=None,figsize=(6, 6),\
    x_component=0,y_component=1,\
        color_labels=[f"Target {t}" for t in dfs.target],\
            ellipse_outline=False,ellipse_fill=True,show_points=True)

