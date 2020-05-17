import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.utils import resample
sns.set(font_scale=2)
df_original = pd.read_pickle('../5.normalization/whole_set_selected2_preprocessed_normalized_fixed.pkl')
# df = df_original.loc[np.logical_or(df_original['DROWSINESS'] == 1, df_original['DROWSINESS'] == 0)]
df = df_original.loc[:, ["n_EAR", 
                    "n_MAR", "n_MOE", "n_EC",
                    "n_LEB", "n_SOP", "PERCLOS", "CLOSENESS", "DROWSINESS"]]



# plot distributions
plt.figure(figsize=(20,20))
for i in range(1, 10):
    plt.subplot(3, 3, i)
    sns.distplot(df[df.columns[i-1]],bins=10,kde_kws={'bw':0.1})
    
# resample to prevent overlappings then display scatter plots of features
r_df = resample(df, n_samples=5000, replace=False, stratify=df['DROWSINESS'], random_state=0)
g = sns.pairplot(data=r_df, hue="DROWSINESS", palette="muted",diag_kind="kde", plot_kws={'size':0.1, 'alpha': 0.3}, diag_kws={'bw':0.1})

# calculate correlation
y = df['DROWSINESS']
X = df.drop(columns=["DROWSINESS"])
x_corr = X.corr()

# correlation matrix
sns.set(font_scale=1)
plt.figure(figsize=(12,9))
sns.heatmap(x_corr, annot=True, fmt="f",vmin=-1, vmax=1, linewidths=.5, cmap = sns.color_palette("RdBu", 100))
plt.yticks(rotation=0)
plt.show()
