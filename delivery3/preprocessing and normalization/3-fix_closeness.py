import pandas as pd
import numpy as np

df = pd.read_pickle('whole_set_selected2_preprocessed_normalized.pkl')

df.drop(columns=['CLOSENESS'], inplace=True)
df.loc[:,'CLOSENESS']=np.where(df['n_EAR'] < df['n_021'], 1,0)

df.to_pickle('whole_set_selected2_preprocessed_normalized_fixed.pkl')