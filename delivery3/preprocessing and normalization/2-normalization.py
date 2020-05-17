import os
import pandas as pd
import numpy as np

df = pd.read_pickle('whole_set_selected2_preprocessed.pkl')

df = df.reset_index(drop=True)

# # drop some -1s in columns due to multiple face detected
# df = df_original.loc[df_original['face_detected'] == 1]

# # drop some -1s in drowsiness column
# df = df[np.logical_or(df['drowsiness']== 1, df['drowsiness']== 0)]

# # drop -1s in perclos column
# df = df[df['reserved_for_calibration']== False]
# df.eq(-1).all()

user_list = list(df['subject'].unique())


list_of_all_original=[]
for user in user_list:
    list_of_all_original.append(df.groupby('subject').get_group(user))

df_nonsleepycombination_and_mix = df[df['facial_actions'].isin(['nonsleepyCombination','mix'])]


list_of_all_groups=[]
for user in user_list:
    list_of_all_groups.append(df_nonsleepycombination_and_mix.groupby('subject').get_group(user))


list_of_all_first_90 = []
for user in user_list:
    list_of_all_first_90.append(df_nonsleepycombination_and_mix.groupby('subject').get_group(user)[:90])


##### standart scaler çalıştırma
from sklearn.preprocessing import StandardScaler

en_son_df_list=[]
for i in range(len(list_of_all_first_90)):
    scaler = StandardScaler()
    scaler.fit(list_of_all_first_90[i].loc[ : , ["left_ear", "right_ear", "avg_ear", 
                                              "left_eye_circularity", 
                                              "right_eye_circularity","avg_eye_circularity", "left_leb", "right_leb", 
                                              "avg_leb","left_sop", "right_sop", "avg_sop",
                                              "EAR","EC","LEB","SOP","MAR","MOE"]])
    son_df=pd.DataFrame(scaler.transform(list_of_all_original[i].loc[ : , ["left_ear", "right_ear", "avg_ear", 
                                              "left_eye_circularity", 
                                              "right_eye_circularity","avg_eye_circularity", "left_leb", "right_leb", 
                                              "avg_leb","left_sop", "right_sop", "avg_sop",
                                              "EAR","EC","LEB","SOP","MAR","MOE"]]))
    
    
    
    son_df.columns=["n_left_ear", "n_right_ear", "n_avg_ear", 
                    "n_left_eye_circularity", 
                    "n_right_eye_circularity","n_avg_eye_circularity", "n_left_leb", "n_right_leb", 
                    "n_avg_leb","n_left_sop", "n_right_sop", "n_avg_sop",
                    "n_EAR","n_EC","n_LEB","n_SOP","n_MAR","n_MOE"]
    
    
    son_df['subject']=user_list[i]    
    en_son_df_list.append(son_df[91:]) 


# scale 021 by using EAR
for i in range(len(list_of_all_first_90)):
    mean=list_of_all_first_90[i].loc[:,'EAR'].mean()
    std=list_of_all_first_90[i].loc[:,'EAR'].std()
    val = 0.21
    n_val = (val-mean)/std
    print (mean, std, val, n_val)
    en_son_df_list[i].loc[:,'n_021'] = n_val
    
con_df = pd.concat(en_son_df_list)

merged = pd.concat([df.reset_index(drop=True), con_df.reset_index(drop=True)], axis=1)

merged.to_pickle("whole_set_selected2_preprocessed_normalized.pkl")