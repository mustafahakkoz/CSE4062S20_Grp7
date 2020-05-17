import pandas as pd
import numpy as np

df_original = pd.read_pickle('whole_set_selected2.pkl')

# drop some -1s in columns due to multiple face detected
df = df_original.loc[df_original['face_detected'] == 1]

# drop some -1s in drowsiness column
df = df[np.logical_or(df['drowsiness']== 1, df['drowsiness']== 0)]

# change missing perclos values to 0
df.loc[df['reserved_for_calibration']== True, 'perclos']=0

# check if are there any -1 in all columns
df.eq(-1).all()

# theres no -1 left so reset indexes
df = df.reset_index(drop=True)

# eye_columns=["left_ear", "right_ear", "avg_ear", 
#          "left_eye_circularity", "right_eye_circularity","avg_eye_circularity", 
#          "left_leb", "right_leb", "avg_leb",
#          "left_sop", "right_sop", "avg_sop"]

# check if leaft and right eyes' values are differ each other
diff = df['left_ear']-df['right_ear']
np.where(np.logical_and(diff>0.21, df['avg_ear']<0.21))

z_score = (diff - diff.mean())/diff.std(ddof=0)
np.where(z_score>3)
np.where(np.logical_and(z_score>3, df['avg_ear']<0.21))

# some samples which are one eye is 0, are detected.
print(df.iloc[285643][['left_ear','right_ear','avg_ear']])

# check if is there more 0 value
a=np.where(np.logical_and(df['left_ear']==0, df['right_ear']!=0))
b=np.where(np.logical_and(df['left_ear']!=0, df['right_ear']==0))

# yes, there are more
for i in a:
    print(df.iloc[i][['left_ear','right_ear','avg_ear']])
    
for j in b:
    print(df.iloc[j][['left_ear','right_ear','avg_ear']])
    
# use not 0 column instead of avg_ear
df['EAR'] = df['avg_ear']
for i in a:
    df.loc[i,'EAR'] = df['right_ear']
for j in b:
    df.loc[i,'EAR'] = df['left_ear']

# do the same with other eye columns
a=np.where(np.logical_and(df['left_eye_circularity']==0, df['right_eye_circularity']!=0))
b=np.where(np.logical_and(df['left_eye_circularity']!=0, df['right_eye_circularity']==0))
df['EC'] = df['avg_eye_circularity']
for i in a:
    df.loc[i,'EC'] = df['right_eye_circularity']
for j in b:
    df.loc[i,'EC'] = df['left_eye_circularity']
    
a=np.where(np.logical_and(df['left_leb']==0, df['right_leb']!=0))
b=np.where(np.logical_and(df['left_leb']!=0, df['right_leb']==0))
df['LEB'] = df['avg_leb']
for i in a:
    df.loc[i,'LEB'] = df['right_leb']
for j in b:
    df.loc[i,'LEB'] = df['left_leb']
    
a=np.where(np.logical_and(df['left_sop']==0, df['right_sop']!=0))
b=np.where(np.logical_and(df['left_sop']!=0, df['right_sop']==0))
df['SOP'] = df['avg_sop']
for i in a:
    df.loc[i,'SOP'] = df['right_sop']
for j in b:
    df.loc[i,'SOP'] = df['left_sop']

# check for more anormal differences
np.where(np.logical_and(z_score>3, df['avg_ear']<0.3))

# one sample. What can we do about it? I guess it's better to use avg_ear for cases like these.
print(df.iloc[25360][['left_ear','right_ear','avg_ear']])

# check if is there 0 in mar, moe, perclos and closeness
df['mar'].describe()
len(np.where(df['mar']==0)[0])

df['moe'].describe()
len(np.where(df['moe']==0)[0])

df['perclos'].describe()
len(np.where(df['perclos']==0)[0])

df['closeness'].describe()
len(np.where(df['closeness']==0)[0])

# MAR is acceptable. 
# closeness will be updated after normalization nothing can be done for now.
# fixing perclos should be done in process_video.py along with all the steps in preprocessing
# it's possible to fix MOE with new values of EAR
# first rename columns
df.rename(columns={"mar": "MAR", "moe": "MOE", "perclos": "PERCLOS", "closeness": "CLOSENESS", "drowsiness":"DROWSINESS"}, inplace=True)

df['MOE'] = df ['MAR'] / df['EAR']

df.to_pickle("whole_set_selected2_preprocessed.pkl")


