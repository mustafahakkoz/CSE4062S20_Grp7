import os
import pandas as pd
import numpy as np

directory = "./test/"
pickles = os.listdir(directory)
pickle_list=[]

for pickle_file in pickles:
    clean_name = os.path.splitext(pickle_file)[0]
    if clean_name.endswith("merged_df"):
        pickle = pd.read_pickle("./test/"+pickle_file)
        pickle_list.append(pickle)

result = pd.concat(pickle_list)
result.reset_index(inplace=True, drop=True)
result.to_pickle("test_set.pkl")
# print(result.iloc[0])

directory = "./train/"
pickles = os.listdir(directory)
pickle_list2=[]

for pickle_file in pickles:
    clean_name = os.path.splitext(pickle_file)[0]
    if clean_name.endswith("merged_df"):
        pickle = pd.read_pickle("./train/"+pickle_file)
        pickle_list2.append(pickle)

result2 = pd.concat(pickle_list2)
result2.reset_index(inplace=True, drop=True)
result2.to_pickle("training_set.pkl")

result3 = pd.concat([result, result2])
result3.reset_index(inplace=True, drop=True)
# result3.loc[:,'drowsiness']=result3.loc[:,'drowsiness'].apply(lambda x: int(x))
result3.to_pickle("whole_set.pkl")

result4 = result3.drop(columns=['blink_no', 'blink_start_frame', 'blink_end_frame'])
# result4.loc[:,'closeness']=np.where(result4.loc[:,'avg_ear'] < 0.21, 1,0)
# result4.loc[:,'closeness']=np.where(result4.loc[:,'avg_ear'] == -1, -1,result4.loc[:,'closeness'])
result4.to_pickle("whole_set_selected.pkl")