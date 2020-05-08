import os
import pandas as pd


directory = "./train/"
files = os.listdir(directory)
for file in files:
    clean_name = os.path.splitext(file)[0]
    if clean_name.endswith('frame_info_df'):
        os.rename('./train/'+file, './fix/'+file)
        frame_info_df = pd.read_pickle('./fix/'+file)
        frame_info_df.loc[:,'facial_actions']=frame_info_df.loc[:,'facial_actions'].apply(lambda x: x[:-4] if x.endswith('.avi') else x)
        frame_info_df.to_pickle('./train/'+file)

        
        
# frame_info_df = pd.read_pickle('./train/'+'036_noglasses_yawning_frame_info_df.pkl')
        
        
