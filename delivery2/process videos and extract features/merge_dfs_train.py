import os
import pandas as pd


annots=[]
frame_infos=[]
video_infos=[]

directory = "./train/"
files = os.listdir(directory)
for file in files:
    clean_name = os.path.splitext(file)[0]
    if clean_name.endswith('annotations'):
        annots.append(file)
    if clean_name.endswith('video_info_df'):
        video_infos.append(file)
    if clean_name.endswith('frame_info_df'):
        frame_infos.append(file)

for file in annots:
    clean_name = os.path.splitext(file)[0]
    first_part = clean_name[:-12]
    
    for file2 in frame_infos:
        clean_name2 = os.path.splitext(file2)[0]
        first_part2 = clean_name2[:-14]
        if first_part == first_part2:
            frame_info_df = pd.read_pickle('./train/'+file2)
            annotation = pd.read_pickle('./train/'+file)
            if len(frame_info_df) !=len(annotation):
                print(file2, len(frame_info_df))
                print(file, len(annotation))
            else: 
                os.rename('./train/'+file, './fix4/'+file)
                os.rename('./train/'+file2, './fix4/'+file2)
                drowsiness_series=annotation.loc[:,'drowsiness']
                frame_info_df.loc[:,'drowsiness']=drowsiness_series
                frame_info_df.to_pickle('./train/'+first_part+'_merged_df.pkl')
                
# deneme = pd.read_pickle('./train/'+'015_glasses_sleepyCombination_merged_df.pkl')
                
# deneme1 = pd.read_pickle('./train/'+'015_glasses_nonsleepyCombination_frame_info_df.pkl')
# deneme2 = pd.read_pickle('./train/'+'015_glasses_nonsleepyCombination_annotations.pkl')     

   