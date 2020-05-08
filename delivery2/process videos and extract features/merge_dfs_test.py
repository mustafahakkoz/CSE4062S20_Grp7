import os
import pandas as pd


annots=[]
frame_infos=[]
video_infos=[]
frame_dfs=[]
annots_dfs=[]
frame_dfs_names=[]
annots_dfs_names=[]
count=0
directory = "./test/"
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
            frame_info_df = pd.read_pickle('./test/'+file2)
            frame_dfs.append(frame_info_df)
            frame_dfs_names.append(file2)
            annotation = pd.read_pickle('./test/'+file)
            annots_dfs.append(annotation)
            annots_dfs_names.append(file)
            if len(frame_info_df) !=len(annotation):
                print(file2, len(frame_info_df))
                print(file, len(annotation))
                count+=1
            else: 
                os.rename('./test/'+file, './fix3/'+file)
                os.rename('./test/'+file2, './fix3/'+file2)
                drowsiness_series=annotation.loc[:,'drowsiness']
                frame_info_df.loc[:,'drowsiness']=drowsiness_series
                frame_info_df.to_pickle('./test/'+first_part+'_merged_df.pkl')
                
print(count)

# deneme=pd.read_pickle('./test/'+'030_glasses_mix_merged_df.pkl')

# df1=frame_dfs[13]
# df2=annots_dfs[13]
# drowsiness_series=df2.loc[:,'drowsiness']
# df1.loc[:,'drowsiness']=drowsiness_series


# deneme0=pd.read_pickle('./test/'+'030_glasses_mix_frame_info_df.pkl')
# deneme1=deneme0.drop(columns=['frame_no'])
# duplicateRowsDF = deneme1[:10][deneme1[:10].duplicated()]
# print("Duplicate Rows except first occurrence based on all columns are :")
# print(duplicateRowsDF)

# print([ele for ele in frame_infos if ele not in frame_dfs_names])
# print([ele for ele in annots if ele not in annots_dfs_names])
# deneme12.iloc[0]
# deneme = pd.read_pickle('./test/'+'004_night_noglasses_mix_frame_info_df.pkl')
# deneme2 = pd.read_pickle('./test/'+'004_night_noglasses_mix_annotations.pkl')
# deneme3=pd.read_pickle('./test/'+'004_glasses_mix_frame_info_df.pkl')
# deneme4=pd.read_pickle('./test/'+'004_glasses_mix_annotations.pkl')

# deneme5=pd.read_pickle('./train/'+'001_night_noglasses_nonsleepyCombination_frame_info_df.pkl')
# deneme6=pd.read_pickle('./train/'+'001_night_noglasses_nonsleepyCombination_annotations.pkl')
# deneme7=pd.read_pickle('./train/'+'001_noglasses_nonsleepyCombination_frame_info_df.pkl')
# deneme8=pd.read_pickle('./train/'+'001_noglasses_nonsleepyCombination_annotations.pkl')

# deneme9=pd.read_pickle('./train/'+'031_night_noglasses_nonsleepyCombination_frame_info_df.pkl')
# deneme10=pd.read_pickle('./train/'+'031_night_noglasses_nonsleepyCombination_annotations.pkl')
# deneme11=pd.read_pickle('./train/'+'031_noglasses_nonsleepyCombination_frame_info_df.pkl')
# deneme12=pd.read_pickle('./train/'+'031_noglasses_nonsleepyCombination_annotations.pkl')