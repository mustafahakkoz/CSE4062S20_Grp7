import os
import pandas as pd

def read_file(filename):
    content=[]
    f = open(filename, 'r')
    while True:
        char=f.read(1)
        if not char: 
            break
        content.append(int(char))
    return content


#################### MAIN #############
df_list=[]
directory = "C:/Users/hakko/Desktop/NTHU-DDD/Training_Evaluation_Dataset/Training Dataset"
subjects = os.listdir(directory)
for subject in subjects:
    sub_folders=os.listdir(directory+'/'+subject)
    for sub_folder in sub_folders:
        if sub_folder == 'sunglasses':
            continue
        annotations=os.listdir(directory+'/'+subject+'/'+sub_folder)
        for annot in annotations:
            clean_name = os.path.splitext(annot)[0]
            extension = os.path.splitext(annot)[1]
            if extension=='.txt' and clean_name.endswith('drowsiness'): 
                video_name=clean_name.split('_')[1]
                file_path = directory+'/'+subject+'/'+sub_folder+'/'+annot
                print(file_path)
                list_of_dicts=[]
                content=read_file(file_path)
                for i in content:  
                    annot_dict = {
                        'subject': subject,
                        'external_factors': sub_folder,
                        'facial_actions': video_name,
                        'drowsiness': i,
                        }
                    list_of_dicts.append(annot_dict)
                content_df=pd.DataFrame(list_of_dicts)
                df_list.append(content_df)
                content_df.to_pickle('./train/{}_{}_{}_annotations.pkl'.format(subject,sub_folder,video_name))
                            
# result_df = pd.concat(df_list)               
# result_df.to_pickle('annotations_training.pkl')
      

