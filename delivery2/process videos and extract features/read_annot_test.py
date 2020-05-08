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
directory = "C:/Users/hakko/Desktop/NTHU-DDD/Training_Evaluation_Dataset/Evaluation Dataset"
subjects = os.listdir(directory)
for subject in subjects:
    annotations=os.listdir(directory+'/'+subject)
    for annot in annotations:
        clean_name = os.path.splitext(annot)[0]
        extension = os.path.splitext(annot)[1]
        if extension=='.txt' and clean_name.endswith('drowsiness'): 
            external_factors =clean_name.split('_')[1]
            video_name='mix'
            if external_factors == 'sunglasses':
                continue
            if external_factors == 'night':
               external_factors =clean_name.split('_')[1]+'_'+clean_name.split('_')[2]
               video_name='mix'
            file_path = directory+'/'+subject+'/'+annot
            print(file_path)
            list_of_dicts=[]
            content=read_file(file_path)
            for i in content:  
                annot_dict = {
                    'subject': subject,
                    'external_factors': external_factors,
                    'facial_actions': video_name,
                    'drowsiness': i,
                    }
                list_of_dicts.append(annot_dict)
            content_df=pd.DataFrame(list_of_dicts)
            df_list.append(content_df)
            content_df.to_pickle('./test/{}_{}_{}_annotations.pkl'.format(subject,external_factors,video_name))
                
# result_df = pd.concat(df_list)               
# result_df.to_pickle('annotations_test.pkl')