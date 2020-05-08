import os
import pandas as pd


directory = "./test/"
files = os.listdir(directory)
for file in files:
    clean_name = os.path.splitext(file)[0]
    if clean_name.endswith('annotations'):
        os.rename('./test/'+file, './fix2/'+file)
        annot = pd.read_pickle('./fix2/'+file)
        subject=annot.loc[0,'subject']
        external_factors=annot.loc[0,'external_factors']
        facial_actions=annot.loc[0,'facial_actions']
        drowsiness=annot.loc[0,'drowsiness']
        new_df=pd.DataFrame({'subject': [subject,subject], 'external_factors': [external_factors,external_factors], 'facial_actions':[facial_actions,facial_actions], 'drowsiness':[-1,-1]})
        new_df = new_df.append(annot)
        new_df = new_df.reset_index(drop=True)
        new_df.to_pickle('./test/'+file)


# frame_info_df = pd.read_pickle('./test/'+'004_glasses_mix_annotations.pkl')
# subject=frame_info_df.loc[0,'subject']
# external_factors=frame_info_df.loc[0,'external_factors']
# facial_actions=frame_info_df.loc[0,'facial_actions']
# drowsiness=frame_info_df.loc[0,'drowsiness']
# new_df=pd.DataFrame({'subject': [subject,subject], 'external_factors': [external_factors,external_factors], 'facial_actions':[facial_actions,facial_actions], 'drowsiness':[-1,-1]})
# new_df = new_df.append(frame_info_df)
# new_df = new_df.reset_index(drop=True)