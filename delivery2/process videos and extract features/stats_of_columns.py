import pandas as pd
import numpy as np
import pickle
import statistics
from statistics import mean
import math
from scipy.stats import entropy
from math import log2



def shannon(array):
    total = sum(array.values) 
    return sum(freq / total * log2(total / freq) for freq in array.values)[0]



pickle_file = pd.read_pickle('whole_set_selected2.pkl')

ear_list = pickle_file['avg_ear'].values.tolist()
left_ear_list = pickle_file['left_ear'].values.tolist()
right_ear_list = pickle_file['right_ear'].values.tolist()
mar_list = pickle_file['mar'].values.tolist()
moe_list = pickle_file['moe'].values.tolist()
perclos_list = pickle_file['perclos'].values.tolist()
leb_list = pickle_file['avg_leb'].values.tolist()
left_leb_list = pickle_file['left_leb'].values.tolist()
right_leb_list = pickle_file['right_leb'].values.tolist()
sop_list = pickle_file['avg_sop'].values.tolist()
left_sop_list = pickle_file['left_sop'].values.tolist()
right_sop_list = pickle_file['right_sop'].values.tolist()
ec_list = pickle_file['avg_eye_circularity'].values.tolist()
left_ec_list = pickle_file['left_eye_circularity'].values.tolist()
right_ec_list = pickle_file['right_eye_circularity'].values.tolist()
drowsiness_list = pickle_file['drowsiness'].values.tolist()
frame_no_list = pickle_file['frame_no'].values.tolist()
face_detected_list = pickle_file['face_detected'].values.tolist()
closeness_list = pickle_file['closeness'].values.tolist()

miss_ear = ear_list.count(-1)
miss_left_ear = left_ear_list.count(-1)
miss_right_ear = right_ear_list.count(-1)
miss_mar = mar_list.count(-1)
miss_moe = moe_list.count(-1)
miss_perclos = perclos_list.count(-1)
miss_leb = leb_list.count(-1)
miss_left_leb = left_leb_list.count(-1)
miss_right_leb = right_leb_list.count(-1)
miss_sop = sop_list.count(-1)
miss_left_sop = left_sop_list.count(-1)
miss_right_sop = right_sop_list.count(-1)
miss_ec = ec_list.count(-1)
miss_left_ec = ec_list.count(-1)
miss_right_ec = ec_list.count(-1)
miss_drowsiness = drowsiness_list.count(-1)
miss_frame_no = frame_no_list.count(-1)
miss_face_detected = face_detected_list.count(-1)
miss_closeness = closeness_list.count(-1)

ear_list[:] = [item for item in ear_list if item != -1]
left_ear_list[:] = [item for item in left_ear_list if item != -1]
right_ear_list[:] = [item for item in right_ear_list if item != -1]
mar_list[:] = [item for item in mar_list if item != -1]
moe_list[:] = [item for item in moe_list if item != -1]
perclos_list[:] = [item for item in perclos_list if item != -1]
leb_list[:] = [item for item in leb_list if item != -1]
left_leb_list[:] = [item for item in left_leb_list if item != -1]
right_leb_list[:] = [item for item in right_leb_list if item != -1]
sop_list[:] = [item for item in sop_list if item != -1]
left_sop_list[:] = [item for item in left_sop_list if item != -1]
right_sop_list[:] = [item for item in right_sop_list if item != -1]
ec_list[:] = [item for item in ec_list if item != -1]
left_ec_list[:] = [item for item in left_ec_list if item != -1]
right_ec_list[:] = [item for item in right_ec_list if item != -1]
drowsiness_list[:] = [item for item in drowsiness_list if item != -1]
frame_no_list[:] = [item for item in frame_no_list if item != -1]
face_detected_list[:] = [item for item in face_detected_list if item != -1]
closeness_list[:] = [item for item in closeness_list if item != -1]

print("\n######## NUMMERICAL COLUMNS ##########")
print(f"\nFRAME_NO: \nMIN: {min(frame_no_list)} \nMAX: {max(frame_no_list)} \nAVG: {mean(frame_no_list)} \nSTDEV: {statistics.stdev(frame_no_list)} \nMISSED_VALUES: {miss_frame_no} \nENTROPY: {entropy(frame_no_list, base=2)}")
print(f"\nFACE_DETECTED: \nMIN: {min(face_detected_list)} \nMAX: {max(face_detected_list)} \nAVG: {mean(face_detected_list)} \nSTDEV: {statistics.stdev(face_detected_list)} \nMISSED_VALUES: {miss_face_detected} \nENTROPY: {entropy(face_detected_list, base=2)}")
print(f"\nAVG_EAR: \nMIN: {min(ear_list)} \nMAX: {max(ear_list)} \nAVG: {mean(ear_list)} \nSTDEV: {statistics.stdev(ear_list)} \nMISSED_VALUES: {miss_ear} \nENTROPY: {entropy(ear_list, base=2)}")
print(f"\nLEFT_EAR: \nMIN: {min(left_ear_list)} \nMAX: {max(left_ear_list)} \nAVG: {mean(left_ear_list)} \nSTDEV: {statistics.stdev(left_ear_list)} \nMISSED_VALUES: {miss_left_ear} \nENTROPY: {entropy(left_ear_list, base=2)}")
print(f"\nRIGHT_EAR: \nMIN: {min(right_ear_list)} \nMAX: {max(right_ear_list)} \nAVG: {mean(right_ear_list)} \nSTDEV: {statistics.stdev(right_ear_list)} \nMISSED_VALUES: {miss_right_ear} \nENTROPY: {entropy(right_ear_list, base=2)}")
print(f"\nMAR: \nMIN: {min(mar_list)} \nMAX: {max(mar_list)} \nAVG: {mean(mar_list)} \nSTDEV: {statistics.stdev(mar_list)} \nMISSED_VALUES: {miss_mar} \nENTROPY: {entropy(mar_list, base=2)}")
print(f"\nMOE: \nMIN: {min(moe_list)} \nMAX: {max(moe_list)} \nAVG: {mean(moe_list)} \nSTDEV: {statistics.stdev(moe_list)} \nMISSED_VALUES: {miss_moe} \nENTROPY: {entropy(moe_list, base=2)}")
print(f"\nPERCLOS: \nMIN: {min(perclos_list)} \nMAX: {max(perclos_list)} \nAVG: {mean(perclos_list)} \nSTDEV: {statistics.stdev(perclos_list)} \nMISSED_VALUES: {miss_perclos} \nENTROPY: {entropy(perclos_list, base=2)}")
print(f"\nAVG_LEB: \nMIN: {min(leb_list)} \nMAX: {max(leb_list)} \nAVG: {mean(leb_list)} \nSTDEV: {statistics.stdev(leb_list)} \nMISSED_VALUES: {miss_leb} \nENTROPY: {entropy(leb_list, base=2)}")
print(f"\nLEFT_LEB: \nMIN: {min(left_leb_list)} \nMAX: {max(left_leb_list)} \nAVG: {mean(left_leb_list)} \nSTDEV: {statistics.stdev(left_leb_list)} \nMISSED_VALUES: {miss_left_leb} \nENTROPY: {entropy(left_leb_list, base=2)}")
print(f"\nRIGHT_LEB: \nMIN: {min(right_leb_list)} \nMAX: {max(right_leb_list)} \nAVG: {mean(right_leb_list)} \nSTDEV: {statistics.stdev(right_leb_list)} \nMISSED_VALUES: {miss_right_leb} \nENTROPY: {entropy(right_leb_list, base=2)}")
print(f"\nAVG_SOP: \nMIN: {min(sop_list)} \nMAX: {max(sop_list)} \nAVG: {mean(sop_list)} \nSTDEV: {statistics.stdev(sop_list)} \nMISSED_VALUES: {miss_sop} \nENTROPY: {entropy(sop_list, base=2)}")
print(f"\nLEFT_SOP: \nMIN: {min(left_sop_list)} \nMAX: {max(left_sop_list)} \nAVG: {mean(left_sop_list)} \nSTDEV: {statistics.stdev(left_sop_list)} \nMISSED_VALUES: {miss_left_sop} \nENTROPY: {entropy(left_sop_list, base=2)}")
print(f"\nRIGHT_SOP: \nMIN: {min(right_sop_list)} \nMAX: {max(right_sop_list)} \nAVG: {mean(right_sop_list)} \nSTDEV: {statistics.stdev(right_sop_list)} \nMISSED_VALUES: {miss_right_sop} \nENTROPY: {entropy(right_sop_list, base=2)}")
print(f"\nAVG_EC: \nMIN: {min(ec_list)} \nMAX: {max(ec_list)} \nAVG: {mean(ec_list)} \nSTDEV: {statistics.stdev(ec_list)} \nMISSED_VALUES: {miss_ec} \nENTROPY: {entropy(ec_list, base=2)}")
print(f"\nLEFT_EC: \nMIN: {min(left_ec_list)} \nMAX: {max(left_ec_list)} \nAVG: {mean(left_ec_list)} \nSTDEV: {statistics.stdev(left_ec_list)} \nMISSED_VALUES: {miss_left_ec} \nENTROPY: {entropy(left_ec_list, base=2)}")
print(f"\nRIGHT_EC: \nMIN: {min(right_ec_list)} \nMAX: {max(right_ec_list)} \nAVG: {mean(right_ec_list)} \nSTDEV: {statistics.stdev(right_ec_list)} \nMISSED_VALUES: {miss_right_ec} \nENTROPY: {entropy(right_ec_list, base=2)}")
print(f"\nCLOSENESS: \nMIN: {min(closeness_list)} \nMAX: {max(closeness_list)} \nAVG: {mean(closeness_list)} \nSTDEV: {statistics.stdev(closeness_list)} \nMISSED_VALUES: {miss_closeness} \nENTROPY: {entropy(closeness_list, base=2)}")
print(f"\nDROWSINESS: \nMIN: {min(drowsiness_list)} \nMAX: {max(drowsiness_list)} \nAVG: {mean(drowsiness_list)} \nSTDEV: {statistics.stdev(drowsiness_list)} \nMISSED_VALUES: {miss_drowsiness} \nENTROPY: {entropy(drowsiness_list, base=2)}")

print("\n######## NOMINAL COLUMNS ##########")
subject_list = pickle_file['subject'].values.tolist()
countings_df=pd.DataFrame([[x,subject_list.count(x)] for x in set(subject_list)], columns=['subject', 'count']).set_index('subject')
print("\nsubject:")
print("MIN: {}".format(countings_df.idxmin().values[0]))
print("MAX: {}".format(countings_df.idxmax().values[0]))
print("ENTROPY: {}".format(shannon(countings_df)))

factors_list = pickle_file['external_factors'].values.tolist()
countings_df2=pd.DataFrame([[x,factors_list.count(x)] for x in set(factors_list)], columns=['factor', 'count']).set_index('factor')
print("\nfactors_list:")
print("MIN: {}".format(countings_df2.idxmin().values[0]))
print("MAX: {}".format(countings_df2.idxmax().values[0]))
print("ENTROPY: {}".format(shannon(countings_df2)))

actions_list = pickle_file['facial_actions'].values.tolist()
countings_df3=pd.DataFrame([[x,actions_list.count(x)] for x in set(actions_list)], columns=['actions', 'count']).set_index('actions')
print("\nfacial_actions:")
print("MIN: {}".format(countings_df3.idxmin().values[0]))
print("MAX: {}".format(countings_df3.idxmax().values[0]))
print("ENTROPY: {}".format(shannon(countings_df3)))

reserved_list = pickle_file['reserved_for_calibration'].values.tolist()
countings_df4=pd.DataFrame([[x,reserved_list.count(x)] for x in set(reserved_list)], columns=['reserved', 'count']).set_index('reserved')
print("\reserved_for_calibration:")
print("MIN: {}".format(countings_df4.idxmin().values[0]))
print("MAX: {}".format(countings_df4.idxmax().values[0]))
print("ENTROPY: {}".format(shannon(countings_df4)))

