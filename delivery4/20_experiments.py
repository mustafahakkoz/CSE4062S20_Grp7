import os 
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, roc_curve, confusion_matrix, classification_report, roc_auc_score, auc
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score, cross_validate, StratifiedKFold
from sklearn.model_selection import GridSearchCV
import sys 
from sklearn.model_selection import learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB, CategoricalNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve, average_precision_score, make_scorer, precision_score, recall_score, f1_score
from scipy import stats
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFECV
import matplotlib.pyplot as plt
import seaborn as sns
from statistics import mean 


################ UTILITY FUNCTIONS ###################
# display test scores
def t_test(results_array, model_names, score_name):
    string=""
    for i in range(len(model_names) - 1):
           for j in range(i, len(model_names)):
               if i == j:
                   continue
               t, p = stats.ttest_ind(results_array[i], results_array[j], equal_var=False)
               string += "\n"+score_name
               string += "T_Test between {} & {}: T Value = {}, P Value = {}".format(model_names[i], model_names[j], t, p)
               if p>0.05:
                   string += "p-value>0.05 so there's NO significant difference between models."
               else:
                   string += "p-value<=0.05 so there's A significant difference between models." 
    return string

# plot ROC 
def plot_roc(fpr,tpr,model_name,selector_name):
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % auc(fpr,tpr))
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(model_name + "/" + selector_name)
    plt.legend(loc="lower right")
    
################ NORMALIZATION ###################
df_original = pd.read_pickle('final_step2.pkl')

X = df_original.loc[:, ["n_EAR", 
                    "n_MAR", "n_MOE", "n_EC",
                    "n_LEB", "n_SOP", "PERCLOS", "CLOSENESS"]]

y = df_original.loc[:, "DROWSINESS"]

# normalize each columns
scaler = StandardScaler()
scaler.fit(X)
X_scaled=scaler.transform(X)

################ MODEL and SCORE DEFINITIONS ###################
models = []
models.append(('KNN-5', KNeighborsClassifier(n_neighbors=5, n_jobs=-1)))
models.append(('CART-gini', DecisionTreeClassifier(criterion="gini"))) 
models.append(('NB', GaussianNB()))
models.append(('KNN-25', KNeighborsClassifier(n_neighbors=25, n_jobs=-1)))
models.append(('CART-entropy', DecisionTreeClassifier(criterion="entropy"))) 

scoring = []
scoring.append(('accuracy', accuracy_score))
scoring.append(('prec', precision_score))
scoring.append(('recall', recall_score))
scoring.append(('f1', f1_score))
scoring.append(('auc', roc_auc_score))


################ VARIABLES ###################
fold_info_list = []
table1_output = ""
fold_no = 0

################ OUTER CV FOR T-TEST (5 FOLD) ###################
outer_cv = StratifiedKFold(n_splits=5,random_state=42, shuffle=True)
for train_index, test_index in outer_cv.split(X_scaled, y): 
    X_train, X_test, y_train, y_test = X_scaled[train_index], X_scaled[test_index], y[train_index], y[test_index]
    
    ############# FEATURE SELECTION (4 METHODS: FULL, ANOVA, MI, RFE-RF) #############
    table1_output += "OUTER CV FOLD NO: {}\n".format(fold_no)
    X_train_FULL = X_train
    
    ANOVA_selector = SelectKBest(f_classif, k=5)
    X_train_ANOVA = ANOVA_selector.fit_transform(X_train, y_train)
    print("ANOVA scores: {}, ANOVA p-values: {}".format(ANOVA_selector.scores_, ANOVA_selector.pvalues_))
    table1_output += "ANOVA scores: {}, ANOVA p-values: {}".format(ANOVA_selector.scores_, ANOVA_selector.pvalues_)
    
    MI_selector = SelectKBest(mutual_info_classif, k=5)
    X_train_MI = MI_selector.fit_transform(X_train, y_train)
    print("MI scores: {}, MI p-values: {}".format(MI_selector.scores_, MI_selector.pvalues_))
    table1_output += "MI scores: {}, MI p-values: {}".format(MI_selector.scores_, MI_selector.pvalues_)
    
    #estimator for recursive feature elimination
    # estimator = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', class_weight= None, max_features = None, random_state = 42,n_jobs=-1)
    estimator = DecisionTreeClassifier(random_state = 42)
    #inner cv for recursive feature elimination
    inner_cv = StratifiedShuffleSplit(n_splits=2,test_size=0.2, random_state = 42)
    RFE_selector = RFECV(estimator, step=1, cv=inner_cv, n_jobs=-1)
    X_train_RFE = RFE_selector.fit_transform(X_train, y_train)
    print("RFE rankings: {}, RFE grid-scores: {}".format(RFE_selector.ranking_, RFE_selector.grid_scores_))
    table1_output += "RFE rankings: {}, RFE grid-scores: {}".format(RFE_selector.ranking_, RFE_selector.grid_scores_)
    
    ############# TEST RESULTS FOR FEATURE SELECTION METHODS #############
    fold_info = {
        model_name:{
            'FULL': {
                score_name:0 for score_name,score in scoring
                },
            'ANOVA': {
                score_name:0 for score_name,score in scoring
                },
            'MI': {
                score_name:0 for score_name,score in scoring
                },
            'RFE': {
                score_name:0 for score_name,score in scoring
                }
            } for model_name, model in models
        }
    for model_name, model in models:
        # transform X_test according to feature selection methods
        X_test_FULL = X_test
        X_test_ANOVA = ANOVA_selector.transform(X_test)
        X_test_MI = MI_selector.transform(X_test)
        X_test_RFE = RFE_selector.transform(X_test)
        # make predictions
        model.fit(X_train_FULL,y_train)
        y_pred_FULL = model.predict(X_test_FULL)
        model.fit(X_train_ANOVA,y_train)
        y_pred_ANOVA = model.predict(X_test_ANOVA)
        model.fit(X_train_MI,y_train)
        y_pred_MI = model.predict(X_test_MI)
        model.fit(X_train_RFE,y_train)
        y_pred_RFE = model.predict(X_test_RFE)
        # save evaluation metrics
        for score_name, score in scoring:
            score_FULL = score(y_test,y_pred_FULL)
            score_ANOVA = score(y_test,y_pred_ANOVA)
            score_MI = score(y_test,y_pred_MI)
            score_RFE = score(y_test,y_pred_RFE)
            fold_info[model_name]['FULL'][score_name] = score_FULL,
            fold_info[model_name]['ANOVA'][score_name] = score_ANOVA,
            fold_info[model_name]['MI'][score_name] = score_MI,
            fold_info[model_name]['RFE'][score_name] = score_RFE,

        # save ROC metrics
        fpr_FULL, tpr_FULL, _ = roc_curve(y_test,y_pred_FULL)
        fold_info[model_name]['FULL']['ROC_fpr'] = fpr_FULL
        fold_info[model_name]['FULL']['ROC_tpr'] = tpr_FULL
        fpr_ANOVA, tpr_ANOVA, _ = roc_curve(y_test,y_pred_ANOVA)
        fold_info[model_name]['ANOVA']['ROC_fpr'] = fpr_ANOVA
        fold_info[model_name]['ANOVA']['ROC_tpr'] = tpr_ANOVA
        fpr_MI, tpr_MI, _ = roc_curve(y_test,y_pred_MI)
        fold_info[model_name]['MI']['ROC_fpr'] = fpr_MI
        fold_info[model_name]['MI']['ROC_tpr'] = tpr_MI
        fpr_RFE, tpr_RFE, _ = roc_curve(y_test,y_pred_RFE)
        fold_info[model_name]['RFE']['ROC_fpr'] = fpr_RFE
        fold_info[model_name]['RFE']['ROC_tpr'] = tpr_RFE
        
        # save confusion matrix
        conf_FULL = confusion_matrix(y_test,y_pred_FULL)
        fold_info[model_name]['FULL']['confusion_matrix'] = conf_FULL
        conf_ANOVA = confusion_matrix(y_test,y_pred_ANOVA)
        fold_info[model_name]['ANOVA']['confusion_matrix'] = conf_ANOVA
        conf_MI = confusion_matrix(y_test,y_pred_MI)
        fold_info[model_name]['MI']['confusion_matrix'] = conf_MI
        conf_RFE = confusion_matrix(y_test,y_pred_RFE)
        fold_info[model_name]['RFE']['confusion_matrix'] = conf_RFE
            
            
    fold_info_list.append(fold_info)
    fold_no += 1 

############# T-TEST #############

results = {
    "FULL": {score_name:[] for score_name,score in scoring},
    "ANOVA": {score_name:[] for score_name,score in scoring},
    "MI": {score_name:[] for score_name,score in scoring},
    "RFE": {score_name:[] for score_name,score in scoring},
    }

# build result arrays for corresponding metrics
for score_name, score in scoring:
    for model_name, model in models:
        model_results_FULL = []
        model_results_ANOVA = []
        model_results_MI = []
        model_results_RFE = []
        for fold_info in fold_info_list:
             model_results_FULL.append(fold_info[model_name]['FULL'][score_name])
             model_results_ANOVA.append(fold_info[model_name]['ANOVA'][score_name])
             model_results_MI.append(fold_info[model_name]['MI'][score_name])
             model_results_RFE.append(fold_info[model_name]['RFE'][score_name])
        results['FULL'][score_name].append(model_results_FULL)
        results['ANOVA'][score_name].append(model_results_ANOVA)
        results['MI'][score_name].append(model_results_MI)
        results['RFE'][score_name].append(model_results_RFE)

# calculate t-test on 5 folds for corresponding model and metric
t_test_output=""
model_names = [model_name for model_name, model in models]
for score_name, score in scoring:
        t_test_output += '\nFULL'
        results_array = results['FULL'][score_name]
        t_test_output += t_test(results_array, model_names, score_name)
        t_test_output += '\nANOVA'
        results_array = results['ANOVA'][score_name]
        t_test_output += t_test(results_array, model_names, score_name)
        t_test_output += '\nMI'
        results_array = results['MI'][score_name]
        t_test_output += t_test(results_array, model_names, score_name)
        t_test_output += '\nRFE'
        results_array = results['RFE'][score_name]
        t_test_output += t_test(results_array, model_names, score_name)

textfile = open('t-test.txt', 'w')
textfile.write(t_test_output)
textfile.close()   

############ EVALUATION METRICS #####################

metrics_output = ""
for model_name, model in models:
    metrics_output += "\n{} with full features\n".format(model_name)
    for score_name, score in scoring:
        metrics_output += "{}: {},{},{},{},{}, average: {}".format(score_name, *[fold_info[model_name]["FULL"][score_name][0] for fold_info in fold_info_list], mean([fold_info[model_name]["FULL"][score_name][0] for fold_info in fold_info_list]))
    metrics_output += "\n{} with ANOVA\n".format(model_name)
    for score_name, score in scoring:
        metrics_output += "{}: {},{},{},{},{}, average: {}".format(score_name, *[fold_info[model_name]["ANOVA"][score_name][0] for fold_info in fold_info_list], mean([fold_info[model_name]["ANOVA"][score_name][0] for fold_info in fold_info_list]))
    metrics_output += "\n{} with MI\n".format(model_name)
    for score_name, score in scoring:
        metrics_output += "{}: {},{},{},{},{}, average: {}".format(score_name, *[fold_info[model_name]["MI"][score_name][0] for fold_info in fold_info_list], mean([fold_info[model_name]["MI"][score_name][0] for fold_info in fold_info_list]))
    metrics_output += "\n{} with RFE\n".format(model_name)
    for score_name, score in scoring:
        metrics_output += "{}: {},{},{},{},{}, average: {}".format(score_name, *[fold_info[model_name]["RFE"][score_name][0] for fold_info in fold_info_list], mean([fold_info[model_name]["RFE"][score_name][0] for fold_info in fold_info_list]))

textfile2 = open('metrics.txt', 'w')
textfile2.write(metrics_output)
textfile2.close()   




metrics_output2 = ""
for model_name, model in models:
    metrics_output += "\n{} with full features\n".format(model_name)
    for score_name, score in scoring:
        metrics_output += "{}: {}".format(score_name, mean([fold_info[model_name]["FULL"][score_name] for fold_info in fold_info_list]))
    metrics_output += "\n{} with ANOVA\n".format(model_name)
    for score_name, score in scoring:
        metrics_output += "{}: {}".format(score_name, mean([fold_info[model_name]["ANOVA"][score_name][0] for fold_info in fold_info_list]))
    metrics_output += "\n{} with MI\n".format(model_name)
    for score_name, score in scoring:
        metrics_output += "{}: {}".format(score_name, mean([fold_info[model_name]["MI"][score_name][0] for fold_info in fold_info_list]))
    metrics_output += "\n{} with RFE\n".format(model_name)
    for score_name, score in scoring:
        metrics_output += "{}: {}".format(score_name, mean([fold_info[model_name]["RFE"][score_name][0] for fold_info in fold_info_list]))

textfile3 = open('metrics2.txt', 'w')
textfile3.write(metrics_output2)
textfile3.close()   
        
        
############ ROC CURVES #########################
# plot curves in grid
plt.figure(figsize=(20,20))
for i in range(0, 5):
    plt.subplot(5, 4, 4*i+1)
    fpr = np.mean([fold_info[model_names[i]]['FULL']['ROC_fpr'] for fold_info in fold_info_list], axis=0)
    tpr = np.mean([fold_info[model_names[i]]['FULL']['ROC_tpr'] for fold_info in fold_info_list], axis=0)
    plot_roc(fpr,tpr,model_names[i],"FULL features")
    
    plt.subplot(5, 4, 4*i+2)
    fpr = np.mean([fold_info[model_names[i]]['ANOVA']['ROC_fpr'] for fold_info in fold_info_list], axis=0)
    tpr = np.mean([fold_info[model_names[i]]['ANOVA']['ROC_tpr'] for fold_info in fold_info_list], axis=0)
    plot_roc(fpr,tpr,model_names[i],"ANOVA selector")
    
    plt.subplot(5, 4, 4*i+3)
    fpr = np.mean([fold_info[model_names[i]]['MI']['ROC_fpr'] for fold_info in fold_info_list], axis=0)
    tpr = np.mean([fold_info[model_names[i]]['MI']['ROC_tpr'] for fold_info in fold_info_list], axis=0)
    plot_roc(fpr,tpr,model_names[i],"Mutual Info selector")
    
    plt.subplot(5, 4, 4*i+4)
    fpr = np.mean([fold_info[model_names[i]]['RFE']['ROC_fpr'] for fold_info in fold_info_list], axis=0)
    tpr = np.mean([fold_info[model_names[i]]['RFE']['ROC_tpr'] for fold_info in fold_info_list], axis=0)
    plot_roc(fpr,tpr,model_names[i],"RFE selector with RF")
    
    
        
############ CONFUSION MATRICES #########################    
sns.set(font_scale=2.0) # for label size


for no, fold_info in enumerate(fold_info_list,1):
    plt.figure(figsize=(20,20))
    plt.subplots_adjust(wspace = 0.5, hspace = 0.5)

    # plt.title("FOLD NO: {}".format(no))
    for i in range(0, 5):
        plt.subplot(5, 4, 4*i+1)
        plt.title(model_names[i] + "/" + "FULL")
        df_cm = pd.DataFrame(fold_info[model_names[i]]['FULL']['confusion_matrix'])
        sns.heatmap(df_cm, annot=True, fmt="d",cbar=False, annot_kws={"size": 18}) # font size
        
        plt.subplot(5, 4, 4*i+2)
        plt.title(model_names[i] + "/" + "ANOVA")
        df_cm = pd.DataFrame(fold_info[model_names[i]]['ANOVA']['confusion_matrix'])
        sns.heatmap(df_cm, annot=True, fmt="d",cbar=False, annot_kws={"size": 18}) # font size
        
        plt.subplot(5, 4, 4*i+3)
        plt.title(model_names[i] + "/" + "MI")
        df_cm = pd.DataFrame(fold_info[model_names[i]]['MI']['confusion_matrix'])
        sns.heatmap(df_cm, annot=True, fmt="d",cbar=False, annot_kws={"size": 18}) # font size
        
        plt.subplot(5, 4, 4*i+4)
        plt.title(model_names[i] + "/" + "RFE")
        df_cm = pd.DataFrame(fold_info[model_names[i]]['RFE']['confusion_matrix'])
        sns.heatmap(df_cm, annot=True, fmt="d", cbar=False, annot_kws={"size": 18}) # font size






