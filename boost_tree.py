import pandas as pd
import numpy as np
np.random.seed(0)
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from collections import Counter
from catboost import CatBoostClassifier
from tqdm import tqdm


validation = True
######################################
# Data Preprocessing
######################################

Data_set = pd.read_csv('challenge_data/Data_set.csv')
Test_set = pd.read_csv('challenge_data/Test_set.csv')

X = Data_set.iloc[:,2:]
y = Data_set.iloc[:,1]
X_test = Test_set.iloc[:,2:]
id_test = Test_set.iloc[:,0]
X = X.fillna(0)
X_test = X_test.fillna(0)

cat_feature_inds = []
cat_unique_thresh = 100
for i, c in enumerate(X.columns):
    num_uniques = len(X[c].unique())
    if num_uniques < cat_unique_thresh \
       and not 'CNT' in c \
       and not 'AMT' in c \
       and not 'NUM' in c \
       and not 'RATE' in c \
       and not 'PREM' in c \
       and not 'AGE' in c \
       and not 'ARPU' in c \
       and not 'PRIN' in c \
       and not 'MDIF' in c \
       and not 'INCM' in c \
       and not 'DATE' in c:
        cat_feature_inds.append(i)
print(cat_feature_inds)

def str_to_int(train, test):
    train = train.fillna(0)
    test = test.fillna(0)
    idx = 0
    for item in train.unique():
        train.loc[train==item] = idx
        test.loc[test==item] = idx
        idx += 1
    return train, test


string_column = []
for i,col in enumerate(X.columns):
    if X[col].dtype != np.float64 and X[col].dtype != np.int64:
        string_column.append(i)
        X[col], X_test[col] = str_to_int(X[col], X_test[col])


if validation:
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0)
else:
    X_train = X
    y_train = y

#print(Counter(y_train))
#sm = SMOTE(random_state=42)
#X_train, y_train = sm.fit_sample(X_train, y_train)
#print(Counter(y_train))

print(Counter(y_train))
sm = SMOTETomek()
X_train, y_train = sm.fit_sample(X_train, y_train)
print(Counter(y_train))

if validation:
    X_train, X_val, y_train, y_val = map(np.array, [X_train, X_val, y_train, y_val])
else:
    X_train, y_train = map(np.array, [X_train, y_train])

######################################
# Train Model
######################################

def f1_measure(con_mat):
    precision = con_mat[1][1] / (con_mat[0][1]+con_mat[1][1])
    accuracy = con_mat[1][1] / (con_mat[1][0]+con_mat[1][1])
    return 2 * (precision*accuracy) / (precision+accuracy)

bst_f0, bst_pred = 0, 0
itr = 400
lr = 0.03
dpt = 7
nem = 5
l2r = 3
class_weight=[1,2]
cat_use=False

if cat_use:
    for i in cat_feature_inds:
        X_train[:,i] = list(map(int, X_train[:,i]))
        if validation:
            X_val[:,i] = list(map(int, X_val[:,i]))

print("SETUP: nem: %d, itr: %d, lr: %.2f, dpt: %d, l2r: %d, weight: [%.1f,%.1f], cat_use: %s" % 
     (nem, itr, lr, dpt, l2r, *class_weight, cat_use))
y_val_pred = 0
y_pred = 0
num_ensembles = nem
for i in tqdm(range(num_ensembles)):
    model = CatBoostClassifier(
        iterations=itr, learning_rate=lr,
        depth=dpt, l2_leaf_reg=l2r,
        class_weights=class_weight,
        random_seed=i)
    if cat_use:
        model.fit(
            X_train, y_train,
            cat_features=cat_feature_inds)
    else:
        model.fit(
            X_train, y_train)
    if validation:
        y_val_pred += model.predict(X_val)
    y_pred += model.predict(X_test)
y_pred /= num_ensembles
y_pred = list(map(int, y_pred))
if validation:
    y_val_pred /= num_ensembles
    y_val_pred = list(map(int, y_val_pred))
    error_rate = np.sum(y_val_pred != y_val) / float(y_val.shape[0])
    score = f1_measure(confusion_matrix(y_val, y_val_pred))
    print('Test error = {}'.format(error_rate))
    print("f1_measure = %f" % score)
    print("confusion matrix")
    print(confusion_matrix(y_val, y_val_pred))

######################################
# Save Result
######################################

id_test = np.array(id_test).reshape(-1,1)
y_pred = np.array(y_pred).reshape(-1, 1)

answer = pd.DataFrame(np.concatenate([id_test, y_pred], axis=1))

answer.to_csv('challenge_data/Answer_sheet.csv', index=False, header=False)

