
# import libraries
import pandas as pd # for data manupulation or analysis
import numpy as np # for numeric calculation


from sklearn.datasets import load_breast_cancer
cancer_dataset = load_breast_cancer()

cancer_df = pd.DataFrame(np.c_[cancer_dataset['data'],cancer_dataset['target']],
             columns = np.append(cancer_dataset['feature_names'], ['target']))

X = cancer_df.drop(['target'], axis = 1)
y = cancer_df['target']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state= 5)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_sc = sc.fit_transform(X_train)
X_test_sc = sc.transform(X_test)

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# XGBoost Classifier
from xgboost import XGBClassifier

xgb_classifier2 = XGBClassifier()
xgb_classifier2.fit(X_train_sc, y_train)
y_pred_xgb_sc = xgb_classifier2.predict(X_test_sc)
accuracy_score(y_test, y_pred_xgb_sc)

import pickle
pickle.dump(xgb_classifier2,open('breastcancer.pkl','wb'))