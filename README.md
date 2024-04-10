import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import RandomOverSampler
from sklearn.decomposition import PCA
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

import warnings
warnings.filterwarnings(action='ignore')
data = pd.read_csv('/content/drive/MyDrive/fetal_health.csv')
data.head()
data.info()
data.isna().sum().sum()
# Null count analysis
null_plot = msno.bar(data, color = "#5F9EA0")
eda_df = data.copy()
plt.figure(figsize=(25, 15))

for i, column in enumerate(eda_df.columns):
    plt.subplot(4, 6, i + 1)
    sns.histplot(data=eda_df[column])
    plt.title(column)

plt.tight_layout()
plt.show()
plt.figure(figsize=(25, 15))

for i, column in enumerate(eda_df.columns):
    plt.subplot(4, 6, i + 1)
    sns.boxplot(data=eda_df[column])
    plt.title(column)

plt.tight_layout()
plt.show()
corr = eda_df.corr()

plt.figure(figsize=(24, 20))
sns.heatmap(corr, annot=True, vmin=-1.0, cmap='mako')
plt.title("Correlation Matrix")
plt.show()
plt.figure(figsize=(10, 10))

plt.pie(
    eda_df['fetal_health'].value_counts(),
    autopct='%.2f%%',
    labels=["NORMAL", "SUSPECT", "PATHOLOGICAL"],
    colors=sns.color_palette('Blues')
)

plt.title("Class Distribution")
plt.show()
# Split df into X and y
df = data.copy()
y = df['fetal_health']
X = df.drop('fetal_health', axis=1)
# OverSampling
oversample = RandomOverSampler(sampling_strategy='not majority')
X_over, y_over = oversample.fit_resample(X, y)
# Scale X
scaler=StandardScaler()
X_scaled=scaler.fit_transform(X_over)
X_scaled=pd.DataFrame(X_scaled,columns=X_over.columns)
X_scaled.shape
# PCA
pca=PCA(n_components=0.95) #0.95 here refers that the total variance explained by the components must be atleast 95%
X_pca_final=pca.fit_transform(X_scaled)
X_pca_final=pd.DataFrame(X_pca_final)
X_pca_final.shape
X_train, X_test, y_train, y_test = train_test_split(X_pca_final, y_over, shuffle=True, random_state=42, stratify = y_over)
# Cross Validation
cv_method = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

knn=KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)
confusion_knn=confusion_matrix(y_test,y_pred)
sns.heatmap(confusion_knn,annot=True)
print(classification_report(y_test,knn.predict(X_test)))
cm = confusion_matrix(y_test, y_pred, labels=knn.classes_)
sns.heatmap(cm/np.sum(cm), annot=True, fmt='.2%', cmap='Blues')
knn_scores=[]
for k in range(1,20):
    knn1=KNeighborsClassifier(n_neighbors=k)
    scores=cross_val_score(knn1,X_train,y_train,cv=5)
    knn_scores.append(scores.mean())

x_ticks = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
x_labels = x_ticks

plt.plot([k for k in range(1,20)],knn_scores)
plt.xticks(ticks=x_ticks, labels=x_labels)
plt.grid()
SVC().get_params()
svm=SVC(random_state=42,gamma = 100, C = 100)
svm.fit(X_train,y_train)
y_pred_svm=svm.predict(X_test)
confusion_svm=confusion_matrix(y_test,svm.predict(X_test))
sns.heatmap(confusion_svm,annot=True)
print(classification_report(y_test,y_pred_svm))
sns.heatmap(confusion_svm/np.sum(confusion_svm), annot=True, fmt='.2%', cmap='Blues')
DecisionTreeClassifier().get_params()
dt=DecisionTreeClassifier(random_state=42,min_samples_leaf = 1, min_samples_split= 2, max_depth= 19)
dt.fit(X_train,y_train)
y_pred_dt=dt.predict(X_test)
confusion_dt=confusion_matrix(y_test,dt.predict(X_test))
sns.heatmap(confusion_dt,annot=True)
print(classification_report(y_test,y_pred_dt))
rf=RandomForestClassifier(random_state=42,min_samples_leaf = 1, min_samples_split= 2, max_depth= 19)
rf.fit(X_train,y_train)
y_pred_rf=rf.predict(X_test)
confusion_rf=confusion_matrix(y_test,rf.predict(X_test))
sns.heatmap(confusion_rf,annot=True)
print(classification_report(y_test,y_pred_rf))
GradientBoostingClassifier().get_params()
gb=GradientBoostingClassifier(random_state=42, learning_rate = 1.0, n_estimators = 50, subsample = 0.8, min_samples_leaf = 1, min_samples_split= 2, max_depth= 19)
gb.fit(X_train,y_train)
y_pred_gb=gb.predict(X_test)
confusion_gb=confusion_matrix(y_test,gb.predict(X_test))
sns.heatmap(confusion_gb,annot=True)
print(classification_report(y_test,y_pred_dt))
XGBClassifier().get_params()
y_train_shifted = y_train - 1
xgb.fit(X_train, y_train_shifted)
print(classification_report(y_test,y_pred_dt))
AdaBoostClassifier().get_params()
ada=AdaBoostClassifier(random_state=42, base_estimator = DecisionTreeClassifier(), learning_rate = 1.0, n_estimators=200)
ada.fit(X_train,y_train)
y_pred_ada=ada.predict(X_test)
confusion_ada=confusion_matrix(y_test,ada.predict(X_test))
sns.heatmap(confusion_ada,annot=True)
print(classification_report(y_test,y_pred_dt))
SGDClassifier().get_params()
sgd=SGDClassifier(random_state=42, alpha=0.0001,
                  eta0 = 1, learning_rate='invscaling',
                 loss = 'hinge', penalty = 'elasticnet')
sgd.fit(X_train,y_train)
y_pred_sgd=sgd.predict(X_test)
confusion_sgd=confusion_matrix(y_test,sgd.predict(X_test))
sns.heatmap(confusion_sgd,annot=True)
print(classification_report(y_test,y_pred_sgd))
LGBMClassifier().get_params()
lgbm=LGBMClassifier(random_state=42, learning_rate = 1.0)
lgbm.fit(X_train,y_train)
y_pred_lgbm=lgbm.predict(X_test)
confusion_lgbm=confusion_matrix(y_test,lgbm.predict(X_test))
sns.heatmap(confusion_lgbm,annot=True)
print(classification_report(y_test,y_pred_lgbm))
print(f"Accuracy of KNN model  -> {round(knn.score(X_test, y_test),4)*100}%")
print(f"Accuracy of SVM model  -> {round(svm.score(X_test, y_test),3)*100}%")
print(f"Accuracy of RF model   -> {round(rf.score(X_test, y_test),4)*100}%")
print(f"Accuracy of DT model   -> {round(dt.score(X_test, y_test),4)*100}%")
print(f"Accuracy of GB model   -> {round(gb.score(X_test, y_test),3)*100}%")
print(f"Accuracy of XGB model  -> {round(xgb.score(X_test, y_test),3)*100}%")
print(f"Accuracy of Ada model  -> {round(ada.score(X_test, y_test),3)*100}%")
print(f"Accuracy of SGD model  -> {round(sgd.score(X_test, y_test),4)*100}%")
print(f"Accuracy of LGBM model -> {round(lgbm.score(X_test, y_test),3)*100}%")
