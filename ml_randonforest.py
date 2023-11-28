import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from sklearn.metrics import accuracy_score
import numpy as np
from numpy import log, log1p
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.linear_model import LogisticRegression
import scipy.stats as stats
from scipy.stats import shapiro,boxcox,yeojohnson
from scipy.stats import boxcox
from sklearn.metrics import mean_squared_error , mean_absolute_error
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.special import logit

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE,ADASYN
from sklearn.feature_selection import mutual_info_regression
from imblearn.combine import SMOTEENN,SMOTETomek

from sklearn.ensemble import RandomForestClassifier
import pickle
df2=pd.read_csv('//Users/comseven/PycharmProjects/pythonProject1/pythonProject/pythonProject/1-31Mar-30Sep2022.csv')
df=pd.read_csv('//Users/comseven/PycharmProjects/pythonProject1/pythonProject/pythonProject/1-31Mar-30Sep2022.csv')
le = LabelEncoder()
print(df.columns)

cat = df.select_dtypes(include = 'object').columns.tolist()
print(cat)
number=df.select_dtypes(exclude="object").columns
ob=df[[ 'UNIVERSITY NAME', 'FACULTY NAME', 'LEVEL TYPE', 'YEARS',
       'BRANCH TYPE', 'CATEGORY NAME', 'SUB SERIES NAME',
       'COLOR', 'PROD SUM PRICE', 'INSTALL NUM']].columns

for i in cat:
  le.fit(df[i])
  df[i]=le.transform(df[i])

df_1=df.dropna(axis='columns')
print(df_1)


from sklearn import tree


x=df_1.drop(['STATUS','UNIVERSITY NAME', 'FACULTY NAME', 'LEVEL TYPE', 'BRANCH TYPE',
        'YEARS','SERIES NAME', 'SUB SERIES NAME', 'COLOR','PROD TOTAL AMT'],axis=1)
y=df2['STATUS']
print(x.columns)
y, uniques = pd.factorize(y)
print(y)
print(uniques)
smt=SMOTE()

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
X_train_sm,y_train_sm=smt.fit_resample(X_train,y_train)
ada=ADASYN(random_state=130)
smtom=SMOTETomek(random_state=139)
smenn=SMOTEENN()
X_train_ada,y_train_ada=smtom.fit_resample(X_train,y_train)
# print(y_train_sm)
# print(y_test)
lr = LogisticRegression()
lr.fit(X_train_ada, y_train_ada)
rf = RandomForestClassifier()
rf.fit(X_train_sm,  y_train_sm)


Y_pred = rf.predict(X_test)
print(classification_report(y_test,Y_pred))
score = accuracy_score(Y_pred, y_test)
print('Our accuracy score for this model is {}'.format(score))


rf_pickle = open('random_forest_iris.pickle', 'wb')
pickle.dump(rf, rf_pickle)
rf_pickle.close()
output_pickle = open('output_iris.pickle', 'wb')
pickle.dump(uniques, output_pickle)
output_pickle.close()



fig, ax = plt.subplots()

ax = sns.barplot(x=rf.feature_importances_, y=x.columns)
plt.title('Which features are the most important for species prediction?')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
fig.savefig('feature_importance.png')
