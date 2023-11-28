import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

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


st.title('Credit Scoring')
st.write("This app uses 6 inputs to predict the Variety of Iris using "
         "a model built on the Palmer's Iris's dataset. Use the form below"
         " to get started!")

iris_file = st.file_uploader('Upload your own Iris data')
# df1=pd.read_csv('1-31Mar-30Sep2022 copy.csv')
if iris_file is None:
    rf_pickle = open('random_forest_iris.pickle', 'rb')
    map_pickle = open('output_iris.pickle', 'rb')

    rf = pickle.load(rf_pickle)
    unique_penguin_mapping = pickle.load(map_pickle)
else:

    df = pd.read_csv(iris_file)
    df1=df.copy()
    le = LabelEncoder()
    cat = df.select_dtypes(include='object').columns.tolist()
    number = df.select_dtypes(exclude="object").columns
    ob = df[['UNIVERSITY NAME', 'FACULTY NAME', 'LEVEL TYPE', 'YEARS',
             'BRANCH TYPE', 'CATEGORY NAME', 'SUB SERIES NAME',
             'COLOR', 'PROD SUM PRICE', 'INSTALL NUM']].columns
    vif_df = df[number]
    vif_data = pd.DataFrame()
    sa = vif_df.dropna()
    for i in cat:
        le.fit(df[i])
        df[i] = le.transform(df[i])

    df_1 = df.dropna(axis='columns')
    x = df_1.drop(['STATUS', 'UNIVERSITY NAME', 'FACULTY NAME', 'LEVEL TYPE', 'BRANCH TYPE',
                'YEARS','SERIES NAME', 'SUB SERIES NAME', 'COLOR', 'PROD TOTAL AMT'], axis=1)
    y = df['STATUS']
    yy=df1['STATUS']
    output, unique_penguin_mapping = pd.factorize(yy)
    smt = SMOTE()

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    X_train_sm, y_train_sm = smt.fit_resample(X_train, y_train)
    ada = ADASYN(random_state=130)
    smtom = SMOTETomek(random_state=139)
    smenn = SMOTEENN()
    X_train_ada, y_train_ada = smtom.fit_resample(X_train, y_train)
    # print(y_train_sm)
    # print(y_test)
    # lr = LogisticRegression()
    # lr.fit(X_train_ada, y_train_ada)
    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(X_train_sm, y_train_sm)
    y_pred=rf.predict(X_test)
    score = round(accuracy_score(y_pred, y_test), 2)
    textscore = '<p style="font-family:Courier; color:Black; font-size: 16px;">We trained a Random Forest model on these data ,it has a Accuracy of {}! Use the inputs below to try out the model.</p>'
    st.write(textscore.format(score), unsafe_allow_html=True)

with st.form('user_inputs'):
    # year = st.number_input(
    #     'ชั้นปี', min_value=0.0, max_value=12.0, value=10.0)
    CHOICES = {1: "Smart Phone", 2: "Tablet", 3: "Laptop"}
    def format_func(option):
        return CHOICES[option]
    category = st.selectbox(" อุปกรณ์ (Device)", options=list(CHOICES.keys()), format_func=format_func)
    # st.write(f"You selected option {category} called {format_func(category)}")
    # category = st.number_input(
    #     'Category Name', min_value=0.0, max_value=12.0, value=10.0)
    prod_sum_price = st.number_input(
        'ยอดสินค้าทั้งหมด', min_value=0.0, max_value=1200000.0, value=10.0
    )
    install_num = st.number_input(
        'ระยะเวลาผ่อน', min_value=0.0, max_value=36.0, value=10.0)
    install_sum_final = st.number_input(
        'ผ่อนต่อเดือน', min_value=0.0, max_value=1200000.0, value=10.0)
    hp_vat_sum = st.number_input(
        'ยอดรวมหลังผ่อนเสร็จ', min_value=0.0, max_value=1200000.0, value=10.0)


    st.form_submit_button()

new_prediction =rf.predict([[category, prod_sum_price,install_num, install_sum_final,  hp_vat_sum
                                   ]])
prediction_species = unique_penguin_mapping[new_prediction][0]
if prediction_species=='Wait Welcome Call':
    s='ลูกหนี้ปกติ'
    textpredict = '<p style="font-family:Courier; color:Black; font-size: 20px;">We predict your Customer is of the {} </p>'
    st.markdown(textpredict.format(s), unsafe_allow_html=True)
else:
    f='ลูกหนี้เสีย'
    textpredict = '<p style="font-family:Courier; color:Black; font-size: 20px;">We predict your Customer is of the {} </p>'
    st.markdown(textpredict.format(f), unsafe_allow_html=True)
choices = ['INSTALL NUM',
               'INSTALL SUM FINAL',
               'HP VAT SUM',
               'YEARS']

selected_x_var = st.selectbox('เลือก แกน x', (choices))
selected_y_var = st.selectbox('เลือก แกน y', (choices))

st.subheader('ข้อมูลตัวอย่าง')
st.write(iris_file)

sns.set_style('darkgrid')
markers = {'Wait Welcome Call': "v", 'Overdue 1': "s"}
markers1 = {'Smart Phone': "o", 'Tablet': "s",'Laptop':"v"}
fig, ax = plt.subplots()
ax = sns.scatterplot(data=iris_file,
                         x=selected_x_var, y=selected_y_var,
                         hue='STATUS', markers=markers, style='STATUS')
plt.xlabel(selected_x_var)
plt.ylabel(selected_y_var)
plt.title("Status")
st.pyplot(fig)

fig, ax = plt.subplots()
ax = sns.scatterplot(data=iris_file,
                         x=selected_x_var, y=selected_y_var,
                         hue='CATEGORY NAME', markers=markers1, style='CATEGORY NAME')
plt.xlabel(selected_x_var)
plt.ylabel(selected_y_var)
plt.title("Device")
st.pyplot(fig)