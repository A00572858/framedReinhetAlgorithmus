import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import sklearn.metrics as sm
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

df_raw = pd.read_csv("beers.csv")
df_clean = df_raw[['abv','ibu','style']].dropna().reset_index().copy()

df_x = df_clean[['abv','ibu']]
df_y = df_clean[['style']]

df_y.loc[df_y['style'].str.contains('IPA'), 'style'] = "IPA"

df_y_t = df_y.copy()

for i in range(len(df_y_t)):
    if df_y_t['style'][i] == "IPA":
        df_y_t['style'][i] = 1
    else:
        df_y_t['style'][i] = 0

df_x['class'] = df_y_t['style']

df_train, df_test = train_test_split(df_x, test_size = 0.2)

df_train_shuffled = df_train.sample(frac=1).reset_index()
df_test_shuffled = df_test.sample(frac=1).reset_index()

x_train = df_train_shuffled[['ibu','abv']]
y_train = df_train_shuffled[['class']]

x_test = df_test_shuffled[['ibu','abv']]
y_test = df_test_shuffled[['class']]

scaler = StandardScaler()
scaled_features = scaler.fit_transform(x_train)

y_train['class']=y_train['class'].astype('int')
y_test['class']=y_test['class'].astype('int')

model = LogisticRegression().fit(x_train, y_train)

prediction = model.predict(x_test)

print(model.score(x_test, y_test))