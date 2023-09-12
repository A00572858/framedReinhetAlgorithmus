import pandas as pd

import sklearn.metrics as sm
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

'''
    Rodrigo Muñoz Guerrero (A00572858)
    Created: 11/09/2023
    Last edited: 11/09/2023

    Title: Framed Reinheit Algorithmus

    Context:
        This file contains the generation of a classification prediction model made with
        sklearn library, with support of pandas library.

        The model is trained with real data taken from https://www.kaggle.com/datasets/nickhould/craft-cans
        which was adapted to fit the algorithm and the purpose of it.
'''

#   Load data from csv file in repository into a pandas data frame.
df_raw = pd.read_csv("beers.csv")

#   The data cleaning starts taking only in consideration the abv (alcohol by volume),
#   ibu (international bitterness unit) and its style since these are the only relevant
#   variables in the dataset for the algorithm.
#   We also drop the data that does not
#   have this information because it is crucial for the model to have them all.
df_clean = df_raw[['abv','ibu','style']].dropna().reset_index().copy()

df_x = df_clean[['abv','ibu']]
df_y = df_clean[['style']]

#   We then transform all the registers that contain the name "IPA" in its style, we do this to
#   include every variation of the style (like session IPA, imperial IPA, etc.)
df_y.loc[df_y['style'].str.contains('IPA'), 'style'] = "IPA"
df_y_t = df_y.copy()

#   The same data is transformed into two possible classes, IPA or not IPA. We do this with
#   integer values so it is easier to handle and we assign 0 to non-IPAs and 1 to its counterpart.
for i in range(len(df_y_t)):
    if df_y_t['style'][i] == "IPA":
        df_y_t['style'][i] = 1
    else:
        df_y_t['style'][i] = 0

df_x['class'] = df_y_t['style']

#   The sklearn library is used again to split our data into two, the train and the test data.
#   It is separated into 80%/20% for train and test respectively. We do this to have a way of
#   testing our model with data we already know is legit.
df_train, df_test = train_test_split(df_x, test_size = 0.2)

#   And we shuffle both the dataframes so we do not get an unpredicted bias on the model.
df_train_shuffled = df_train.sample(frac=1).reset_index()
df_test_shuffled = df_test.sample(frac=1).reset_index()

x_train = df_train_shuffled[['ibu','abv']]
y_train = df_train_shuffled[['class']]

x_test = df_test_shuffled[['ibu','abv']]
y_test = df_test_shuffled[['class']]

#   As we have two not-so-equal variables in value range, we use sklearn library to scale
#   the data so it is easier for the model to learn and not bias itself.
scaler = StandardScaler()
scaled_features = scaler.fit_transform(x_train)

y_train['class']=y_train['class'].astype('int')
y_test['class']=y_test['class'].astype('int')

#   Here we create and train our model with the sklearn library
model = LogisticRegression().fit(x_train, y_train)

#   If wanted to, the lines below can be uncommented to show the prediction array of the test data
# prediction = model.predict(x_test)
# print(prediction)

#   And finally, the accuracy or score is printed in console (aproximately 83% accurate)
print(model.score(x_test, y_test))