import pandas as pd
import matplotlib as plt

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

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

'''
    Load data from csv file in repository into a pandas data frame.
'''
df_raw = pd.read_csv("beers.csv")

'''
    The data cleaning starts taking only in consideration the abv (alcohol by volume),
    ibu (international bitterness unit) and its style since these are the only relevant
    variables in the dataset for the algorithm.
    We also drop the data that does not
    have this information because it is crucial for the model to have them all.
'''
df_clean = df_raw[['abv','ibu','style']].dropna().reset_index().copy()

df_x = df_clean[['abv','ibu']]
df_y = df_clean[['style']]

'''
    We then transform all the registers that contain the name "IPA" in its style, we do this to
    include every variation of the style (like session IPA, imperial IPA, etc.)
'''
df_y.loc[df_y['style'].str.contains('IPA'), 'style'] = "IPA"
df_y_t = df_y.copy()

'''
    The same data is transformed into two possible classes, IPA or not IPA. We do this with
    integer values so it is easier to handle and we assign 0 to non-IPAs and 1 to its counterpart.
'''
for i in range(len(df_y_t)):
    if df_y_t['style'][i] == "IPA":
        df_y_t['style'][i] = 1
    else:
        df_y_t['style'][i] = 0

df_x['class'] = df_y_t['style']

'''
    The sklearn library is used again to split our data into two, the train and the test data.
    It is separated into 80%/20% for train and test respectively. We do this to have a way of
    testing our model with data we already know is legit.
'''
df_train, df_test = train_test_split(df_x, test_size = 0.2)

'''
    And we shuffle both the dataframes so we do not get an unpredicted bias on the model.
'''
df_train_shuffled = df_train.sample(frac=1).reset_index()
df_test_shuffled = df_test.sample(frac=1).reset_index()

x_train = df_train_shuffled[['ibu','abv']]
y_train = df_train_shuffled[['class']]

x_test = df_test_shuffled[['ibu','abv']]
y_test = df_test_shuffled[['class']]

'''
    As we have two not-so-equal variables in value range, we use sklearn library to scale
    the data so it is easier for the model to learn and not bias itself.
'''
scaler = StandardScaler()
scaled_features = scaler.fit_transform(x_train)

y_train['class']=y_train['class'].astype('int')
y_test['class']=y_test['class'].astype('int')

'''
    Here we create and train our model with the sklearn library
'''
model = LogisticRegression().fit(x_train, y_train)

'''
    And we predict the data for our train data to check on the performance
'''
train_prediction = model.predict(x_train)

'''
    If you wish to see the grphics of the difference between the training prediction and the real data,
    uncomment the code below and run
'''
# fig, (plot1, plot2) = plt.subplots(1,2, figsize=(10,5))
# plot1.scatter(x_train['ibu'], x_train['abv'], c=y_train['class'])
# plot1.set_title("Real")
# plot2.scatter(x_train['ibu'], x_train['abv'], c=train_prediction)
# plot2.set_title("Prediction")

'''
    We calculate the confusion matrix with help of sklearn and assign the data to the auxiliar variables
'''
train_conma = confusion_matrix(y_train, train_prediction)
TTP = train_conma[1, 1]     # Train True Positives
TTN = train_conma[0, 0]     # Train True Negatives
TFP = train_conma[0, 1]     # Train False Positives
TFN = train_conma[1, 0]     # Train False Negatives

'''
    And we calculate and display the accuracy and F1 score of the training run
'''
train_accuracy = ( TTP + TTN ) / ( TTP + TTN + TFP + TFN )
train_f1 = ( 2 * TTP ) / ( 2 * TTP + TFP + TFN )

print("Train accuaracy is: " + str(train_accuracy * 100) + "%")
print("Train F1 score is: " + str(train_f1 * 100) + "%")
print("=========================================")

'''
    Finally, we predict the test data
'''
prediction = model.predict(x_test)

'''
    If you wish to see the grphics of the difference between the test prediction and the real data,
    uncomment the code below and run
'''
# fig, (plot1, plot2) = plt.subplots(1,2, figsize=(10,5))
# plot1.scatter(x_test['ibu'], x_test['abv'], c=y_test['class'])
# plot1.set_title("Real")
# plot2.scatter(x_test['ibu'], x_test['abv'], c=prediction)
# plot2.set_title("Prediction")

'''
    We calculate the confusion matrix again and assign the data to the auxiliar variables
'''
conma = confusion_matrix(y_test, prediction)
TP = train_conma[1, 1]     # Train True Positives
TN = train_conma[0, 0]     # Train True Negatives
FP = train_conma[0, 1]     # Train False Positives
FN = train_conma[1, 0]     # Train False Negatives

'''
    And we finally, calculate and display the accuracy and F1 score of the test run
'''
accuracy = ( TP + TN ) / ( TP + TN + FP + FN )
f1 = ( 2 * TP ) / ( 2 * TP + FP + FN )
print("Test accuaracy is: " + str(accuracy * 100) + "%")
print("Test F1 score is: " + str(f1 * 100) + "%")
print("=========================================")