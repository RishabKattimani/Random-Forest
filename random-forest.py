#-------------------------------------------------------------------------------
# Imports

import pandas as pd
from matplotlib import pyplot as plt
import numpy
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

#-------------------------------------------------------------------------------
# Setting Up the Data

df = pd.read_csv("age-data.csv")

y = df["Group"].values
y = y.astype('int')

x = df.drop(labels=["Group"], axis=1)

#-------------------------------------------------------------------------------
# Predicting

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=20)

model = RandomForestClassifier(n_estimators = 10, random_state = 30)
model.fit(x_train, y_train)

#-------------------------------------------------------------------------------
# Testing

age_group = ["Baby", "Kid", "Adult", "Senior"]


predict = model.predict([[39]])
print(age_group[predict[0]-1])













#-------------------------------------------------------------------------------
