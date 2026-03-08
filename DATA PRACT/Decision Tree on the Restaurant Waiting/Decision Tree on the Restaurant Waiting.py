import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
import matplotlib.pyplot as plt

# Dataset
data = {
'Alt':['Yes','Yes','No','Yes','Yes','No','No','No','No'],
'Bar':['No','No','Yes','No','No','Yes','Yes','No','Yes'],
'Fri':['No','No','No','Yes','Yes','No','No','No','Yes'],
'Hun':['Yes','Yes','No','Yes','No','Yes','No','Yes','No'],
'Pat':['Some','Full','Some','Full','Full','Some','None','Some','Full'],
'Price':[1200,2500,2200,1245,4300,3400,1000,3200,3400],
'Rain':['No','No','No','No','Yes','Yes','Yes','Yes','Yes'],
'Res':['Yes','No','No','No','Yes','Yes','No','Yes','No'],
'Type':['French','Thai','Burger','Thai','French','Italian','Burger','Thai','Burger'],
'Est':['0-10','30-60','0-10','30-60','>60','0-10','0-10','0-10','>60'],
'Wait':['Yes','No','Yes','Yes','No','Yes','No','Yes','No']
}

df = pd.DataFrame(data)

# Convert categorical data to numeric
le = LabelEncoder()
for col in df.columns:
    df[col] = le.fit_transform(df[col])

# Split data
X = df.drop('Wait', axis=1)
y = df['Wait']

# Train model
model = DecisionTreeClassifier()
model.fit(X, y)

# Plot tree
plt.figure(figsize=(12,6))
tree.plot_tree(model, feature_names=X.columns, filled=True)
plt.show()

# Prediction example
print(model.predict([X.iloc[0]]))
