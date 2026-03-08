import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# Create dataset
data = {
'Height':[151,174,138,186,128,136,179,163,152],
'Weight':[63,81,56,91,47,57,76,72,62]
}

df = pd.DataFrame(data)

print("Dataset:")
print(df)

# Create classification column
df['Category'] = df['Weight'].apply(lambda x: 1 if x >= 70 else 0)

print("\nDataset with Category:")
print(df)

# Feature and target
X = df[['Height']]
y = df['Category']

# Model
model = DecisionTreeClassifier()
model.fit(X,y)

# Prediction (fixed)
test = pd.DataFrame({'Height':[170]})
prediction = model.predict(test)

print("\nPrediction for height 170:")

if prediction[0] == 1:
    print("Heavy Person")
else:
    print("Not Heavy Person")
