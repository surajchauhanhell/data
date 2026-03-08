import pandas as pd
from sklearn.linear_model import LinearRegression

# Create dataset
data = {
'Height':[151,174,138,186,128,136,179,163,152],
'Weight':[63,81,56,91,47,57,76,72,62]
}

df = pd.DataFrame(data)

print("Dataset:")
print(df)

# Independent variable
X = df[['Height']]

# Dependent variable
y = df['Weight']

# Create model
model = LinearRegression()

# Train model
model.fit(X,y)

# Predict weight
pred = model.predict(X)

print("\nPredicted Weights:")
print(pred)
