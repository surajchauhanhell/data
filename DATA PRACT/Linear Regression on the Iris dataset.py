import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_iris

# Load Iris dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# Select features
X = df[['petal length (cm)']]
y = df['petal width (cm)']

# Train model
model = LinearRegression()
model.fit(X, y)

# Prediction (fixed warning)
pred = model.predict(pd.DataFrame([[5]], columns=['petal length (cm)']))
print("Predicted petal width for petal length 5 cm:", pred)

# Model parameters
print("Slope:", model.coef_)
print("Intercept:", model.intercept_)

# Plot graph
plt.scatter(X, y)
plt.plot(X, model.predict(X))
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.title("Linear Regression on Iris Dataset")
plt.show()
