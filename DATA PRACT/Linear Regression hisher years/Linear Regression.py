import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Dataset
experience = np.array([2,10,4,20,8,12,22]).reshape(-1,1)
salary = np.array([30000,95000,45000,178000,84000,120000,200000])

# Create and train model
model = LinearRegression()
model.fit(experience, salary)

# Prediction
pred = model.predict([[15]])
print("Predicted Salary for 15 years experience:", pred)

# Model parameters
print("Slope:", model.coef_)
print("Intercept:", model.intercept_)

# Plot graph
plt.scatter(experience, salary)
plt.plot(experience, model.predict(experience))
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Linear Regression: Experience vs Salary")
plt.show()
