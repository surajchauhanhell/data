import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Read dataset
df = pd.read_csv("data.csv")

print("Original Dataset:\n", df)

# Select numerical features
X = df[["Age","Salary"]]

# Standardization
scaler = StandardScaler()
X_standard = scaler.fit_transform(X)

print("\nStandardized Data:")
print(X_standard)

# Normalization
norm = MinMaxScaler()
X_normalized = norm.fit_transform(X)

print("\nNormalized Data:")
print(X_normalized)
