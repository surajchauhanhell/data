from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.datasets import fetch_california_housing

# Load dataset
data = fetch_california_housing()
X = data.data

# Standardization
scaler = StandardScaler()
X_standard = scaler.fit_transform(X)

print("Standardized Data (first 5 rows):")
print(X_standard[:5])

# Normalization
norm = MinMaxScaler()
X_norm = norm.fit_transform(X)

print("\nNormalized Data (first 5 rows):")
print(X_norm[:5])
