import pandas as pd

# Step 1: Read CSV file
df = pd.read_csv("employee.csv")

print("Original Dataset:\n")
print(df)

# Step 2: Handle Missing Values
df["Age"] = df["Age"].fillna(df["Age"].mean())
df["Salary"] = df["Salary"].fillna(df["Salary"].mean())

print("\nAfter Handling Missing Values:\n")
print(df)

# Step 3: Detect Outliers using IQR
Q1 = df["Salary"].quantile(0.25)
Q3 = df["Salary"].quantile(0.75)

IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

# Step 4: Remove Outliers
df_no_outliers = df[(df["Salary"] >= lower) & (df["Salary"] <= upper)]

print("\nDataset After Removing Outliers:\n")
print(df_no_outliers)
