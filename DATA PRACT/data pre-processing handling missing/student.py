import pandas as pd
import numpy as np

# Read CSV
df = pd.read_csv("student.csv")

print("Original Data")
print(df)

# Handle Missing Values (fixed)
df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Marks'] = df['Marks'].fillna(df['Marks'].mean())

print("\nAfter Handling Missing Values")
print(df)

# Detect Outliers using IQR
Q1 = df['Marks'].quantile(0.25)
Q3 = df['Marks'].quantile(0.75)
IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

outliers = df[(df['Marks'] < lower) | (df['Marks'] > upper)]

print("\nOutliers in Marks column")
print(outliers)
