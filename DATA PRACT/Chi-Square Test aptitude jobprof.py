import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

# Data
aptitude = [85,65,50,68,87,74,65,96,68,94,73,84,85,87,91]
jobprof = [70,90,80,89,88,86,78,67,86,90,92,94,99,93,87]

# Create DataFrame
data = pd.DataFrame({
    'Aptitude': aptitude,
    'JobProficiency': jobprof
})

# Convert scores into categories (Low, Medium, High)
data['Aptitude_cat'] = pd.cut(data['Aptitude'], bins=3, labels=["Low","Medium","High"])
data['Job_cat'] = pd.cut(data['JobProficiency'], bins=3, labels=["Low","Medium","High"])

# Contingency Table
table = pd.crosstab(data['Aptitude_cat'], data['Job_cat'])

# Chi-Square Test
chi2, p, dof, expected = chi2_contingency(table)

print("Contingency Table:\n", table)
print("\nChi-square value:", chi2)
print("P-value:", p)

alpha = 0.05

if p < alpha:
    print("\nReject H0: There is a relationship between aptitude and job proficiency.")
else:
    print("\nFail to Reject H0: No significant relationship between aptitude and job proficiency.")
