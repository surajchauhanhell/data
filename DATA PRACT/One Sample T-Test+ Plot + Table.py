import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Sample data
scores = [72, 88, 64, 74, 67, 79, 85, 75, 89, 77]

# Hypothesized mean
hypothesized_mean = 70

# One Sample T-Test
t_stat, p_value = stats.ttest_1samp(scores, hypothesized_mean)

# Sample mean
sample_mean = np.mean(scores)

# Plot histogram
plt.figure(figsize=(8,6))
plt.hist(scores, bins=6, alpha=0.7)

# Mean lines
plt.axvline(sample_mean, linestyle='--', label="Sample Mean")
plt.axvline(hypothesized_mean, linestyle='--', label="Hypothesized Mean")

plt.title("One Sample T-Test Visualization")
plt.xlabel("Scores")
plt.ylabel("Frequency")

# Table data
table_data = [
    ["Sample Mean", round(sample_mean,2)],
    ["Hypothesized Mean", hypothesized_mean],
    ["T-statistic", round(t_stat,4)],
    ["P-value", round(p_value,4)]
]

# Add table
table = plt.table(
    cellText=table_data,
    colLabels=["Statistic", "Value"],
    loc="bottom",
    cellLoc="center"
)

table.scale(1,1.5)

plt.legend()
plt.tight_layout()
plt.show()

# Interpretation
alpha = 0.05
if p_value < alpha:
    print("Reject H0: Sample mean is significantly different from 70")
else:
    print("Fail to Reject H0: Sample mean is not significantly different from 70")
