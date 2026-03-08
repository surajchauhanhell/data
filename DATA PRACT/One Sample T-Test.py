from scipy import stats

# Sample data
scores = [72, 88, 64, 74, 67, 79, 85, 75, 89, 77]

# Hypothesized population mean
hypothesized_mean = 70

# Perform One Sample T-Test
t_stat, p_value = stats.ttest_1samp(scores, hypothesized_mean)

# Display results
print("T-statistic:", t_stat)
print("P-value:", p_value)

# Interpretation
alpha = 0.05

if p_value < alpha:
    print("Reject H0: The sample mean is significantly different from 70.")
else:
    print("Fail to Reject H0: The sample mean is not significantly different from 70.")
