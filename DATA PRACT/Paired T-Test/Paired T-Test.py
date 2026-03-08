from scipy import stats

# Test scores
test1 = [85,68,67,84,98,60,94,80,94,98,95,80,85,87,75]
test2 = [70,90,80,89,88,86,78,87,90,86,92,94,99,93,86]

# Paired T-Test
t_stat, p_value = stats.ttest_rel(test1, test2)

print("T-statistic:", t_stat)
print("P-value:", p_value)

# Decision rule
alpha = 0.05

if p_value < alpha:
    print("Reject H0: There is significant difference in scores.")
else:
    print("Fail to Reject H0: No significant difference in scores.")
