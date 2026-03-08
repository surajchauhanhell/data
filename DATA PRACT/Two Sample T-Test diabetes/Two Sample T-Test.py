from scipy import stats

# Time taken by two groups
group1 = [85,95,100,80,90,97,104,95,88,92,94,99]
group2 = [83,85,96,92,100,104,94,95,88,90,93,94]

# Two Sample T-Test
t_stat, p_value = stats.ttest_ind(group1, group2)

print("T-statistic:", t_stat)
print("P-value:", p_value)

alpha = 0.05

if p_value < alpha:
    print("Reject H0: Significant difference between groups.")
else:
    print("Fail to Reject H0: No significant difference between groups.")
