from scipy import stats

# Scores of three classes
classA = [85, 90, 88, 82, 87]
classB = [76, 78, 80, 81, 75]
classC = [92, 88, 94, 89, 90]

# Perform One Way ANOVA
f_stat, p_value = stats.f_oneway(classA, classB, classC)

print("F-statistic:", f_stat)
print("P-value:", p_value)

alpha = 0.05

if p_value < alpha:
    print("Reject H0: Significant difference between class means.")
else:
    print("Fail to Reject H0: No significant difference between class means.")
