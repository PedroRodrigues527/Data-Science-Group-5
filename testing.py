from scipy.stats import shapiro, ttest_rel
from pandas import *

def paired_t_testing_apples(df: DataFrame):
    
    group1 = df[df['Quality'] == 1]
    group2 = df[df['Quality'] == 0]
    min_length = min(len(group1), len(group2))
    group1 = group1[:min_length]
    group2 = group2[:min_length]
    t_statistic, p_value = paired_t_test(group1, group2)
    print(f'T-Statistic: {t_statistic}, P-Value: {p_value}')

def paired_t_test(group1, group2):
    # Paired T-Test
    t_statistic, p_value = ttest_rel(group1, group2)
    return t_statistic, p_value

def shapiro_wilk_test(data):
    # Shapiro-Wilk Test
    for column in data.columns:
        stat, p = shapiro(data[column])
        print("\n")
        print('Statistics=%.3f, p=%.3f' % (stat, p))
        # interpret
        alpha = 0.05
        if p > alpha:
            print(column, 'Sample looks Gaussian (fail to reject H0)')
        else:
            print(column, 'Sample does not look Gaussian (reject H0)')