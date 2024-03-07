import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, f_oneway, ttest_rel, wilcoxon, kruskal, friedmanchisquare, probplot, shapiro
from sklearn.datasets import load_iris

import matplotlib.pyplot as plt

import statsmodels.api as sm
import statsmodels.stats.api as sms
from statsmodels.formula.api import ols
class HypothesisTester:
    """
    The t-test assumes that the data is normally distributed and that the variances are equal between groups (for
    unpaired t-test) or within groups (for paired t-test).
    The ANOVA test assumes that the data is normally distributed and that the variances are equal between groups.
    """
    def unpaired_t_test(self, group1, group2):
        """
        Perform unpaired t-test for two groups.

        Parameters:
        - group1: List or array-like object containing data for group 1.
        - group2: List or array-like object containing data for group 2.

        Returns:
        - t_statistic: The calculated t-statistic.
        - p_value: The p-value associated with the t-statistic.
        """
        t_statistic, p_value = ttest_ind(group1, group2)
        return t_statistic, p_value

    def unpaired_anova(self, *groups):
        """
        Perform unpaired ANOVA for more than two groups.

        Parameters:
        - *groups: Variable length argument containing data for each group. Each argument should be a list or array-like
        object.

        Returns:
        - f_statistic: The calculated F-statistic.
        - p_value: The p-value associated with the F-statistic.
        """
        f_statistic, p_value = f_oneway(*groups)
        return f_statistic, p_value

    def paired_t_test(self, group1, group2):
        """
        Perform paired t-test for two groups.

        Parameters:
        - group1: List or array-like object containing data for group 1.
        - group2: List or array-like object containing data for group 2.
                  Should have the same length as group1.

        Returns:
        - t_statistic: The calculated t-statistic.
        - p_value: The p-value associated with the t-statistic.
        """
        t_statistic, p_value = ttest_rel(group1, group2)
        return t_statistic, p_value

    def paired_anova(self, data):
        """
        Perform paired ANOVA (repeated measures ANOVA) for more than two groups.

        Parameters:
        - data: Pandas DataFrame containing the data with columns representing different conditions.

        Returns:
        - f_statistic: The calculated F-statistic.
        - p_value: The p-value associated with the F-statistic.
        """
        model = ols('value ~ C(condition)', data=data).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        return anova_table['F'][0], anova_table['PR(>F)'][0]

    def wilcoxon_ranksum_test(self, group1, group2):
        """
        Perform Wilcoxon rank-sum test (Mann-Whitney U test) for two independent samples.

        Parameters:
        - group1: List or array-like object containing data for sample 1.
        - group2: List or array-like object containing data for sample 2.

        Returns:
        - statistic: The calculated test statistic.
        - p_value: The p-value associated with the test statistic.
        """
        statistic, p_value = sms.stattools.stats.mannwhitneyu(group1, group2)

        return statistic, p_value

    def wilcoxon_signedrank_test(self, group1, group2):
        """
        Perform Wilcoxon signed-rank test for paired samples.
        Defines the alternative hypothesis with ‘greater’ option, this the distribution underlying d is stochastically
        greater than a distribution symmetric about zero; d represent the difference between the paired samples:
        d = x - y if both x and y are provided, or d = x otherwise.

        Parameters:
        - group1: List or array-like object containing data for sample 1.
        - group2: List or array-like object containing data for sample 2.
                  Should have the same length as group1.

        Returns:
        - statistic: The calculated test statistic.
        - p_value: The p-value associated with the test statistic.
        """
        statistic, p_value = wilcoxon(group1, group2, alternative="greater")
        return statistic, p_value

    def kruskal_wallis_test(self, *groups):
        """
        Perform Kruskal-Wallis H test for independent samples.

        Parameters:
        - *groups: Variable length argument containing data for each group. Each argument should be a list or array-like
        object.

        Returns:
        - statistic: The calculated test statistic.
        - p_value: The p-value associated with the test statistic.
        """
        statistic, p_value = kruskal(*groups)
        return statistic, p_value

    def friedman_test(self, *groups):
        """
        Perform Friedman test for related samples.

        Parameters:
        - *groups: Variable length argument containing data for each group. Each argument should be a list or array-like
        object representing measurements of the same individuals under different conditions.

        Returns:
        - statistic: The calculated test statistic.
        - p_value: The p-value associated with the test statistic.
        """
        statistic, p_value = friedmanchisquare(*groups)
        return statistic, p_value

    def qq_plots(self, variable_names, *data_samples, distribution='norm'):
        """
        Generate Q-Q plots for multiple data samples.

        Parameters:
        - *variable_names: List with the names of the variables to be plotted
        - data_samples: Variable number of 1D array-like objects representing the data samples.
        - distribution: String indicating the theoretical distribution to compare against. Default is 'norm' for normal
        distribution.

        Returns:
        - None (displays the Q-Q plots)
        """
        num_samples = len(data_samples)
        num_rows = (num_samples + 1) // 2  # Calculate the number of rows for subplots
        num_cols = 2 if num_samples > 1 else 1  # Ensure at least 1 column for subplots

        # Generate Q-Q plots
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 6))
        axes = axes.flatten()  # Flatten axes if multiple subplots

        for i, data in enumerate(data_samples):
            ax = axes[i]
            probplot(data, dist=distribution, plot=ax)
            ax.set_title(f'Q-Q Plot ({distribution})')
            ax.set_xlabel('Theoretical Quantiles')
            ax.set_ylabel(variable_names[i])

        # Adjust layout and show plots
        plt.tight_layout()
        plt.show()

    def test_normality(self, variable_names, *data_samples):
        """
        Test the normality of multiple data samples using Shapiro-Wilk test.

        Parameters:
        - variable_names: List with the names of the variables to be tested.
        - data_samples: Variable number of 1D array-like objects representing the data samples.

        Returns:
        - results: Dictionary containing the test results for each data sample.
                   The keys are the variable names and the values are a tuple (test_statistic, p_value) for
                   Shapiro-Wilk test.
        """
        results = {}
        for name, data in zip(variable_names, data_samples):
            results[name] = shapiro(data)
        for variable_name, shapiro_result in results.items():
            print(f'{variable_name}:')
            print(f'Shapiro-Wilk test - Test statistic: {shapiro_result.statistic}, p-value: {shapiro_result.pvalue}')
        return results

# Load the Iris dataset
iris = load_iris()
sepal_lengths = iris.data[:, 0]  # Sepal length data
sepal_widths = iris.data[:, 1]  # Sepal width data

# Sepal lengths for each species
setosa_lengths = sepal_lengths[iris.target == 0]
versicolor_lengths = sepal_lengths[iris.target == 1]
virginica_lengths = sepal_lengths[iris.target == 2]

# Sepal widths for each species
setosa_widths = sepal_widths[iris.target == 0]
versicolor_widths = sepal_widths[iris.target == 1]
virginica_widths = sepal_widths[iris.target == 2]

# Petal lengths for each species
petal_lengths = iris.data[:, 2]  # Petal length data
setosa_petals = petal_lengths[iris.target == 0]

# Initialize the HypothesisTester class with the data
tester = HypothesisTester()

# Perform normality analysis, first by visual checking using a Q-Q plot and then by normality test
tester.qq_plots(['setosa_lengths', 'versicolor_lengths', 'virginica_lengths', 'setosa_widths',
                 'versicolor_widths', 'virginica_widths'], setosa_lengths, versicolor_lengths,
                virginica_lengths, setosa_widths, versicolor_widths, virginica_widths)
tester.test_normality(['setosa_lengths', 'versicolor_lengths', 'virginica_lengths', 'setosa_widths',
                 'versicolor_widths', 'virginica_widths'], setosa_lengths, versicolor_lengths,
                virginica_lengths, setosa_widths, versicolor_widths, virginica_widths)

# Interpretation:
#
# Shapiro-Wilk Test:
# The Shapiro-Wilk test is a test of normality.
# The test statistic measures the discrepancy between the data and the normal distribution.
# The p-value indicates the probability of observing the data if the null hypothesis (data is normally distributed) is
# true.
# A higher p-value (closer to 1) suggests that the data is more likely to be normally distributed.
# A common significance level used to assess normality is 0.05. If the p-value is greater than 0.05, we fail to reject
# the null hypothesis and conclude that the data is approximately normally distributed.
# For setosa_lengths, versicolor_lengths, and virginica_lengths, the p-values are all greater than 0.05 (0.4595, 0.4647,
# 0.2583), indicating that we fail to reject the null hypothesis of normality. Therefore, we can conclude that these
# variables are approximately normally distributed.
# For setosa_widths, versicolor_widths, and virginica_widths, the p-values are all greater than 0.05 (0.2715, 0.3380,
# 0.1809), indicating that we fail to reject the null hypothesis of normality. Therefore, we can conclude that these
# variables are approximately normally distributed.

# Perform unpaired t-test between Setosa and Versicolor species
t_stat, p_val = tester.unpaired_t_test(setosa_lengths, versicolor_lengths)
print("\nUnpaired t-test between Setosa and Versicolor species:")
print("t-statistic:", t_stat)
print("p-value:", p_val)

# Perform unpaired ANOVA among all three species
f_stat, p_val_anova = tester.unpaired_anova(setosa_lengths, versicolor_lengths, virginica_lengths)
print("\nUnpaired ANOVA among all three species:")
print("F-statistic:", f_stat)
print("p-value:", p_val_anova)

# Perform paired t-test for Sepal length and Petal length within Setosa species
t_stat, p_val = tester.paired_t_test(setosa_lengths, setosa_petals)
print("\nPaired t-test for Sepal length and Petal length within Setosa species:")
print("t-statistic:", t_stat)
print("p-value:", p_val)

# Perform paired ANOVA for Sepal width within all three species
data = pd.DataFrame({
        'value': np.concatenate([setosa_widths, versicolor_widths, virginica_widths]),
        'condition': np.repeat(['setosa', 'versicolor', 'virginica'], len(setosa_widths))
    })
f_stat, p_val = tester.paired_anova(data)
print("\nPaired ANOVA for Sepal width within all three species:")
print("F-statistic:", f_stat)
print("p-value:", p_val)

# Perform Wilcoxon rank-sum test between Setosa and Versicolor species for sepal lengths
statistic, p_value = tester.wilcoxon_ranksum_test(setosa_lengths, versicolor_lengths)
print("\nWilcoxon rank-sum test between Setosa and Versicolor species for sepal lengths:")
print("Test statistic:", statistic)
print("p-value:", p_value)

# Perform Kruskal-Wallis test for Sepal width within all three species
statistic, p_value = tester.kruskal_wallis_test(setosa_widths, versicolor_widths, virginica_widths)
print("\nKruskal-Wallis test for Sepal width within all three species:")
print("Test statistic:", statistic)
print("p-value:", p_value)

# Perform Wilcoxon signed-rank test for Sepal length and Petal length within Setosa species
statistic, p_value = tester.wilcoxon_signedrank_test(setosa_lengths, setosa_petals)
print("\nWilcoxon signed-rank test for Sepal length and Petal length within Setosa species:")
print("Test statistic:", statistic)
print("p-value:", p_value)

# Perform Friedman test for Sepal length within all three species
statistic, p_value = tester.friedman_test(setosa_lengths, versicolor_lengths, virginica_lengths)
print("\nFriedman test for Sepal length within all three species:")
print("Test statistic:", statistic)
print("p-value:", p_value)

# Interpretation:
#
# Unpaired t-test between Setosa and Versicolor species:
# t-statistic: The calculated t-statistic is approximately -10.52 (indicates that, on average, the sepal lengths of
# Setosa species are lower than those of the Versicolor species). This value represents the difference in means between
# the sepal lengths of Setosa and Versicolor species.
# p-value: The p-value associated with the t-statistic is approximately 8.99e-18. This p-value is very small, indicating
# strong evidence against the null hypothesis.
# In this case, it suggests that the difference in sepal lengths between Setosa and Versicolor species is statistically
# significant.
#
# Unpaired ANOVA among all three species:
# F-statistic: The calculated F-statistic is approximately 119.26. This value represents the ratio of variability
# between groups to variability within groups in sepal lengths among all three species.
# p-value: The p-value associated with the F-statistic is approximately 1.67e-31. The t-test, this p-value is extremely
# small, indicating strong evidence against the null hypothesis.
# It suggests that there is a statistically significant difference in sepal lengths among at least one pair of species.
#
# Paired t-test for Sepal length and Petal length within Setosa species:
# t-statistic: The calculated t-statistic is approximately 71.84. This value represents the difference in means between
# sepal lengths and petal lengths within the Setosa species relative to the variability in the data.
# p-value: The p-value associated with the t-statistic is approximately 2.54e-51. This p-value is extremely small,
# indicating strong evidence against the null hypothesis.
# It suggests that the difference between sepal lengths and petal lengths within the Setosa species is statistically
# significant.
#
# Paired ANOVA for Sepal width within all three species:
# F-statistic: The calculated F-statistic is approximately 49.16. This value represents the ratio of variability between
# groups to variability within groups in sepal widths among all three species.
# p-value: The p-value associated with the F-statistic is approximately 4.49e-17.This p-value is very small, indicating
# strong evidence against the null hypothesis.
# It suggests that there is a statistically significant difference in sepal widths among at least one pair of species.
#
# Wilcoxon rank-sum test between Setosa and Versicolor species for sepal lengths:
# Test statistic: The calculated test statistic is approximately 168.5. This value represents the difference in ranks
# between the two groups (Setosa and Versicolor) for sepal lengths.
# p-value: The p-value associated with the test statistic is approximately 8.35e-14. This small p-value indicates strong
# evidence against the null hypothesis.
# It suggests that there is a statistically significant difference in sepal lengths between Setosa and Versicolor
# species.
#
# Kruskal-Wallis test for Sepal width within all three species:
# Test statistic: The calculated test statistic is approximately 63.57. This value represents the sum of ranks across
# all groups for sepal widths.
# p-value: The p-value associated with the test statistic is approximately 1.57e-14. Similar to the previous result,
# this small p-value indicates strong evidence against the null hypothesis.
# It suggests that there is a statistically significant difference in sepal widths among the three species (Setosa,
# Versicolor, and Virginica).
#
# Wilcoxon signed-rank test for Sepal length and Petal length within Setosa species:
# Test statistic: The calculated test statistic is 1275.0. This value represents the sum of signed ranks of differences
# between the paired observations (sepal lengths and petal lengths) within the Setosa species.
# p-value: The p-value associated with the test statistic is approximately 8.88e-16. This small p-value indicates
# strong evidence against the null hypothesis.
# It suggests that there is a statistically significant difference between sepal lengths and petal lengths within the
# Setosa species.
#
# Friedman test for Sepal length within all three species:
# Test statistic: The calculated test statistic is approximately 73.79. This value represents the sum of ranks across
# all groups for sepal lengths.
# p-value: The p-value associated with the test statistic is approximately 9.50e-17. Once more, this small p-value
# indicates strong evidence against the null hypothesis.
# It suggests that there is a statistically significant difference in sepal lengths among the three species (Setosa,
# Versicolor, and Virginica).