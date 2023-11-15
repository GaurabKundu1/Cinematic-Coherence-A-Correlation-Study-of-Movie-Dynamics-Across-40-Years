```markdown
# Cinematic Coherence: A Correlation Study of Movie Dynamics Across 40 Years
## A Project by Gaurab Kundu

### Import Necessary Libraries
```python
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib
plt.style.use('ggplot')
from matplotlib.pyplot import figure
%matplotlib inline
matplotlib.rcParams['figure.figsize'] = (12, 8)
pd.options.mode.chained_assignment = None
```

### Load Dataset
```python
dataFrame = pd.read_csv('movies.csv')
```

### Quick View of Data
```python
dataFrame.head()
```

### Basic Data Exploration
```python
dataFrame.describe()
dataFrame.shape
dataFrame.columns
# There are 7688 rows and 15 columns
```

### Data Cleaning
```python
# Let's loop through the data and see % of missing values
for col in dataFrame.columns:
    pct_missing = np.mean(dataFrame[col].isnull())
    print('{} - {}%'.format(col, round(pct_missing * 100)))

# Let's drop rows with missing values
dataFrame = dataFrame.dropna()
# After dropping rows with missing values, we have 5421 rows and 15 columns

# Data types for columns
dataFrame.dtypes

# Transform data type of columns with floating-point numbers except "score"
dataFrame['votes'] = dataFrame['votes'].astype('int64')
dataFrame['budget'] = dataFrame['budget'].astype('int64')
dataFrame['gross'] = dataFrame['gross'].astype('int64')
dataFrame['runtime'] = dataFrame['runtime'].astype('int64')
dataFrame.dtypes
```

### Hypothesis Testing

#### Hypothesis One
*Movie budget will have a high correlation with Movie Gross Earnings*
```python
# Scatter plot: Correlation between Movie Budget and Gross Earnings 
plt.scatter(x=dataFrame['budget'], y=dataFrame['gross'])
plt.title('Movie Budget vs Gross Earnings')
plt.xlabel('Movie Budget')
plt.ylabel('Gross Earnings')
plt.show()

# Plot budget vs gross using seaborn
sns.regplot(x='budget', y='gross', data=dataFrame, scatter_kws={"color": "blue"}, line_kws={"color": "black"})

# Are there outliers?
dataFrame.boxplot(column=['gross'])

# Types of correlation: pearson, kendall, and spearman
dataFrame.corr(method='pearson', numeric_only=True)  # pearson

# Visualization of pearson product-moment correlation reveals a high correlation between movie budget and gross earnings
correlation_matrix = dataFrame.corr(method='pearson', numeric_only=True)
sns.heatmap(correlation_matrix, annot=True)
plt.title('Correlation Matrix for numeric features')
plt.xlabel('Movie Features')
plt.ylabel('Movie Features')
plt.show()
```
*Hypothesis One is confirmed. Employing Pearson Product Moment Correlation, it appears movie budget is highly correlated with Movie Gross Earnings. The Pearson correlation measures the strength of the linear relationship between two variables. It has a value between -1 to 1, with a value of -1 meaning a total negative linear correlation, 0 being no correlation, and + 1 meaning a total positive correlation. The statistical strength between movie budget and gross earnings is 0.74.*

#### Hypothesis Two
*Movie production company will have a high correlation with Movie Gross Earnings*
```python
# Examine Movie production company
df_numerized = dataFrame

# Converting column with string datatype to numeric
for col_name in df_numerized.columns:
    if(df_numerized[col_name].dtype == 'object'):
        df_numerized[col_name] = df_numerized[col_name].astype('category')
        df_numerized[col_name] = df_numerized[col_name].cat.codes

# Compare highest correlation in matrix table
correlation_mat = df_numerized.corr()
corr_pairs = correlation_mat.unstack()

# Print correlation pairs
print(corr_pairs)

# Sort correlation pairs
sorted_pairs = corr_pairs.sort_values(kind="quicksort")

# Print sorted correlation pairs
print(sorted_pairs)

# Take a look at the ones that have a high correlation (> 0.5)
high_corr = sorted_pairs[abs(sorted_pairs) > 0.5]

# Print high correlation pairs
print(high_corr)
```
*Results: Hypothesis 2, which stated that movie production company will have a high correlation with gross earnings, is wrong. The correlation matrix reveals a low positive correlation between movie production company and gross earnings. The statistical strength between movie production company and gross earning is 0.15.*

### Findings
- Movie votes have a high positive correlation with movie gross earnings.
- According to the correlation matrix, the statistical strength between movie votes and gross earning is 0.614.

*Based on these findings, it's clear that audience engagement, as measured by votes, plays a significant role in the financial success of a movie.*

### Project Summary
*In this project, I conducted a comprehensive analysis of movie dynamics across 40 years, focusing on the correlation between various factors. The exploration involved data cleaning, visualization, and hypothesis testing. While confirming the positive correlation between movie budget and gross earnings, the project debunked the hypothesis that movie production company significantly influences gross earnings. Additionally, the study highlighted the crucial role of audience votes in a movie's financial success, showcasing a strong positive correlation. This project not only honed my Python skills but also provided valuable insights into the intricate dynamics of the film industry.*
```
