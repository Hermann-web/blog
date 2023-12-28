---
date: 2023-11-08
authors: [hermann-web]
description: >
  This blog shows interesting stuff to know
  It is flavored by mkdocs
categories:
  - frameworks
  - data
  - python
  - frontend
  - data-visualization
  - beginners
links:
  - blog/posts/a-roadmap-for-web-dev.md
  - blog/posts/code-practises/software-licences.md
title: Seaborn in Practice
---


# Seaborn in Practice: Syntax and Guide

Seaborn is a powerful data visualization library in Python that provides a high-level interface for drawing attractive and informative statistical graphics. One common misconception about Seaborn and programming in general is the necessity to remember all the syntax. In reality, it's more about understanding the tool's capabilities and how to leverage its functions to visualize data effectively.

??? question "So, what can i do exactly with seaborn ?"

## import the library
```python
import seaborn as sns
sns.set(style="whitegrid")
```

<!-- more -->

## Load some dataset
we will be using a dataset containing tips from a restaurant. We will know more more about it down the road
```python
df = sns.load_dataset("tips")
```

Let's see a preview of the dataset
```python
df
```

|   | total_bill | tip  | sex    | smoker | day  | time   | size |
|---|------------|------|--------|--------|------|--------|------|
| 0 | 16.99      | 1.01 | Female | No     | Sun  | Dinner | 2    |
| 1 | 10.34      | 1.66 | Male   | No     | Sun  | Dinner | 3    |
| 2 | 21.01      | 3.50 | Male   | No     | Sun  | Dinner | 3    |
| 3 | 23.68      | 3.31 | Male   | No     | Sun  | Dinner | 2    |
| 4 | 24.59      | 3.61 | Female | No     | Sun  | Dinner | 4    |
|...| ...        | ...  | ...    | ...    | ...  | ...    | ...  |
|239| 29.03      | 5.92 | Male   | No     | Sat  | Dinner | 3    |
|240| 27.18      | 2.00 | Female | Yes    | Sat  | Dinner | 2    |
|241| 22.67      | 2.00 | Male   | Yes    | Sat  | Dinner | 2    |
|242| 17.82      | 1.75 | Male   | No     | Sat  | Dinner | 2    |
|243| 18.78      | 3.00 | Female | No     | Thur | Dinner | 2    |

We have an overview of the data but it is not enough. We will do a broad visualisation of the columns with pairplot.

## Pairplot

A quick way to visualize relationships in a dataset is by using the method `pairplot`.

```python
sns.pairplot(df)
```
??? Output "Result"
    ![pairplot example](./seaborn-in-practice/1-pairplot-example.png)

You can see here, we have a table of graphs. The 3 rows and 3 columns correpond to the 3 numerical values in out dataset: `tip`, `total_bill` and `size`
In each cell, one column is plot against another:
  - In the diagonals, a column is plotted againt itselt and you have histograms 
  - In the anti-diagonals, 2 columns are plotted against each other and you have a scatterplot 

You can also notice only 3 columns of our dataframe is here. It is because they contain numerical values. The 3 others (`sex`, `smoker`, `day` and `time`) are 

## Histogram

Histograms are used both in univariate statistics and multivariate statistics
To display the average notes by gender, you can use a bar plot:

```python
import seaborn as sns
# group by gender, then get the column "notes" then, compute the mean of notes in a group
df1 = df.groupby('gender')[['notes']].mean().reset_index()
sns.set(style="whitegrid")
# show a barplot of out new dataframe (mean_notes = fct(gender))
ax = sns.barplot(x="gender", y="notes", data=df1)
```

Counting occurrences can be visualized using a categorical plot:

```python
sns.catplot(x='gender', kind='count', data=ratings_df)
```

You can extend this to visualize counts by gender and skin color:

```python
sns.catplot(x='gender', hue='couleur', kind='count', data=ratings_df)
```

Further stratifying by region:

```python
sns.catplot(x='gender', hue='couleur', row='region', kind='count', data=ratings_df, height=3, aspect=2)
```


## Scatterplot

Scatterplots offer a powerful way to visualize relationships between two variables:

To represent points based on 'eval' as a function of 'age':

```python
ax = sns.scatterplot(x='age', y='eval', data=ratings_df)
```

You can distinguish points by gender using different colors:

```python
ax = sns.scatterplot(x='age', y='eval', hue='sex', data=ratings_df)
```

For more complex visualizations involving multiple categorical variables:

```python
sns.relplot(x="age", y="eval", hue="sex", row="region", data=ratings_df, height=3, aspect=2)
```

Including regression lines on scatterplots:

```python
sns.lmplot(data=ratings_df, x="var1", y="var2", height=5, aspect=1.5)  # Height 5, width 1.5 times larger than height
```

Creating scatter plots with histograms for marginal distributions:

```python
sns.jointplot(data=df, x="var1", y="var2", height=3.5)
```

## Boxplot

Boxplots provide a visual summary of the distribution of data:

To view the average ages and percentiles at 5% and 95%:

```python
sns.boxplot(ratings_df['age'], orient='v')
```

Visualizing the average notes and percentiles for each gender:

```python
ax = sns.boxplot(x='sexe', y='notes', data=ratings_df)
```

Further stratifying data for insights:

```python
df["xxx_grp"] = pd.cut(df.xxx, [18, 30, 40, 50, 60, 70, 80])  # Creating age strata
sns.boxplot(x="xxx_grp", y="xxx", hue="yyy", data=df)  # Optional hue for differentiation
```

## Distribution Plot

Understanding the distribution of data:

```python
ax = sns.distplot(ratings_df['notes'], kde=False)
```

Analyzing note distribution by gender:

```python
sns.distplot(ratings_df[ratings_df['sexe'] == 'female']['eval'], color='green', kde=False)
sns.distplot(ratings_df[ratings_df['sexe'] == 'male']['eval'], color="orange", kde=False)
plt.show()
```

## Heatmap

Utilizing heatmaps to visualize numerical data:

```python
corr = new_df.corr()  # Calculating feature correlations
ax = sns.heatmap(corr, vmin=0, vmax=1, cmap="YlGnBu", annot=True)
plt.savefig('seabornPandas.png')
plt.show()
```

## Conclusion

These examples showcase how Seaborn can be effectively utilized for various visualization needs without the necessity to memorize all the syntax.


Understanding the basic syntax and functionality of Seaborn allows you to explore various plots and graphs that suit your data analysis requirements. Through simple examples and by focusing on the visual representation of data, you can gain deeper insights without the burden of remembering intricate details.

Remember, Seaborn is designed to assist in the visual exploration of your data, offering a wide range of options for customizing and fine-tuning plots to suit your specific needs.

Experiment with different plot types and functionalities to better understand the story your data has to tell. And don't hesitate to refer to the documentation and various online resources available to enrich your understanding and application of Seaborn.

Let the visualization journey begin, and may your data tell its story vividly through Seaborn!

