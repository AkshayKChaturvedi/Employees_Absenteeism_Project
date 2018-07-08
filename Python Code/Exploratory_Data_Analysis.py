import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

from Utility_Functions import impute_missing_values, split_data

df = pd.read_excel('C:/Users/Dell/Downloads/Absenteeism_at_work.xls')

# To check number of missing values in each column
print(df.info())

num = ['Distance from Residence to Work', 'Service time', 'Age', 'Work load Average/day ', 'Transportation expense',
       'Hit target', 'Son', 'Pet', 'Weight', 'Height', 'Body mass index', 'Absenteeism time in hours']

cat = list(set(df.columns) - set(num))

# Fill missing values
for var in num:
    df[var] = impute_missing_values(data=df, col=var, method='median')

for var in cat:
    df[var] = impute_missing_values(data=df, col=var, method='mode')

print(df.info())

x_train, x_test, y_train, y_test = split_data(df.iloc[:, df.columns != 'Absenteeism time in hours'],
                                              df.iloc[:, -1])

df_train = pd.concat([x_train, y_train], axis=1)

num.remove('Absenteeism time in hours')

# -------------- Histogram of every numerical variable with Normal distribution curve superimposed over it -------------

fig = plt.figure()
for i, var_name in enumerate(num):
    mu, std = norm.fit(df[var_name])
    ax = fig.add_subplot(4, 3, i+1)
    df_train[var_name].hist(density=True, edgecolor='k', color='w', ax=ax)
    xmin, xmax = min(df_train[var_name]), max(df_train[var_name])
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    ax.plot(x, p, 'k', linewidth=2)
    ax.set_title(var_name)
fig.tight_layout()  # Reduces size of each plot to fit each variable name thereby improving the appearance
plt.show()

# ----------------------------------------- Correlation plot of numerical variables ------------------------------------

plt.matshow(df_train[num].corr())
plt.xticks(range(len(df_train[num].columns)), df_train[num].columns, fontsize=8, rotation=90)
plt.yticks(range(len(df_train[num].columns)), df_train[num].columns)
plt.colorbar()
plt.show()

# ------------------------------------------ Box plot of every numerical variable --------------------------------------

fig = plt.figure()
for i, var_name in enumerate(num):
    ax = fig.add_subplot(5, 3, i+1)
    df_train.boxplot(column=var_name, ax=ax, flierprops=dict(marker='.', markerfacecolor='black', markersize=4))
fig.tight_layout()  # Reduces size of each plot to fit each variable name thereby improving the appearance
plt.show()

# ----------------------------------------------------- End ------------------------------------------------------------
