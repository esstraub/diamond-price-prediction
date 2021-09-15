# Import required files and initialise the input path
import matplotlib
import numpy as np
import pandas as pd  # data processing, CSV file from GitHub
import seaborn as sns  # plot
import matplotlib.pyplot as plt
from statsmodels.stats.anova import anova_lm
from statsmodels.formula.api import ols
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

diamondData = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/diamonds.csv').dropna()
diamondData.head()

print(diamondData)

# EDA - Examine distribution of target variable (Price)
sns.displot(diamondData['price'])
plt.title("Distribution of Price Variable", family="DejaVu Sans", size=18, weight=50)
plt.tight_layout()
matplotlib.pyplot.show()

# Set Color Palette for EDA
# colors = ['#003f5c', '#2f4b7c', '#665191', '#a05195', '#d45087', '#f95d6a', '#ff7c43', '#ffa600']
colors = sns.color_palette("magma")
sns.set(palette=colors, font='DejaVu Sans', style='white',
        rc={'axes.facecolor': 'whitesmoke', 'figure.facecolor': 'whitesmoke'})
sns.palplot(colors)
plt.title("Color Palette for EDA", family="DejaVu Sans", size=15, weight=50)
plt.tight_layout()
matplotlib.pyplot.show()

# EDA - Explore relationship between price and other variables
int_cols = diamondData.select_dtypes(exclude='object').columns.to_list()

# print(int_cols)
int_cols.remove('price')
j = 0
fig = plt.figure(figsize=(15, 10), constrained_layout=True)
plt.suptitle("Regression of the Numeric variables", font='DejaVu Sans', size=20, weight='bold')
for i in int_cols:
    ax = plt.subplot(331 + j)
    ax = sns.regplot(data=diamondData, x=i, y='price', color=colors[1], line_kws={'color': '#ffa600'})
    ax.set_title("Price and {} comparision analysis".format(i), family='DejaVu Sans')
    for s in ['left', 'right', 'top', 'bottom']:
        ax.spines[s].set_visible(False)

    j = j + 1
plt.tight_layout()
matplotlib.pyplot.show()  # Works to show linear relationship but there are outliers

# EDA - Distribution of ints
int_cols = diamondData.select_dtypes(exclude='object').columns.to_list()
j = 0
fig = plt.figure(figsize=(15, 10), constrained_layout=True)
plt.suptitle("Distribution of the Numeric variables", family='DejaVu Sans', size=20, weight='bold')
for i in int_cols:
    ax = plt.subplot(331 + j)
    # ax.set_title('Title')
    # print(df[i])
    ax = sns.kdeplot(data=diamondData, x=i, color=colors[0], fill=True, edgecolor=colors[-1], alpha=1)
    ax.set_title("Distribution of Numeric variables - {}".format(i), family='DejaVu Sans')
    for s in ['left', 'right', 'top', 'bottom']:
        ax.spines[s].set_visible(False)

    j = j + 1
plt.tight_layout()
matplotlib.pyplot.show()

# Correlation with Price
np.triu(np.ones_like(diamondData.corr()))
plt.figure(figsize=(16, 6))
mask = np.triu(np.ones_like(diamondData.corr(), dtype=np.bool))
heatmap = sns.heatmap(diamondData.corr(), mask=mask, vmin=-1, vmax=1, annot=True, cmap='BrBG')
heatmap.set_title('Triangle Correlation Heatmap to Display Corrolation with Price', fontdict={'fontsize': 18}, pad=16)

fig = plt.figure(figsize=(15, 10), constrained_layout=True)

# Need to find the target variable relationship with Categorical variables
plt.suptitle("Categorical feature comparison with Price", family='Sherif', size=20, weight='bold')
cat_cols = diamondData.select_dtypes(include='object').columns.to_list()
ax = fig.subplot_mosaic("""
                        AAB
                        AAC
                        AAD
                        """)
sns.kdeplot(diamondData['price'], fill=True, edgecolor=colors[-1], linewidth=2, color=colors[0], ax=ax['A'], alpha=0.8)
ax['A'].text(x=2000, y=0.00023, s="Comparing Price with Categorical feature - Median is simlar",
             family='San', fontweight='bold')
sns.boxplot(data=diamondData, x=cat_cols[0], y='price', ax=ax['B'])
sns.boxplot(data=diamondData, x=cat_cols[1], y='price', ax=ax['C'])
sns.boxplot(data=diamondData, x=cat_cols[2], y='price', ax=ax['D'])
for i in 'ABCD':
    for s in ['left', 'right', 'top', 'bottom']:
        ax[i].spines[s].set_visible(False)
plt.tight_layout()
matplotlib.pyplot.show()

# View distribuion of top 3 (cut,
cat_cols = diamondData.select_dtypes(include='object').columns.to_list()

fig = plt.figure(figsize=(15, 5))
plt.suptitle("Distribution of Primary Categorical Variables", family='DejaVu Sans', size=20, weight='bold')
ax1 = plt.subplot(131)
sns.countplot(data=diamondData, x=cat_cols[0], ax=ax1, linewidth=2, edgecolor=colors[-1])
for s in ['left', 'right', 'top', 'bottom']:
    ax1.spines[s].set_visible(False)
ax2 = plt.subplot(132, sharey=ax1)
sns.countplot(data=diamondData, x=cat_cols[1], ax=ax2, linewidth=2, edgecolor=colors[-1])
for s in ['left', 'right', 'top', 'bottom']:
    ax2.spines[s].set_visible(False)
ax3 = plt.subplot(133, sharey=ax1)
sns.countplot(data=diamondData, x=cat_cols[2], ax=ax3, linewidth=2, edgecolor=colors[-1])
for s in ['left', 'right', 'top', 'bottom']:
    ax3.spines[s].set_visible(False)
plt.tight_layout()
matplotlib.pyplot.show()

# We now understand the data, time test hypothosis - Compare Price value with Categorical feature
# H0 = there is no significant difference
# H1 = there is a significant difference
formula = 'price ~ C(clarity)'
model = ols(formula, diamondData).fit()
print(np.round(anova_lm(model, typ=2), 3))
print(model.summary())
if np.round(model.f_pvalue, 2) < 0.05:
    print("Reject Null Hypothesis and accept the alternate hypothesis")
else:
    print("Accept the Null Hypothesis")

formula='price ~ C(color)'
model=ols(formula, diamondData).fit()
print(np.round(anova_lm(model, typ=2),3))
print(model.summary())
if np.round(model.f_pvalue,2)<0.05:
    print("Reject Null Hypothesis and accept the alternate hypothesis")
else:
    print("Accept the Null Hypothesis")

formula='price ~ C(cut)'
model=ols(formula, diamondData).fit()
print(np.round(anova_lm(model, typ=2),3))
print(model.summary())
if np.round(model.f_pvalue,2)<0.05:
    print("Reject Null Hypothesis and accept the alternate hypothesis")
else:
    print("Accept the Null Hypothesis")

formula='price ~ C(cut)+C(color)+C(clarity)'
model=ols(formula, diamondData).fit()
print(np.round(anova_lm(model, typ=2),3))
print(model.summary())
if np.round(model.f_pvalue,2)<0.05:
    print("Reject Null Hypothesis and accept the alternate hypothesis")
else:
    print("Accept the Null Hypothesis")

#Verdict : Price has significant impact on the Cut, Clarity & Color of the Dimond

#Make it machine-readable using one hot encoding.
df1=pd.get_dummies(diamondData, columns=cat_cols, drop_first=True)
df1.head()

#Train the test split
X=df1.drop('price', axis=1)
y=df1['price']

X_train: object
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=42)


#creating Polynomial instead of linear relationships
scaler = PolynomialFeatures(degree=2, interaction_only=True)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train, y_train)
pred=model.predict(X_test)
print()

fig=plt.figure(figsize=(15,8))
residual = y_test - pred
plt.suptitle("Comparing y_test and Predicted value", family='Sherif', size=20, weight='bold')
ax=fig.subplot_mosaic("""AA
                          BB
                          CC""")
sns.scatterplot(y_test, residual, ax=ax['A'])
ax['A'].axhline(y=0, ls='--', c=colors[-1], linewidth=3)
sns.kdeplot(residual, ax=ax['B'], fill=True, color=colors[0], edgecolor=colors[-1], linewidth=2)

from sklearn.metrics import mean_squared_error
ax['C'].text(x=0.2,y=0.2,s="Root squared mean error: {}".format(np.round(mean_squared_error(y_test, pred, squared=False),2)), ha='left',family='cursive' ,weight='bold', size=15, style='italic')
ax['C'].text(x=0.2,y=0.4,s="Accuracy of model with Train data: {}".format(np.round(model.score(X_train, y_train),2)), ha='left',family='cursive' ,weight='bold', size=15, style='italic')
ax['C'].text(x=0.2,y=0.6,s="Accuracy of model with Test data: {}".format(np.round(model.score(X_test, y_test),2)), ha='left',family='cursive' ,weight='bold', size=15, style='italic')
ax['C'].text(x=0.2,y=0.8,s="Result:", ha='left',family='cursive' ,weight='bold', size=15, style='italic')

ax['C'].axis('off')

for i in 'ABC':
    for s in ['left','right','top','bottom']:
        ax[i].spines[s].set_visible(False)

matplotlib.pyplot.show()

