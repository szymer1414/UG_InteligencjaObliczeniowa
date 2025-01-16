import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import mlxtend as ml
titanic = pd.read_csv("titanic.csv")
print(titanic.head(3))

#sns.countplot(x = 'Survived', data = titanic, order = titanic['Survived'].value_counts().iloc[:10].index)
#plt.xticks(rotation=90)
titanic['Survived'] = titanic['Survived'].map({'No': 0, 'Yes': 1})


#df = titanic.groupby(['Class','Age']).size().reset_index(name='count')
basket = (titanic
          .groupby(['Sex', 'Age','Class'])['Survived']
          .sum().unstack().reset_index().fillna(0)
          .set_index(['Sex', 'Age']))
def encode_units(x):
 return x > 0
basket_sets = basket.applymap(encode_units)
#basket_sets.drop('POSTAGE', inplace=True, axis=1)
frequent_itemsets = apriori(basket_sets, min_support=0.005, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=0.5,num_itemsets=len(frequent_itemsets))
print("rules:",rules)
filtered_rules = rules[(rules['lift'] >= 6) & (rules['confidence'] >= 0.8)]

print(basket)
#print(frequent_itemsets)
#print(filtered_rules)
basket.plot(kind='bar', stacked=True, figsize=(12, 7), colormap='viridis')


plt.title('Survived Counts by Sex, Age, and Class', fontsize=16)
plt.xlabel('Sex and Age', fontsize=12)
plt.ylabel('Number of Survivors', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.legend(title='Class', fontsize=10)
plt.tight_layout()

plt.show()