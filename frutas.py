# %%

import pandas as pd

df = pd.read_excel("data/dados_frutas.xlsx")
df

# %%


from sklearn import tree

tree_f = tree.DecisionTreeClassifier(random_state=42)


# %%

y = df['Fruta']

attributes = ['Arredondada', 'Suculenta', 'Vermelha', 'Doce']
x = df[attributes]


# %%
tree_f.fit(x, y)

# %%
tree_f.predict([[0,1,0,0]])


# %%

import matplotlib.pyplot as plt

plt.figure(dpi=400)

tree.plot_tree(tree_f, feature_names = attributes, class_names= tree_f.classes_ , filled= True)

# %%
proba = tree_f.predict_proba([[1,1,1,1]])[0]
pd.Series(proba, index=tree_f.classes_)
# %%
