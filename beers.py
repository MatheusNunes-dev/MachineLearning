# %%

import pandas as pd

# df = dataFrame
df = pd.read_excel("data/dados_cerveja.xlsx")
df.head()

# %%
y = df["classe"]

features = ["temperatura", "copo", "espuma", "cor"]
x = df[features]

x = x.replace({"mud": 1, "pint": 2,
     "sim": 1, "não": 0,
     "clara": 0, "escura": 1 
})
# %%
from sklearn import tree
model = tree.DecisionTreeClassifier(random_state=42)

model.fit(x, y)
# %%

import matplotlib.pyplot as plt

plt.figure(dpi=400)
tree.plot_tree(model, feature_names= features, class_names= model.classes_ , filled= True)


# %%

proba = model.predict_proba([[1,1,1,1]])[0]

pd.Series(proba, index=model.classes_)
# %%
