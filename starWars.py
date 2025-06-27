# %%
import pandas as pd

df = pd.read_parquet("data/dados_clones.parquet")

# %%

y = df["Status "]

features = ["Massa(em kilos)", "Estatura(cm)"]
x = df[features]


# %%

from sklearn import tree

model = tree.DecisionTreeClassifier(random_state=42)

model.fit(x, y)

# %%

import matplotlib.pyplot as plt

plt.figure(dpi = 400)

tree.plot_tree(model, feature_names= features, class_names= model.classes_, filled= True, max_depth=3)