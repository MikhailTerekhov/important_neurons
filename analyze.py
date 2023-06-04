import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('importance', type=Path)
parser.add_argument('answers', type=Path)
args = parser.parse_args()

imp_data = pd.read_csv(args.importance)
ans_data = pd.read_csv(args.answers)
for i in [1, 3, 4, 5, 6]:
    ans_data[f"question{i}"] -= 1

ans_data["interpretability"] = ans_data["question4"] / 8 + ans_data["question1"] * ans_data["question3"] / 32
ans_data["polysemanticity"] = ans_data["question5"] / 8 + ans_data["question6"] / 8

print(f"mean interp = {ans_data['interpretability'].mean()}")

# merge imp_data and ans_data by neuron index
merged = pd.merge(imp_data, ans_data, left_on="feature_ind", right_on="neuron", how="inner")
# create a new column that stores "none" if the neuron is not in the top 40, and "top" if it is
merged["top"] = np.where(merged["importance"] > 50, "Top 40 features", "Middle 40 features")

print(merged.groupby("top")["interpretability"].mean())

fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
ax[0].set_title("Middle 40 features", fontsize=18)
sns.histplot(merged.loc[merged["importance"] < 50], x="interpretability", ax=ax[0], bins=10, binrange=(0, 1))

ax[1].set_title("Top 40 features", fontsize=18)
sns.histplot(merged.loc[merged["importance"] > 50], x="interpretability", ax=ax[1], bins=10, binrange=(0, 1))
# Make larger labels and tick labels

ax[0].set_ylabel("Count", fontsize=16)
ax[0].set_xlabel("Interpretability", fontsize=16)
ax[1].set_xlabel("Interpretability", fontsize=16)
#  ax[1].set_ylabel("Count", fontsize=16)
ax[0].tick_params(labelsize=14)
ax[1].tick_params(labelsize=14)

plt.figure()
sns.scatterplot(merged, x="importance", y="interpretability", hue="top")

# Make larger labels and tick labels
plt.xlabel("Importance", fontsize=16)
plt.ylabel("Interpretability", fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(title=None, fontsize=14)



# sns.displot(ans_data, x="interpretability", kind="kde", rug=True)
# Do a kde plot of interpretability with ticks at value locations

plt.show()
