import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import seaborn as sns
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument('stats', type=Path)
parser.add_argument('--num_top', type=int, default=40)
parser.add_argument('--num_normal', type=int, default=40)
args = parser.parse_args()


df = pd.read_csv(args.stats)
df = df.sort_values(by='importance', ascending=False)

mean_all = df['importance'].mean()
max_all = df['importance'].max()

top_inds = df.head(args.num_top)['feature_ind'].values
first = (df.shape[0] - args.num_normal) // 2
last = first + args.num_normal
normal_inds = df.iloc[first:last]['feature_ind'].values

all_inds = (np.concatenate([top_inds, normal_inds]))
# random shuffle all_inds
np.random.shuffle(all_inds)

mean_norm = df[df['feature_ind'].isin(normal_inds)]['importance'].mean()
mean_top = df[df['feature_ind'].isin(top_inds)]['importance'].mean()

print(f'{max_all=} {mean_all=} {mean_norm=} {mean_top=} {len(all_inds)=} {len(normal_inds)=} {len(top_inds)=}')
print(f'neurons = {list(all_inds)}')

sns.histplot(df, x='importance')
# make labels and axis tick labels large
plt.xlabel('Importance', fontsize=16)
plt.ylabel('Count', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()
# sns.ecdfplot(df, x='importance')
# sns.ecdfplot(df[df['feature_ind'].isin(all_inds)], x='importance', color='red')
plt.show()
