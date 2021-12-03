import pandas as pd
import numpy as np

df = pd.read_csv("results/7_10_results.csv")



means = df.drop('Unnamed: 0',axis=1).groupby('num_svms', as_index=False).mean()
max = means['f1'].max()
best_results = means.loc[means['f1'] == max]

print(means)
#best num machines
num_machines = best_results['num_svms']

# print(int(num_machines))

# best C param
result_best_svms = df.loc[df['num_svms'] == int(num_machines)]

print(result_best_svms)
best_C_param = result_best_svms.sort_values('f1', ascending=False)['c'][:3]


