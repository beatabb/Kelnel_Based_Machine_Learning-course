import pandas as pd
import numpy as np
import ast

# results of parameter validation and than test set of best C parameters with non overlapping data
df_param_validation_nonoverlap = pd.read_csv("results/non_overlap_param_val.csv")
df_nonoverlap_test_set = pd.read_csv("results/experiment_nonoverlap_results.csv").drop('Unnamed: 0',axis=1)


# results of parameter validation and than test set of best C parameters with overlapping data
df_param_validation_overlap = pd.read_csv("results/overlap_params_results.csv")
df_overlap_test_set = pd.read_csv("results/experiment_overlap_results.csv").drop('Unnamed: 0',axis=1)


#results of using only one svm
df_single_svm = pd.read_csv("results/results_single_svm.csv").drop('Unnamed: 0',axis=1)



def group_by_machines_and_return_max_c(df):
    means = df.drop('Unnamed: 0',axis=1).groupby('num_svms', as_index=False).mean()
    max = means['f1'].max()
    best_results = means.loc[means['f1'] == max]

    print("\n----- Averaged scores grouped by number of machines ------\n")
    print(means)
    #best num machines
    num_machines = best_results['num_svms']

    # print(int(num_machines))

    print("\n----- Scores for number of machines with highest average F1 score ------\n")
    # best C param
    result_best_svms = df.loc[df['num_svms'] == int(num_machines)].drop('Unnamed: 0',axis=1).sort_values('f1', ascending=False)

    print(result_best_svms)
    best_C_param = result_best_svms['c'][:3]
    print("\n----- 3 best F1 scores ------\n")
    print(best_C_param)
    print("\n---------------------------------------------------------------------------\n")
    # returns the best C
    return result_best_svms['c'][:1].values[0]



def eval_test_for_best_c(df, c):
    ### u can evaluate all cs that u want by just setting c to a fixed value

    # c = 1000 #for example


    #creates list of best c: 100 -> [100,100,100,100,100] depends on num svms
    cs = [int(c) for i in range(df.num_svms.min())]

    filter_best_c = []
    for index, row in df.iterrows():
        #convert string in df to list of ints
        r = list(map(int,ast.literal_eval(row.c)))
        if r == (cs): 
            filter_best_c = row
    print(filter_best_c)


def combination_evaluation(df):
    print(df.sort_values('f1', ascending=False))

def eval_single_svm(df):
    print(df)



####### ALL results using the NOT overlapping data buckets ############

print("\n####### Evaluation of the best parameters using validation set using non overlapping data ############\n")
best_c_nonoverlap = group_by_machines_and_return_max_c(df_param_validation_nonoverlap)

print("\n------  get test run results of best validated c of nonoverlapping data ------\n")
eval_test_for_best_c(df_nonoverlap_test_set, best_c_nonoverlap)

print("\n------  get best results of all cs, including the combination of cs of non overlapping data ------\n")
combination_evaluation(df_nonoverlap_test_set)




####### ALL results using the  OVERLAPPING data buckets ############

print("\n####### Evaluation of the best parameters using validation set using overlapping data ############\n")
best_c_overlap = group_by_machines_and_return_max_c(df_param_validation_nonoverlap)

print("\n####### Evaluation of the best parameters using validation set using overlapping data ############\n")
eval_test_for_best_c(df_overlap_test_set, best_c_overlap)

print("\n------  get best results of all cs, including the combination of cs of non overlapping data ------\n")
combination_evaluation(df_overlap_test_set)

########### Results single SVM ###############
### Attention time in seconds !!! ###

print("\n####### Evaluation of the single SVM for comparison ############\n")

eval_single_svm(df_single_svm)