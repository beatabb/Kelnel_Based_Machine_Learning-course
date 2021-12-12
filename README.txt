The code was used for experimentation and is therefore not written to be run in once. 

If you want to run the final model (non-overlapping SVM ensemble with best parameters) you need to do:

- Please run the file svm.py which will automatically run the function run_final_model() which runs our final model on the test set.
- These SVMs are bulding the final ensemble with which we are using in the report:
svm_0_data_non_n_5_c_[100, 100, 100, 100, 100].sav 
svm_1_data_non_n_5_c_[100, 100, 100, 100, 100].sav 
svm_2_data_non_n_5_c_[100, 100, 100, 100, 100].sav
svm_3_data_non_n_5_c_[100, 100, 100, 100, 100].sav
svm_4_data_non_n_5_c_[100, 100, 100, 100, 100].sav

The function run_single_experiment() can be used to create a new single SVM with parameters c, n, data.

The function parameter_eval() is used for parameter validation, trains on the training set and predicts on the validation set.

The function experimentation_and_training() is used for the final evaluation on test data, it will run on the training set and predict on test set.

For evaluating our results we wrote the script exp_and_eval.py.

