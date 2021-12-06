import matplotlib.pyplot as plt
import numpy as np
from numpy.random.mtrand import seed
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, f1_score, recall_score, precision_score
import pandas as pd
import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import validation
from preprocess import preprocess
import pickle
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier


### DATA ###
df = pd.read_csv('dataset/kdd_train.csv')
d_train = np.array(df)
d_train_X = d_train[:, 0:-1]
d_train_y = d_train[:, -1]
X_train, X_validation, y_train, y_validation = train_test_split(d_train_X, d_train_y, test_size=0.2)


train_data = np.concatenate((X_train, np.array(y_train).reshape(len(y_train), 1)), axis=1)
validation_data = np.concatenate((X_validation, np.array(y_validation).reshape(len(y_validation), 1)), axis=1)

#pd.DataFrame(validation_data).to_csv('dataset/validation_data.csv')

#validation_data =pd.read_csv("dataset/validation_data.csv")

test = np.array(pd.read_csv('dataset/kdd_test.csv'))

sum_class = len(d_train_y)

class_weights = [int(((1 - (list(d_train_y).count(i) / sum_class))/4)*1000) for i in range(5)]


#############

def prepare_data(data, n):
    check = False
    while(not check):
        #shuffle data on every run
        #split in N chunks
        new_data = np.array_split(data, n)
        check = True
        for arr in new_data: 
            if len(np.unique(np.array(arr)[:,-1])) < 5:
                check = False
    return np.array(new_data)


def prepare_data_overlapping(data, n):
    new_data = []
    check=False
    while(not check):
        #np.random.shuffle(data)
        temp_buckets = np.array_split(data, n)
        np.random.seed(42) 
        np.random.shuffle(data)
        new_random_bucks = np.array_split(data, n)
        new_data = [np.concatenate((np.array(temp_buckets[i]).squeeze(), np.array(new_random_bucks[i]).squeeze()), axis=0) for i in range(len(temp_buckets))]
        check = True
        for arr in new_data: 
            if len(np.unique(np.array(arr)[:,-1])) < 5:
                check = False
        

    return np.array(new_data)


#create multiple svms and store in array
def get_mult_svm(c_parameters, kernel, n):
    weights = {0: class_weights[0], 1: class_weights[1], 2: class_weights[2], 3: class_weights[3], 4: class_weights[4]}
    SVMs = [SVC(C=c_parameters[i], kernel=kernel, class_weight=weights) for i in range(n)]
    return SVMs

#train every svm on the subset
#cnt is number of rounds
def train(svms, data):

    # shuffle data random for every new training round
    start_time = time.time()

    for i in range(len(svms)):
        d = data[i]
        X = d[:, 0:-1]
        y = d[:,-1] 
        svm = svms[i]
        print("Training of SVM nr " + str(i) + " started at ... " + str(time.time()))
        svm.fit(X,y)
    print("--- %s seconds ---" % (time.time() - start_time))
    t = (time.time() - start_time)

    return svms, t


def predict(svms, test):
    test_X = test[:, 0:-1]
    test_y = test[:,-1] 
    y_pred = []
    for elem in range(len(test_X)):
        pred = []
        for i in range(len(svms)):
            # prediction of svm nr i
            p = svms[i].predict(test_X[elem].reshape(1, -1))
            pred.append(int(p.squeeze()))
            # vote which is the actual predition
        pred = np.bincount(pred).argmax()
        y_pred.append(pred)
    y_pred = np.asarray(y_pred)
    acc = accuracy_score(test_y, y_pred)
    f1 = f1_score(test_y, y_pred, average='weighted')
    rec = recall_score(test_y, y_pred, average='weighted')
    prec = precision_score(test_y, y_pred, average='weighted')
    print("Accuracy:", acc)
    print("F1:", f1)
    print("Recall:", rec)
    print("Precision:", prec)

    return acc, f1, rec, prec
    


    # report
    # cr_matrix = classification_report(test_y, y_pred)
    # print(cr_matrix)

    ### confusion matrix
    # cm = confusion_matrix(test_y, y_pred)
    # fig, ax = plt.subplots(figsize=(8, 8))
    # ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["normal", "DoS", "Probe", "U2R", "R2L"]
    #                        ).plot(ax=ax)
    # plt.title('Confusion Marix of majority voting SVM')
    # plt.show()


#### Saving and Loading the SVMs

def save_models(models, d, n, c):
    for i in range(len(models)):
        filename = 'models/svm_{}_data_{}_n_{}_c_{}.sav'.format(i, d, n, c)
        pickle.dump(models[i], open(filename, 'wb'))


def load_models():
    svms = []
    for i in range(N):
        filename = 'models/svm_{}.sav'.format(i)
        svms.append(pickle.load(open(filename, 'rb')))
    return svms

### Random Forest Implementation

def train_tree(X, y):
    clf = RandomForestClassifier(max_depth=40, random_state=10, verbose=1)
    clf.fit(X, y)
    return clf


def pred_tree(X, y, clf):
    y_pred = clf.predict(X)
    print("Accuracy:", accuracy_score(y, y_pred))


def pred_bag(X, y, clf):
    print("predict")
    y_pred = clf.predict(X)
    print("Accuracy:", accuracy_score(y, y_pred))



#create N svms and return them in list
def parameter_eval():
    ns = [ 7, 10]
    cs = [10**(-5), 10**(-4), 10**(-3), 10**(-2), 10**(-1), 1, 10, 100, 1000]
    kernel = 'linear'


    # data = prepare_data_overlapping(train_data, n)

    column_names = ['num_svms', 'c', 'acc', 'f1', 'time']
    df = pd.DataFrame(columns = column_names)

    # np.random.shuffle(train_data)
    # print(np.array(train_data).shape)
    # pd.DataFrame(np.array(train_data)).to_csv('random_train_data.csv')

    train_data = np.asarray(pd.read_csv('random_train_data.csv'))

    train_data = np.delete(train_data, 0, axis=1)


    for n in ns:
        data = prepare_data_overlapping(train_data, n)
        for c in cs:
            c_parameters = [c for i in range(n)]
            print(str(n) + " machines, c-parameter: " + str(c) )
            svms = get_mult_svm(c_parameters=c_parameters, kernel=kernel, n=n)
            
            #data = prepare_data_overlapping(train_data, n)
            #print(len(d_train))
            #train N svms in cnt rounds
            svms, t = train(svms, data)
            #save_models(svms)
            #svms = load_models()
            #save_models(svms)
            acc, f1, rec, prec = predict(svms, test=validation_data)
            df = df.append({'num_svms': n, 'c': c, 'acc': acc, 'f1': f1, 'recall': rec, 'precision':prec ,'time': t}, ignore_index=True)
        df.to_csv('results/overlap_params_results.csv')



def exp2():
    n = 7
    data = ['overlap']
    cs = [
        [10,10,10,10,10, 10,10], [100,100,100,100,100,100, 100], [1000,1000,1000,1000,1000,1000,1000],
        [10,10,10,10,10,10,100], [10,10,10,10,10,10,1000], [100, 100, 100, 100,100,100, 10], [100, 100, 100, 100,100,100, 1000], [1000, 1000, 1000, 1000,1000,1000, 10], [1000, 1000, 1000, 1000,1000,1000, 100],
        [10, 10, 10,10,10, 100, 100], [10, 10, 10,10,10, 1000, 1000], [100, 100, 100,100,100, 10, 10], [100, 100, 100,100,100, 1000, 1000], [1000, 1000, 1000,1000,1000, 10, 10], [1000, 1000, 1000, 1000,1000, 100, 100],
        [10, 10, 10,10,100, 100, 100], [10, 10, 10,10,1000, 1000, 1000], [100, 100, 100,100,10, 10, 10], [100, 100, 100,100,1000, 1000, 1000], [1000, 1000, 1000,1000,10, 10, 10], [1000, 1000, 1000, 1000,100, 100, 100],
        [10,10,10,10,10,100,1000], [100,100,100,100,100,10,1000], [1000,1000,1000,1000,1000,10,100],
        [10,10,10,10,100,100,1000], [10,10,10,10,100,1000,1000],[100,100,100,100,10,10,1000], [100,100,100,100,10,1000,1000], [1000,1000,1000,1000,10,10,100], [1000,1000,1000,1000,10,100,100],
        [10,10,10,100,100,100,1000],[10,10,10,100,100,1000,1000],[10,10,10,100,1000,1000,1000], [100,100,100,10,10,10,1000],[100,100,100,10,10,1000,1000],[100,100,100,10,1000,1000,1000], [100,100,10,10,1000,1000,1000], 
        ]



    kernel = 'linear'


    # data = prepare_data_overlapping(train_data, n)

    column_names = ['data','num_svms', 'c', 'acc', 'f1', 'time']
    df = pd.DataFrame(columns = column_names)

    # np.random.shuffle(train_data)
    # print(np.array(train_data).shape)
    # pd.DataFrame(np.array(train_data)).to_csv('random_train_data.csv')

    train_data = np.asarray(pd.read_csv('random_train_data.csv'))

    train_data = np.delete(train_data, 0, axis=1)

    for d in data:
        if d == 'non': data = prepare_data(train_data, n)
        else: data = prepare_data_overlapping(train_data, n)
        for c in cs:
            print(str(n) + " machines, c-parameter: " + str(c) + 'data: ' + d)
            svms = get_mult_svm(c_parameters=c, kernel=kernel, n=n)
            
            #data = prepare_data_overlapping(train_data, n)
            #print(len(d_train))
            #train N svms in cnt rounds
            svms, t = train(svms, data)
            save_models(svms, d, n, c)
            print("... models saved ...")
            #svms = load_models()
            #save_models(svms)
            acc, f1, rec, prec = predict(svms, test=test)
            print('time: {} f1: {}'.format(t,f1))
            df = df.append({'data':d, 'num_svms': n, 'c': c, 'acc': acc, 'f1': f1, 'recall': rec, 'precision':prec ,'time': t}, ignore_index=True)
            df.to_csv('results/experiment_overlap_results.csv')


#exp2()
#parameter_eval()
# clf = train_tree(train_data)
# pred_tree(test, clf)

column_names = ['data','num_svms', 'c', 'acc', 'f1', 'time']
df = pd.DataFrame(columns = column_names)
cs = [10,100,1000]
for c in cs:
    svms = get_mult_svm(c_parameters=[c], kernel='linear', n=1)
    train_data = np.asarray(pd.read_csv('random_train_data.csv'))
    train_data = np.delete(train_data, 0, axis=1)
    X = train_data[:, 0:-1]
    y = train_data[:,-1] 
    start_time = time.time()
    svms[0].fit(X,y)
    t = (time.time() - start_time)
    acc, f1, rec, prec = predict(svms, test=test)
    print('time: {} f1: {}'.format(t,f1))
    df = df.append({'data': 'non', 'num_svms': 1, 'c': c, 'acc': acc, 'f1': f1, 'recall': rec, 'precision':prec ,'time': t}, ignore_index=True)
    df.to_csv('results/results_single_svm.csv')