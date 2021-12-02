import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import time
from sklearn.preprocessing import MinMaxScaler
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


test = np.array(pd.read_csv('dataset/kdd_test.csv'))

sum_class = len(d_train_y)

class_weights = [int(((1 - (list(d_train_y).count(i) / sum_class))/4)*1000) for i in range(5)]


#############

def prepare_data(data, n):
    #shuffle data on every run
    np.random.shuffle(data)
    #split in N chunks
    new_data = np.array_split(data, n)
    return np.array(new_data)


def prepare_data_overlapping(data, n):
    np.random.shuffle(data)
    temp_buckets = np.array_split(data, n)

    np.random.shuffle(data)
    new_random_bucks = np.array_split(data, n)
    new_data = [np.concatenate((np.array(temp_buckets[i]).squeeze(), np.array(new_random_bucks[i]).squeeze()), axis=0) for i in range(len(temp_buckets))]

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

    for i in range(len(svms)):
        d = data[i]
        X = d[:, 0:-1]
        y = d[:,-1] 
        svm = svms[i]
        start_time = time.time()
        print("Training of SVM nr " + str(i) + " started at ... " + str(time.time()))
        svm.fit(X,y)
        print("--- %s seconds ---" % (time.time() - start_time))
    return svms


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
    print("Accuracy:", accuracy_score(test_y, y_pred))
    # report
    cr_matrix = classification_report(test_y, y_pred)
    print(cr_matrix)
    ### confusion matrix
    cm = confusion_matrix(test_y, y_pred)
    fig, ax = plt.subplots(figsize=(8, 8))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["normal", "DoS", "Probe", "U2R", "R2L"]
                           ).plot(ax=ax)
    plt.title('Confusion Marix of majority voting SVM')
    plt.show()


#### Saving and Loading the SVMs

def save_models(models):
    for i in range(len(models)):
        filename = 'models/svm_{}.sav'.format(i)
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

n = 15
c_parameters = [(i+1)*10 for i in range(n)]
kernel = 'linear'

svms = get_mult_svm(c_parameters=c_parameters, kernel=kernel, n=n)
data = prepare_data(train_data, n)
#data = prepare_data_overlapping(train_data, n)
#print(len(d_train))
#train N svms in cnt rounds
train(svms, data)
#save_models(svms)
#svms = load_models()
#save_models(svms)
predict(svms, test=test)

# clf = train_tree(train_data)
# pred_tree(test, clf)


