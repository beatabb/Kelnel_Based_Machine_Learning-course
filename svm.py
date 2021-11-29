import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import time
from sklearn.preprocessing import MinMaxScaler
from preprocess import preprocess
import pickle
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier


#num of svms
N = 1


### DATA ###
df = pd.read_csv('dataset/kdd_train.csv')
d_train= np.array(df)
test = np.array(pd.read_csv('dataset/kdd_test.csv'))
#############

def prepare_data(data):
    #shuffle data on every run
    np.random.shuffle(data)
    #split in N chunks
    new_data = np.array_split(data, N)
    return np.array(new_data)

#create multiple svms and store in array
def get_mult_svm():
    SVMs = [SVC(C=1, kernel='rbf', gamma=10) for i in range(N)]
    return SVMs

#train every svm on the subset
#cnt is number of rounds
def train(svms, train_data, cnt):
    for c in range(cnt):
        # shuffle data random for every new training round
        data = prepare_data(train_data)
        for i in range(len(svms)):
            d = data[i]
            X = d[:, 0:-1]
            y = d[:,-1] 
            svm = svms[i]
            start_time = time.time()
            print("Training of SVM nr " + str(i) + " in round " + str(c+1) + " started at ... " + str(time.time()))
            svm.fit(X,y)
            print("--- %s seconds ---" % (time.time() - start_time))
    return svms


def predict(svms, test):
    X = test[:, 0:-1]
    y = test[:,-1] 
    y_pred = []
    for elem in range(len(X)):
        pred = []
        for i in range(len(svms)):
            #prediction of svm nr i
            p = svms[i].predict(X[elem].reshape(1, -1))
            pred.append(int(p.squeeze()))
            # vote which is the actual predition
        pred = np.bincount(pred).argmax()
        y_pred.append(pred)
    y_pred = np.asarray(y_pred)  
    print("Accuracy:", accuracy_score(y, y_pred))
    ### confusion matrix
    cm = confusion_matrix(y, y_pred)
    fig, ax = plt.subplots(figsize=(8, 8))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["normal", "DoS", "Probe", "U2R", "R2L"]
                           ).plot(ax=ax)
    plt.title('Confusion Marix of majority voting SVM')
    plt.show()


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


def train_tree(data):
    X = data[:, 0:-1]
    y = data[:,-1] 
    clf = RandomForestClassifier(max_depth=40, random_state=10, verbose=1)
    clf.fit(X,y)
    return clf

def pred_tree(test,clf):
    X = test[:, 0:-1]
    y = test[:,-1] 
    y_pred = clf.predict(X)
    print("Accuracy:", accuracy_score(y, y_pred))


#create N svms and return them in list
svms = get_mult_svm()
print(len(d_train))
#train N svms in cnt rounds
train(svms, d_train, cnt=1)
#save_models(svms)

#svms = load_models()
#save_models(svms)
predict(svms, test=test)

#clf = train_tree(d_train)
#pred_tree(test, clf)