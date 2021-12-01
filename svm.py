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

# num of svms
N = 10

### DATA ###
df = pd.read_csv('dataset/kdd_train.csv')
d_train = np.array(df)
d_train_X = d_train[:, 0:-1]
d_train_y = d_train[:, -1]
X_train, X_validation, y_train, y_validation = train_test_split(d_train_X, d_train_y, test_size=0.2)
test = np.array(pd.read_csv('dataset/kdd_test.csv'))
d_test_X = test[:, 0:-1]
d_test_y = test[:, -1]

sum_class = len(d_train_y)

class_weights = [int(((1 - (list(d_train_y).count(i) / sum_class))/4)*100) for i in range(5)]
print(class_weights)


#############

def prepare_data(data):
    # split in N chunks
    new_data = np.array_split(data, N)
    return np.array(new_data)


# create multiple svms and store in array
def get_mult_svm():
    weights = {0: class_weights[0], 1: class_weights[1], 2: class_weights[2], 3: class_weights[3], 4: class_weights[4]}

    SVMs = [SVC(C=(i + 1) * 10, kernel='linear', class_weight=weights) for i in range(N)]
    return SVMs


def train_baggingClf(X, y):
    print("start training")
    clf = BaggingClassifier(base_estimator=SVC(kernel="linear", verbose=True),
                            n_estimators=10, random_state=0)
    clf.fit(X, y)
    return clf


# train every svm on the subset
def train(svms):
    data_X = prepare_data(X_train)
    data_y = prepare_data(y_train)
    for i in range(len(svms)):
        X = data_X[i]
        y = data_y[i]
        svm = svms[i]
        start_time = time.time()
        print("Training of SVM nr " + str(i) + " in round started at ... " + str(time.time()))
        svm.fit(X, y)
        print("--- %s seconds ---" % (time.time() - start_time))
    return svms


def predict(svms, test_X, test_y):
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


# create N svms and return them in list
svms = get_mult_svm()
# print(len(d_train))
# train N svms in cnt rounds
train(svms)
# save_models(svms)

# svms = load_models()
# save_models(svms)
predict(svms, d_test_X, d_test_y)

# clf = train_tree(d_train)
# pred_tree(test, clf)

# clf = train_baggingClf(d_train)
# pred_bag(test, clf)
