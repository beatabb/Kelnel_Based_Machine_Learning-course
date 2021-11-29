from zipfile import ZipFile
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, multilabel_confusion_matrix
from sklearn.metrics import plot_confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.preprocessing import LabelEncoder



sns.set_style("white")


def preprocess():
    df_train = pd.read_csv('dataset/KDDTrain+.txt',delimiter=',', header=None)
    df_test = pd.read_csv('dataset/KDDTest+.txt',delimiter=',', header=None)

    # The CSV file has no column heads, so add them
    df_train.columns = [
        'duration',
        'protocol_type',
        'service',
        'flag',
        'src_bytes',
        'dst_bytes',
        'land',
        'wrong_fragment',
        'urgent',
        'hot',
        'num_failed_logins',
        'logged_in',
        'num_compromised',
        'root_shell',
        'su_attempted',
        'num_root',
        'num_file_creations',
        'num_shells',
        'num_access_files',
        'num_outbound_cmds',
        'is_host_login',
        'is_guest_login',
        'count',
        'srv_count',
        'serror_rate',
        'srv_serror_rate',
        'rerror_rate',
        'srv_rerror_rate',
        'same_srv_rate',
        'diff_srv_rate',
        'srv_diff_host_rate',
        'dst_host_count',
        'dst_host_srv_count',
        'dst_host_same_srv_rate',
        'dst_host_diff_srv_rate',
        'dst_host_same_src_port_rate',
        'dst_host_srv_diff_host_rate',
        'dst_host_serror_rate',
        'dst_host_srv_serror_rate',
        'dst_host_rerror_rate',
        'dst_host_srv_rerror_rate',
        'outcome',
        'difficulty'
    ]
    df_test.columns = df_train.columns

    class_DoS = ['apache2', 'back', 'land', 'neptune', 'mailbomb', 'pod', 
                'processtable', 'smurf', 'teardrop', 'udpstorm', 'worm']
    class_Probe = ['ipsweep', 'mscan', 'nmap', 'portsweep', 'saint', 'satan']

    class_U2R = ['buffer_overflow', 'loadmodule', 'perl', 'ps', 'rootkit', 'sqlattack', 'xterm']

    class_R2L = ['ftp_write', 'guess_passwd', 'httptunnel',  'imap', 'multihop', 'named', 
                'phf', 'sendmail', 'snmpgetattack', 'spy', 'snmpguess', 'warezclient', 
                'warezmaster', 'xlock', 'xsnoop']



    df_train['class'] = df_train['outcome']
    df_train['class'].replace(class_DoS, value='DoS', inplace=True)
    df_train['class'].replace(class_Probe, value='Probe',inplace=True)
    df_train['class'].replace(class_U2R, value='U2R',inplace=True)
    df_train['class'].replace(class_R2L, value='R2L', inplace=True)
    print(df_train['class'].unique())

    df_test['class'] = df_test['outcome']
    df_test['class'].replace(class_DoS, value='DoS', inplace=True)
    df_test['class'].replace(class_Probe, value='Probe',inplace=True)
    df_test['class'].replace(class_U2R, value='U2R',inplace=True)
    df_test['class'].replace(class_R2L, value='R2L', inplace=True)
    labels = df_test['class'].unique()
    print(labels)

    '''convert nominal label to numerical ones '''
    labels={"normal": 0, "DoS": 1, "Probe": 2, "U2R": 3, "R2L":4}
    df_train['label'] = df_train['class'].replace(labels)
    df_test['label'] = df_test['class'].replace(labels)
    df_train['label'].nunique()
    
    # df_train = df_train[df_train['class'].isin(['DoS', 'Probe', 'R2L', 'normal'])]
    # df_test = df_test[df_test['class'].isin(['DoS', 'Probe', 'R2L', 'normal'])]
    # label_num = len(df_test['class'].unique())
    # label_num
    # print(df_train['label'].unique())
    # print(df_train['class'].unique())
 

    #outlier removing@
    df_train_obj = df_train.iloc[:, :-4].select_dtypes(include='object')
    df_train_num = df_train.iloc[:, :-4].select_dtypes(exclude='object')

    df_test_obj = df_test.iloc[:, :-4].select_dtypes(include='object')
    df_test_num = df_test.iloc[:, :-4].select_dtypes(exclude='object')



    #hot encode and minmax

    enc = OneHotEncoder(handle_unknown='ignore')
    # df_train_obj = df_train.iloc[:,:-3]
    df_train_enc = enc.fit_transform(df_train_obj).toarray()
    train_enc_features = enc.get_feature_names(input_features=df_train_obj.columns)
    df_test_enc = enc.transform(df_test_obj).toarray()
    test_enc_features = enc.get_feature_names(input_features=df_test_obj.columns)
    # print(len(train_enc_features), len(test_enc_features))
    X_train_enc = np.c_[df_train_num, df_train_enc]
    X_test_enc = np.c_[df_test_num, df_test_enc]

    scaler = MinMaxScaler()
    X_train_scaler = scaler.fit_transform(X_train_enc)
    X_test_scaler = scaler.transform(X_test_enc)

    # X_train_normal = X_train_scaler[df_train['class']=='normal']
    # X_train_DoS = X_train_scaler[df_train['class']=='DoS']
    # X_train_Probe = X_train_scaler[df_train['class']=='Probe']
    # #X_train_U2R = X_train_scaler[df_train['class']=='U2R']
    # X_train_R2L = X_train_scaler[df_train['class']=='R2L']

    # X_test_normal = X_test_scaler[df_test['class']=='normal']
    # X_test_DoS = X_test_scaler[df_test['class']=='DoS']
    # X_test_Probe = X_test_scaler[df_test['class']=='Probe']
    # #X_test_U2R = X_test_scaler[df_test['class']=='U2R']
    # X_test_R2L = X_test_scaler[df_test['class']=='R2L']

    X_train = X_train_scaler
    X_test = X_test_scaler
    
    le = LabelEncoder()
    y_train = le.fit_transform(df_train['class'])
    y_test = le.transform(df_test['class'])

    X_train = X_train.reshape(len(X_train), 122)
    y_train = y_train.reshape(len(y_train), 1)
    X_test = X_test.reshape(len(X_test), 122)
    y_test = y_test.reshape(len(y_test), 1)


    train = np.append(X_train,y_train, axis=1)
    df_train = pd.DataFrame(data=train)
    df_train.iloc[: , -1]= df_train.iloc[: , -1].astype('int')

    #df_train.to_csv('dataset/kdd_train.csv', index=False)

    test = np.append(X_test,y_test, axis=1)
    df_test = pd.DataFrame(data=test)
    df_test.iloc[: , -1]= df_test.iloc[: , -1].astype('int')
    #df_test.to_csv('dataset/kdd_test.csv', index=False)


def combine():
    X = np.append(X_train, X_test).reshape(148398, 122)
    y = np.append(y_train, y_test).reshape(148398, 1)
    data = np.append(X,y, axis=1)
    df = pd.DataFrame(data=data)
    df.iloc[: , -1]= df.iloc[: , -1].astype('int')

    return df