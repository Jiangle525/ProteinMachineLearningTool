from sklearn.linear_model import LogisticRegression
from sklearn import tree, svm
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


def LR(**params):
    if not params:
        lr_optimizer = LogisticRegression(max_iter=10000)
    else:
        lr_optimizer = LogisticRegression(**params)
    return lr_optimizer


def SVM(**params):
    return svm.SVC(probability=True, **params)


def DT(**params):
    return tree.DecisionTreeClassifier(**params)


def RF(**params):
    return RandomForestClassifier(**params)


def GBDT(**params):
    return GradientBoostingClassifier(**params)


def LightGBM(**params):
    return lgb.LGBMClassifier(**params)


def XGboost(**params):
    return xgb.XGBClassifier(**params)


def cnn(lr=0.0001, input_shape=(400, 1), output_shape=2):
    model = Sequential()
    model.add(Conv1D(128, 3, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(3))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(output_shape, activation='sigmoid'))
    adam = Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=1)
    model.compile(loss='binary_crossentropy',
                  optimizer=adam, metrics=['accuracy'])
    return model
