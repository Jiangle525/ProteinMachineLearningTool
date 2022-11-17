import copy
import time

import numpy as np
from PySide2.QtWidgets import QGraphicsScene
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import KFold

from lib.MySignal import *
from lib.Visualization import draw_roc
from lib.FeatureExtraction import *
from lib.Visualization import *
from lib.ModelDefinition import *


def LoadData(fileName, fileIndex=0):
    data = []
    lines = open(fileName, 'r', encoding='utf-8').readlines()
    linesLength = len(lines)
    one_percent = max(1, linesLength // 100)
    i = 0
    while i < linesLength:
        if lines[i][0] == '>':
            name = lines[i].strip()
            i += 1
            sequence = ''
            while i < linesLength and lines[i][0] != '>':
                sequence += lines[i].strip()
                i += 1
            data.append((name, sequence))
        else:
            i += 1
        if i % one_percent == 0 or i == linesLength:
            my_emit(signal.progressBar, 100 * i / linesLength)
            file_content = ['', '', '', '']
            file_content[fileIndex] = '\n'.join(
                ['\n'.join(item) for item in data[-(one_percent if i % one_percent == 0 else i % one_percent):]])
            my_emit(signal.tabWidget_File_TrainFile_TestFile_Prediction_Preparation, *file_content)
    return data


def GetSequencesLabels(data):
    labels, sequences = [], []
    for item in data:
        label = int(item[0].split('|')[1])
        sequence = item[1]
        labels.append(label)
        sequences.append(sequence)
    return sequences, labels


def k_fold_cross_validation(base_model, X, y, k=10):
    fprs, tprs = [], []
    best_auc = 0
    best_model = None
    i = 1
    kf = KFold(n_splits=k, shuffle=False)  # 初始化KFold
    for train_index, test_index in kf.split(X):
        my_emit(signal.progressBar, 100 * i // k)
        model = copy.deepcopy(base_model)
        x_train, y_train = X[train_index], y[train_index]
        x_test, y_test = X[test_index], y[test_index]
        model.fit(x_train, y_train)
        y_pred_prob = model.predict_proba(x_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
        if auc(fpr, tpr) > best_auc:
            best_auc = auc(fpr, tpr)
            best_model = model
        fprs.append(fpr)
        tprs.append(tpr)
        i += 1
    my_emit(signal.lineEdit_System_Tips, "{} fold training completed!".format(k))
    return best_model, fprs, tprs


def StartTrain(trainFileData, testFileData, encodingName, encodingParams, modelName, modelParams, validation):
    train_sequences, train_labels = GetSequencesLabels(trainFileData)
    test_sequences, test_labels = GetSequencesLabels(testFileData)

    encodingFunc = globals()[encodingName]
    X_train, X_test = encodingFunc(train_sequences, **encodingParams), encodingFunc(test_sequences, **encodingParams)
    y_train, y_test = np.array(train_labels), np.array(test_labels)

    classifier_func = globals()[modelName]
    model = classifier_func(**modelParams)
    bestModel, fprs, tprs = k_fold_cross_validation(model, X_train, y_train, k=validation)

    fig = draw_roc(fprs, tprs)
    my_emit(signal.lineEdit_System_Tips, 'Training is overed!')
    graphicsSceneROC = QGraphicsScene(fig)

    # my_emit(signal.tabWidget_Metrics_ROCCurve_ConfusionMatrix_ClassificationReport, )
