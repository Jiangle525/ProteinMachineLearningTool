import copy

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

from lib.MySignal import *
from lib.FeatureExtraction import *
from lib.ModelDefinition import *
from lib.ShareInfo import shareInfo
from lib.Visualization import *


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
        '''需要处理data名字没有格式化！！！！！！！！！'''
        label = int(item[0].split('|')[-2])
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
    my_emit(signal.progressBar, 0)
    for train_index, test_index in kf.split(X):
        my_emit(signal.lineEdit_System_Tips, '{} fold is training...'.format(i))
        model = copy.deepcopy(base_model)
        x_train, y_train = X[train_index], y[train_index]
        x_test, y_test = X[test_index], y[test_index]
        model.fit(x_train, y_train)
        y_pred_prob = model.predict_proba(x_test)[:, 1]
        y_pred = model.predict(x_test)
        cm = confusion_matrix(y_test,y_pred)
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
        if auc(fpr, tpr) > best_auc:
            best_auc = auc(fpr, tpr)
            best_model = model
        fprs.append(fpr)
        tprs.append(tpr)
        my_emit(signal.progressBar, 100 * i // k)
        my_emit(signal.textBrowser_Message, str((cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1])))
        i += 1
    my_emit(signal.lineEdit_System_Tips, "{} fold training completed!".format(k))
    return best_model, fprs, tprs


def GetMetrics(model, X, y):
    cr = dict()
    y_pred = model.predict(X)
    cm = confusion_matrix(y, y_pred)
    y_pred_prob = model.predict_proba(X)[:, 1]
    TN, FP, FN, TP = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    cr['Accuracy'] = (TN + TP) / (TN + FP + FN + TP), (TN + TP) / (TN + FP + FN + TP)
    cr['Precision'] = TN / (FN + TN), TP / (FP + TP)
    cr['Recall'] = TN / (TN + FP), TP / (TP + FN)
    cr['F1'] = 2 * cr['Precision'][0] * cr['Recall'][0] / (cr['Precision'][0] + cr['Recall'][0]), \
               2 * cr['Precision'][1] * cr['Recall'][1] / (cr['Precision'][1] + cr['Recall'][1])
    if (TP * TN - FP * FN) != 0:
        cr['MCC'] = (TP * TN - FP * FN) / (math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))), \
                    (TP * TN - FP * FN) / (math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)))
    else:
        cr['MCC'] = 0, 0
    cr['AUC'] = roc_auc_score(y, y_pred_prob), roc_auc_score(y, y_pred_prob)
    return cm, cr


def StartTrain(trainFileData, testFileData, encodingName, encodingParams, modelName, modelParams, validation):
    train_sequences, train_labels = GetSequencesLabels(trainFileData)
    test_sequences, test_labels = GetSequencesLabels(testFileData)

    encodingFunc = globals()[encodingName]
    my_emit(signal.progressBar, 0)
    my_emit(signal.lineEdit_System_Tips, 'Encoding train set...')
    X_train = encodingFunc(train_sequences, **encodingParams)
    y_train = np.array(train_labels)
    my_emit(signal.progressBar, 0)
    my_emit(signal.lineEdit_System_Tips, 'Encoding test set...')
    X_test = encodingFunc(test_sequences, **encodingParams)
    y_test = np.array(test_labels)

    classifier_func = globals()[modelName]
    model = classifier_func(**modelParams)
    shareInfo.menuModel.trainedModel, fprs, tprs = k_fold_cross_validation(model, X_train, y_train, k=validation)
    cm, cr = GetMetrics(shareInfo.menuModel.trainedModel, X_test, y_test)
    shareInfo.menuModel.canvasROC = draw_roc(fprs, tprs)
    shareInfo.menuModel.canvasConfusionMatrix = draw_confusion_matrix(cm)
    shareInfo.menuModel.classificationReport = cr

    my_emit(signal.tabWidget_Metrics_ROCCurve_ConfusionMatrix_ClassificationReport,
            *[shareInfo.menuModel.canvasROC, shareInfo.menuModel.canvasConfusionMatrix, cr])


def StartPrediction(predictionFileData, model, encodingName, encodingParams):
    predictionSequences = [i[1] for i in predictionFileData]
    encodingFunc = globals()[encodingName]
    X_prediction = encodingFunc(predictionSequences, **encodingParams)
    shareInfo.menuPrediction.listPredictionResult = model.predict(X_prediction)
