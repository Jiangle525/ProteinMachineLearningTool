import itertools
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from sklearn.metrics import auc, confusion_matrix, roc_curve
import numpy as np
import math


class MyCanvas(FigureCanvas):

    def __init__(self, width=6, height=4, dpi=50):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MyCanvas, self).__init__(fig)


def draw_history(history):
    """_summary_

    Args:
        history (_type_): object of history

    Returns:
        _type_: fig
    """

    fig_history = plt.figure(figsize=(10, 5))

    fig_trian = plt.subplot(1, 2, 1)
    fig_trian.plot(history.history['accuracy'])
    fig_trian.plot(history.history['val_accuracy'])
    fig_trian.set_title('Train accuracy')
    fig_trian.set_ylabel('Accuracy')
    fig_trian.set_xlabel('Epoch')
    fig_trian.legend(['Train', 'Validation'], loc='upper left')

    fig_validation = plt.subplot(1, 2, 2)
    fig_validation.plot(history.history['loss'])
    fig_validation.plot(history.history['val_loss'])
    fig_validation.set_title('Train loss')
    fig_validation.set_ylabel('Loss')
    fig_validation.set_xlabel('Epoch')
    fig_validation.legend(['Train', 'Validation'], loc='upper left')

    return fig_history


def draw_roc(fprs, tprs):
    """_summary_

    Args:
        fprs (_type_): list of fpr
        tprs (_type_): list of tpr

    Returns:
        _type_: fig
    """

    colors = ['darkorange', 'aqua', 'cornflowerblue',
              'blueviolet', 'deeppink', 'cyan']
    canvasROC = MyCanvas()
    canvasROC.axes.set_title('ROC curve')
    canvasROC.axes.set_xlim([0.0, 1.0])
    canvasROC.axes.set_ylim([0.0, 1.0])
    canvasROC.axes.set_xlabel('False Positive Rate')
    canvasROC.axes.set_ylabel('True Positive Rate')
    canvasROC.axes.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
                        label='Random', alpha=0.8)

    aucs = []
    mean_fpr = np.linspace(0, 1, max([len(i) for i in fprs]))
    interp_tprs = []

    for i in range(len(fprs)):
        canvasROC.axes.plot(fprs[i], tprs[i], lw=1, alpha=0.5, color=colors[i % len(colors)],
                            label='{} fold (AUC = {:.2f})'.format(i + 1, auc(fprs[i], tprs[i])))
        interp_tpr = np.interp(mean_fpr, fprs[i], tprs[i])
        interp_tprs.append(interp_tpr)
        aucs.append(auc(fprs[i], tprs[i]))

    mean_tpr = np.mean(interp_tprs, axis=0)
    mean_tpr[0], mean_tpr[-1] = 0, 1
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = round(np.std(aucs), 2)
    std_auc = 0.01 if not std_auc else std_auc

    canvasROC.axes.plot(mean_fpr, mean_tpr, color='b', label=r'Mean (AUC = {:.2f} ± {})'.format(
        mean_auc, std_auc), lw=2, alpha=0.8)

    std_tpr = np.std(interp_tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    canvasROC.axes.fill_between(mean_fpr, tprs_lower, tprs_upper, color='slategray', alpha=0.2,
                                label=r'± 1 std. dev.')

    canvasROC.axes.legend(loc="lower right")
    return canvasROC


def draw_confusion_matrix(cm, labels=('pos', 'neg'), percentage=False, cmap=plt.cm.Blues):
    """_summary_

    Args:
        cm (_type_): confusion matrix
        labels (tuple, optional): class of label. Defaults to ('pos', 'neg').
        percentage (bool, optional): display percentage. Defaults to False.
        cmap (_type_, optional): colors of confusion matrix . Defaults to plt.cm.Blues.

    Returns:
        _type_: fig
    """

    canvasCM = MyCanvas()
    canvasCM.axes.set_title('Confusion matrix')
    canvasCM.axes.set_xlabel('Predicted label')
    canvasCM.axes.set_ylabel('True label')
    # ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    if percentage:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    canvasCM.axes.imshow(cm, interpolation='nearest', cmap=cmap)
    ticks = np.arange(len(labels))
    canvasCM.axes.set_xticks(ticks=ticks)
    canvasCM.axes.set_yticks(ticks=ticks)
    canvasCM.axes.set_xticklabels(labels)
    canvasCM.axes.set_yticklabels(labels)
    fmt = '.2%' if percentage else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        canvasCM.axes.text(j, i, format(cm[i][j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    return canvasCM


def test_draw_history(data):
    model = cnn()
    encoded = DDE(data)
    encoded = np.asarray(encoded).astype(
        np.float32).reshape(encoded.shape + (1,))
    label = [[0, 1]] * (math.ceil(len(encoded) / 2))
    label.extend([[1, 0]] * (len(encoded) // 2))
    label = np.array(label, dtype=int)
    shuffle_data_set(encoded, label)
    X_train, y_train, X_test, y_test = train_test_split(encoded, label)
    history = model.fit(X_train, y_train, validation_data=(
        X_test, y_test), epochs=10, batch_size=32)
    fig = draw_history(history)

    plt.show()


def test_draw_roc(data):
    model = cnn()

    encoded = DDE(data)
    encoded = np.asarray(encoded).astype(np.float32)
    label = [[0, 1]] * (math.ceil(len(encoded) / 2))
    label.extend([[1, 0]] * (len(encoded) // 2))
    label = np.array(label, dtype=int)
    shuffle_data_set(encoded, label)
    X_train, y_train, X_test, y_test = train_test_split(encoded, label)

    model.fit(X_train, y_train, validation_data=(
        X_test, y_test), epochs=10, batch_size=32)
    y_pred_prob = model.predict(X_test)
    y_pred_prob1 = model.predict(X_train)

    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
    fpr1, tpr1, thresholds1 = roc_curve(y_train, y_pred_prob1)
    fig = draw_roc([fpr, fpr1], [tpr, tpr1])
    plt.show()


def test_draw_confusion_matrix(data):
    model = cnn()
    encoded = DDE(data)
    encoded = np.asarray(encoded).astype(np.float32)
    label = [[0, 1]] * (math.ceil(len(encoded) / 2))
    label.extend([[1, 0]] * (len(encoded) // 2))
    label = np.array(label, dtype=int)
    shuffle_data_set(encoded, label)
    X_train, y_train, X_test, y_test = train_test_split(encoded, label)

    model.fit(X_train, y_train, validation_data=(
        X_test, y_test), epochs=10, batch_size=32)
    y_pred_prob = model.predict(X_test)

    cm = confusion_matrix(np.argmax(y_test, axis=1),
                          np.argmax(y_pred_prob, axis=1))
    fig = draw_confusion_matrix(cm)
    plt.show()


if __name__ == '__main__':
    from DataProcessing.DataPreparation import load_data, train_test_split, shuffle_data_set
    from ManualFeatures.DDE import DDE
    from Model.Definition.CNN import cnn

    data = load_data('../../data/pos-all.fasta')
    test_draw_history(data)
    test_draw_roc(data)
    test_draw_confusion_matrix(data)
