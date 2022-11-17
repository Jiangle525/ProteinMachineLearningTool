import time
from PySide2.QtCore import Signal, QObject


class MySignal(QObject):
    lineEdit_System_Tips = Signal(str)  # System Tips
    progressBar = Signal(int)  # Progress Bar
    widget_Params_Encoding_Model_CrossValidation_SuperParams = Signal(str, str, str, str)  # Params
    tabWidget_File_TrainFile_TestFile_Prediction_Preparation = Signal(str, str, str, str)  # File Display
    tabWidget_Metrics_ROCCurve_ConfusionMatrix_ClassificationReport = Signal(str, str, str)  # Metrics Display
    textBrowser_Message = Signal(str)  # Message Display
    graphicsView_Feature = Signal(str)  # Feature Display


def my_emit(sig, *args):
    sig.emit(*args)
    time.sleep(0.0001)


signal = MySignal()
