import time
from lib.Visualization import MyCanvas
from PyQt5.QtCore import pyqtSignal, QObject


class MySignal(QObject):
    lineEdit_System_Tips = pyqtSignal(str)  # System Tips
    progressBar = pyqtSignal(int)  # Progress Bar
    widget_Params_Encoding_Model_CrossValidation_SuperParams = pyqtSignal(str, str, str, str)  # Params
    tabWidget_File_TrainFile_TestFile_Prediction_Preparation = pyqtSignal(str, str, str, str)  # File Display
    tabWidget_Metrics_ROCCurve_ConfusionMatrix_ClassificationReport = pyqtSignal(MyCanvas, MyCanvas, dict)  # Metrics Display
    textBrowser_Message = pyqtSignal(str)  # Message Display
    graphicsView_Feature = pyqtSignal(str)  # Feature Display


def my_emit(sig, *args):
    sig.emit(*args)
    time.sleep(0.0001)


signal = MySignal()
