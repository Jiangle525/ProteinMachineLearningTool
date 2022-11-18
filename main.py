import sys
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5 import uic
from lib.DealAction import DealAction
from lib.MySignal import signal
import ui.background  # 加载图片


class My_Main_Window(QMainWindow, DealAction):

    def __init__(self):
        super().__init__()

        # Load main window
        self.ui_Main = uic.loadUi("ui/MainWindow.ui")

        '''Menubar'''
        # File
        self.ui_Main.action_Import_Train_File.triggered.connect(super().action_Import_Train_File)
        self.ui_Main.action_Import_Test_File.triggered.connect(super().action_Import_Test_File)
        self.ui_Main.action_Import_Prediction_File.triggered.connect(super().action_Import_Prediction_File)
        self.ui_Main.action_Import_Preparation_File.triggered.connect(super().action_Import_Preparation_File)

        self.ui_Main.action_Import_Model.triggered.connect(super().action_Import_Model)

        self.ui_Main.action_Import_Feature.triggered.connect(super().action_Import_Feature)

        self.ui_Main.action_Close_All_Files.triggered.connect(super().action_Close_All_Files)

        self.ui_Main.action_Clear_Output.triggered.connect(super().action_Clear_Output)

        self.ui_Main.action_Exit.triggered.connect(super().action_Exit)

        # Preparation
        self.ui_Main.action_Duplication.triggered.connect(super().action_Duplication)

        self.ui_Main.action_Length_Clipping.triggered.connect(super().action_Length_Clipping)

        self.ui_Main.action_CD_HIT.triggered.connect(super().action_CD_HIT)

        self.ui_Main.action_Format_File.triggered.connect(super().action_Format_File)

        self.ui_Main.action_Save_Preparation.triggered.connect(super().action_Save_Preparation)

        # Feature
        self.ui_Main.action_Feature_Extraction.triggered.connect(super().action_Feature_Extraction)

        self.ui_Main.action_Save_Feature.triggered.connect(super().action_Save_Feature)

        # Model
        self.ui_Main.action_Start_Training.triggered.connect(super().action_Start_Training)

        self.ui_Main.action_Encoding_Method.triggered.connect(super().action_Encoding_Method)

        self.ui_Main.action_Select_Model.triggered.connect(super().action_Select_Model)

        self.ui_Main.action_Validation.triggered.connect(super().action_Validation)

        self.ui_Main.action_Save_Model.triggered.connect(super().action_Save_Model)

        self.ui_Main.action_Save_All_Metrics.triggered.connect(super().action_Save_All_Metrics)
        self.ui_Main.action_Save_Classification_Report.triggered.connect(super().action_Save_Classification_Report)
        self.ui_Main.action_Save_Confusion_Matrix.triggered.connect(super().action_Save_Confusion_Matrix)
        self.ui_Main.action_Save_Train_Loss.triggered.connect(super().action_Save_Train_Loss)
        self.ui_Main.action_Save_ROC_Curve.triggered.connect(super().action_Save_ROC_Curve)
        self.ui_Main.action_Save_PRC_Curve.triggered.connect(super().action_Save_PRC_Curve)

        # Prediction
        self.ui_Main.action_Start_Prediction.triggered.connect(super().action_Start_Prediction)

        self.ui_Main.action_Save_Prediction_Result.triggered.connect(super().action_Save_Prediction_Result)

        # Visualization
        self.ui_Main.action_Dimension_Reduction.triggered.connect(super().action_Dimension_Reduction)

        self.ui_Main.action_Feature_Ranking.triggered.connect(super().action_Feature_Ranking)

        self.ui_Main.action_Network_Visualization.triggered.connect(super().action_Network_Visualization)

        # Help
        self.ui_Main.action_Document.triggered.connect(super().action_Document)
        self.ui_Main.action_About.triggered.connect(super().action_About)

        # stop
        self.ui_Main.action_Stop.triggered.connect(super().action_Stop)

        # 交互区域
        self.ui_Main.comboBox_Select_Feature.currentIndexChanged.connect(super().comboBox_Select_Feature)

        self.ui_Main.comboBox_Select_Model.currentIndexChanged.connect(super().comboBox_Select_Model)

        # 信号
        signal.lineEdit_System_Tips.connect(super().set_lineEdit_System_Tips)
        signal.progressBar.connect(super().set_progressBar)
        signal.widget_Params_Encoding_Model_CrossValidation_SuperParams.connect(
            super().set_widget_Params_Encoding_Model_CrossValidation_SuperParams)
        signal.tabWidget_File_TrainFile_TestFile_Prediction_Preparation.connect(
            super().set_tabWidget_File_TrainFile_TestFile_PreparationFile)
        signal.tabWidget_Metrics_ROCCurve_ConfusionMatrix_ClassificationReport.connect(
            super().set_tabWidget_Metrics_ROCCurve_ConfusionMatrix_ClassificationReport)
        signal.textBrowser_Message.connect(super().set_textBrowser_Message)
        signal.graphicsView_Feature.connect(super().set_graphicsView_Feature)


if __name__ == '__main__':
    # 启用高分辨率缩放
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    # 使用高像素图片
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)

    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # 设置 fusion 风格
    # app.setStyle('Windows')  # 设置 Windows 风格
    # app.setStyle('WindowsXP')  # 设置 WindowsXP 风格
    # app.setStyle('WindowsVista')  # 设置 WindowsVista 风格
    my_Main_Window = My_Main_Window()
    my_Main_Window.ui_Main.show()
    sys.exit(app.exec_())
