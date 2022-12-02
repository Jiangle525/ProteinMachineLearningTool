from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt
from PyQt5 import uic
from lib.MyThreads import *
from lib.ShareInfo import *
from lib.Visualization import *
import matplotlib

matplotlib.use('Qt5Agg')


class MyQTableWidgetItem(QTableWidgetItem):
    def __init__(self, text):
        super(MyQTableWidgetItem, self).__init__(text)

    def __lt__(self, other):
        return float(self.text().strip('%')) < float(other.text().strip('%'))


def DeleteLayoutItem(layout):
    for i in list(range(layout.count()))[::-1]:
        item = layout.itemAt(i)
        if item.widget():
            item.widget().deleteLater()
        layout.removeItem(item)


def GetLayoutItemValue(formLayout):
    dictItems = dict()
    for i in range(formLayout.rowCount()):
        labelItem = formLayout.itemAt(2 * i)
        filedItem = formLayout.itemAt(2 * i + 1)
        if labelItem:
            label = labelItem.widget().text()
            if type(filedItem.widget()) == QComboBox:
                filed = filedItem.widget().currentText()  # QComboBox
                if filed == 'True':
                    filed = True
                if filed == 'False':
                    filed = False
            else:
                filed = filedItem.widget().text()
                if filed.isdigit():
                    filed = int(filed)
                else:
                    filed = float(filed)
            dictItems[label[:-1]] = filed
    return dictItems


def SetLayoutItemValue(formLayout, listWidget):
    dictItems = dict()
    for i in range(formLayout.rowCount()):
        labelItem = formLayout.itemAt(2 * i)
        filedItem = formLayout.itemAt(2 * i + 1)
        if labelItem:
            label = labelItem.widget().text()
            if type(filedItem.widget()) == QComboBox:
                filed = filedItem.widget().currentText()  # QComboBox
                if filed == 'True':
                    filed = True
                if filed == 'False':
                    filed = False
            else:
                filed = filedItem.widget().text()
                if filed.isdigit():
                    filed = int(filed)
                else:
                    filed = float(filed)
            dictItems[label[:-1]] = filed
    return dictItems


def SelectFeature(comboBox, formLayout):
    currentItemText = comboBox.currentText()
    DeleteLayoutItem(formLayout)
    if currentItemText == 'CKSAAP':
        spinBoxGap = QSpinBox()
        spinBoxGap.setMaximum(500)
        spinBoxGap.setMinimum(1)
        formLayout.addRow('gap:', spinBoxGap)
    if currentItemText == 'KSCTriad':
        spinBoxGap = QSpinBox()
        spinBoxGap.setMaximum(500)
        spinBoxGap.setMinimum(1)
        formLayout.addRow('gap:', spinBoxGap)
    if currentItemText == 'CTriad':
        spinBoxGap = QSpinBox()
        spinBoxGap.setMaximum(500)
        spinBoxGap.setMinimum(1)
        formLayout.addRow('gap:', spinBoxGap)
    if currentItemText == 'Word2Vector':
        spinBoxKmer = QSpinBox()
        spinBoxKmer.setMaximum(500)
        spinBoxKmer.setMinimum(1)
        formLayout.addRow('k_mer:', spinBoxKmer)

        spinBoxVectorSize = QSpinBox()
        spinBoxVectorSize.setMaximum(500)
        spinBoxVectorSize.setMinimum(5)
        formLayout.addRow('vector_size:', spinBoxVectorSize)


def SelectModel(comboBox, formLayout):
    currentItemText = comboBox.currentText()
    DeleteLayoutItem(formLayout)

    if currentItemText == 'DT':
        comboBox_criterion = QComboBox()
        comboBox_criterion.addItems(['gini', 'entropy'])
        formLayout.addRow('criterion:', comboBox_criterion)

        spinBox_min_samples_leaf = QSpinBox()
        spinBox_min_samples_leaf.setMaximum(30)
        spinBox_min_samples_leaf.setMinimum(3)
        formLayout.addRow('min_samples_leaf:', spinBox_min_samples_leaf)

        spinBox_min_samples_split = QSpinBox()
        spinBox_min_samples_split.setMaximum(20)
        spinBox_min_samples_split.setMinimum(2)
        formLayout.addRow('min_samples_split:', spinBox_min_samples_split)

    if currentItemText == 'RF':
        spinBox_n_estimators = QSpinBox()
        spinBox_n_estimators.setMaximum(1000)
        spinBox_n_estimators.setMinimum(50)
        spinBox_n_estimators.setSingleStep(50)
        formLayout.addRow('n_estimators:', spinBox_n_estimators)

        comboBox_oob_score = QComboBox()
        comboBox_oob_score.addItems(['True', 'False'])
        formLayout.addRow('oob_score:', comboBox_oob_score)

    if currentItemText == 'SVM':
        spinBox_C = QSpinBox()
        spinBox_C.setMinimum(1)
        spinBox_C.setMaximum(100)
        formLayout.addRow('C:', spinBox_C)

        doubleSpinBox_gamma = QDoubleSpinBox()
        doubleSpinBox_gamma.setMinimum(10 ** -6)
        doubleSpinBox_gamma.setMaximum(10)
        doubleSpinBox_gamma.setDecimals(6)
        doubleSpinBox_gamma.setSingleStep(10 ** -6)
        formLayout.addRow('gamma:', doubleSpinBox_gamma)

        comboBox_kernel = QComboBox()
        comboBox_kernel.addItems(['rbf', 'linear'])
        formLayout.addRow('kernel:', comboBox_kernel)

    if currentItemText == 'LightGBM':
        spinBox_n_estimators = QSpinBox()
        spinBox_n_estimators.setMaximum(1000)
        spinBox_n_estimators.setMinimum(50)
        spinBox_n_estimators.setSingleStep(50)
        formLayout.addRow('n_estimators:', spinBox_n_estimators)

        spinBox_max_depth = QSpinBox()
        spinBox_max_depth.setMaximum(20)
        spinBox_max_depth.setMinimum(3)
        formLayout.addRow('max_depth:', spinBox_max_depth)

        doubleSpinBox_learning_rate = QDoubleSpinBox()
        doubleSpinBox_learning_rate.setDecimals(6)
        doubleSpinBox_learning_rate.setMaximum(10)
        doubleSpinBox_learning_rate.setMinimum(10 ** -6)
        doubleSpinBox_learning_rate.setSingleStep(10 ** -6)
        formLayout.addRow('learning_rate:', doubleSpinBox_learning_rate)

    if currentItemText == 'GBDT':
        spinBox_n_estimators = QSpinBox()
        spinBox_n_estimators.setMaximum(1000)
        spinBox_n_estimators.setMinimum(50)
        spinBox_n_estimators.setSingleStep(50)
        formLayout.addRow('n_estimators:', spinBox_n_estimators)

        doubleSpinBox_subsample = QDoubleSpinBox()
        doubleSpinBox_subsample.setDecimals(1)
        doubleSpinBox_subsample.setMaximum(0.9)
        doubleSpinBox_subsample.setMinimum(0.5)
        doubleSpinBox_subsample.setSingleStep(0.1)
        formLayout.addRow('subsample:', doubleSpinBox_subsample)

        comboBox_loss = QComboBox()
        comboBox_loss.addItems(['deviance', 'exponential'])
        formLayout.addRow('loss:', comboBox_loss)

        spinBox_max_depth = QSpinBox()
        spinBox_max_depth.setMaximum(20)
        spinBox_max_depth.setMinimum(3)
        formLayout.addRow('max_depth:', spinBox_max_depth)

    if currentItemText == 'XGboost':
        spinBox_n_estimators = QSpinBox()
        spinBox_n_estimators.setMaximum(1000)
        spinBox_n_estimators.setMinimum(50)
        spinBox_n_estimators.setSingleStep(50)
        formLayout.addRow('n_estimators:', spinBox_n_estimators)

        doubleSpinBox_learning_rate = QDoubleSpinBox()
        doubleSpinBox_learning_rate.setDecimals(6)
        doubleSpinBox_learning_rate.setMaximum(10)
        doubleSpinBox_learning_rate.setMinimum(10 ** -6)
        doubleSpinBox_learning_rate.setSingleStep(10 ** -6)
        formLayout.addRow('learning_rate:', doubleSpinBox_learning_rate)

    if currentItemText == 'CNN':
        pass
        # comboBox_oob_score = QComboBox()
        # comboBox_oob_score.addItems(['True', 'False'])
        # formLayout.addRow('n_estimators:', comboBox_oob_score)

    if currentItemText == 'LSTM':
        pass
        # comboBox_oob_score = QComboBox()
        # comboBox_oob_score.addItems(['True', 'False'])
        # formLayout.addRow('n_estimators:', comboBox_oob_score)


class DealAction:
    def __init__(self):
        self.ui_Main = None
        self.ui_Length_Clipping = None
        self.ui_Format_File = None
        self.ui_CD_HIT = None
        self.ui_Feature_Extraction = None
        self.ui_EncodingMethod = None
        self.ui_SelectModel = None
        self.ui_Validation = None
        self.ui_Document = None
        self.ui_About = None

        self.thread_Import_Train_File = None
        self.thread_Import_Test_File = None
        self.thread_Import_Prediction_File = None
        self.thread_Import_Prediction_File = None
        self.thread_Import_Model = None
        self.thread_Import_Feature = None
        self.thread_Duplication = None
        self.thread_Length_Clipping = None
        self.thread_Save_Preparation = None
        self.thread_Format_File = None
        self.thread_CD_HIT = None
        self.thread_Save_Feature = None
        self.thread_Feature_Extraction = None
        self.thread_Start_Training = None

    # File
    def action_Import_Train_File(self):
        my_emit(signal.lineEdit_System_Tips, 'Selecting train file.')
        file_names_tuple = QFileDialog().getOpenFileNames(parent=self.ui_Main,
                                                          caption='Select Train Files',
                                                          directory='./',
                                                          filter="Fasta files (*.fasta);;Fasta files (*.fa)")
        if not file_names_tuple[0]:
            my_emit(signal.lineEdit_System_Tips, 'Select train file is canceled! No file selected!')
            return
        self.ui_Main.tabWidget_File.setCurrentIndex(0)
        self.ui_Main.textBrowser_Train_File.setText('')
        self.thread_Import_Train_File = Thread_Import_Train_File(file_names_tuple[0])
        self.thread_Import_Train_File.start()

    def action_Import_Test_File(self):
        my_emit(signal.lineEdit_System_Tips, 'Selecting test file.')
        file_names_tuple = QFileDialog().getOpenFileNames(parent=self.ui_Main,
                                                          caption='Select Test Files',
                                                          directory='./',
                                                          filter="Fasta files (*.fasta);;Fasta files (*.fa)")
        if not file_names_tuple[0]:
            my_emit(signal.lineEdit_System_Tips, 'Select test file is canceled! No file selected!')
            return
        self.ui_Main.tabWidget_File.setCurrentIndex(1)
        self.ui_Main.textBrowser_Test_File.setText('')
        self.thread_Import_Test_File = Thread_Import_Test_File(file_names_tuple[0])
        self.thread_Import_Test_File.start()

    def action_Import_Prediction_File(self):
        my_emit(signal.lineEdit_System_Tips, 'Selecting prediction file.')
        file_names_tuple = QFileDialog().getOpenFileNames(parent=self.ui_Main,
                                                          caption='Select Prediction Files',
                                                          directory='./',
                                                          filter="Fasta files (*.fasta);;Fasta files (*.fa)")
        if not file_names_tuple[0]:
            my_emit(signal.lineEdit_System_Tips, 'Select prediction file is canceled! No file selected!')
            return
        self.ui_Main.tabWidget_File.setCurrentIndex(2)
        self.ui_Main.textBrowser_Prediction_File.setText('')
        self.thread_Import_Prediction_File = Thread_Import_Prediction_File(file_names_tuple[0])
        self.thread_Import_Prediction_File.start()

    def action_Import_Preparation_File(self):
        my_emit(signal.lineEdit_System_Tips, 'Selecting preparation file.')
        file_names_tuple = QFileDialog().getOpenFileNames(parent=self.ui_Main,
                                                          caption='Select Preparation Files',
                                                          directory='./',
                                                          filter="Fasta files (*.fasta);;Fasta files (*.fa)")
        if not file_names_tuple[0]:
            my_emit(signal.lineEdit_System_Tips, 'Select preparation file is canceled! No file selected!')
            return
        self.ui_Main.tabWidget_File.setCurrentIndex(3)
        self.ui_Main.textBrowser_Preparation_File.setText('')
        self.thread_Import_Train_File = Thread_Import_Preparation_File(file_names_tuple[0])
        self.thread_Import_Train_File.start()

    def action_Import_Model(self):
        my_emit(signal.lineEdit_System_Tips, 'Selecting model file.')
        file_name_tuple = QFileDialog().getOpenFileName(parent=self.ui_Main,
                                                        caption='Select Model File',
                                                        directory='./',
                                                        filter="Pickle file (*.pickle);;H5 file (*.h5);;All files (*)")
        if not file_name_tuple[0]:
            my_emit(signal.lineEdit_System_Tips, 'Select model file is canceled! No file selected!')
            return
        self.thread_Import_Model = Thread_Import_Model(file_name_tuple[0])
        self.thread_Import_Model.start()

    def action_Import_Feature(self):
        my_emit(signal.lineEdit_System_Tips, 'Selecting feature file.')
        file_name_tuple = QFileDialog().getOpenFileName(parent=self.ui_Main,
                                                        caption='Select Feature File',
                                                        directory='./',
                                                        filter="Text file (*.txt);;All file (*)")
        if not file_name_tuple[0]:
            my_emit(signal.lineEdit_System_Tips, 'Select feature file is canceled! No file selected!')
            return
        self.thread_Import_Feature = Thread_Import_Feature(file_name_tuple[0])
        self.thread_Import_Feature.start()

    def action_Close_All_Files(self):
        self.ui_Main.textBrowser_Train_File.setText('')
        self.ui_Main.textBrowser_Test_File.setText('')
        self.ui_Main.textBrowser_Prediction_File.setText('')
        self.ui_Main.textBrowser_Preparation_File.setText('')
        shareInfo.DefaultMenuFile()
        my_emit(signal.lineEdit_System_Tips, 'Closed all files!')

    def action_Clear_Output(self):
        self.ui_Main.textBrowser_Message.setText('')
        my_emit(signal.lineEdit_System_Tips, 'Cleared message display!')

    def action_Exit(self):
        self.ui_Main.close()

    # Preparation
    def action_Duplication(self):
        if not self.CheckExistence(shareInfo.menuFile.preparationFileData, 'Preparation file isn\'t imported!'):
            return
        self.action_Clear_Output()
        self.thread_Duplication = Thread_Duplication()
        self.thread_Duplication.start()

    def action_Length_Clipping(self):
        if not self.CheckExistence(shareInfo.menuFile.preparationFileData, 'Preparation file isn\'t imported!'):
            return
        self.ui_Length_Clipping = uic.loadUi("ui/LengthClipping.ui")
        self.ui_Length_Clipping.setWindowModality(Qt.ApplicationModal)  # Set as modal window
        self.ui_Length_Clipping.buttonBox.accepted.connect(self.ui_Length_Clipping_buttonBoxAccepted)
        self.ui_Length_Clipping.buttonBox.rejected.connect(self.ui_Length_Clipping_buttonBoxRejected)
        self.ui_Length_Clipping.show()
        my_emit(signal.lineEdit_System_Tips, 'Length clipping setting.')

    def action_CD_HIT(self):
        if not self.CheckExistence(shareInfo.menuFile.preparationFileData, 'Preparation file isn\'t imported!'):
            return

        self.ui_CD_HIT = uic.loadUi("ui/CD_HIT.ui")
        self.ui_CD_HIT.setWindowModality(Qt.ApplicationModal)  # Set as modal window
        self.ui_CD_HIT.buttonBox.accepted.connect(self.ui_CD_HIT_buttonBoxAccepted)
        self.ui_CD_HIT.buttonBox.rejected.connect(self.ui_CD_HIT_buttonBoxRejected)
        self.ui_CD_HIT.show()
        my_emit(signal.lineEdit_System_Tips, 'CD-HIT setting.')

    def action_Format_File(self):
        if not self.CheckExistence(shareInfo.menuFile.preparationFileData, 'Preparation file isn\'t imported!'):
            return
        self.ui_Format_File = uic.loadUi("ui/FormatFile.ui")
        self.ui_Format_File.setWindowModality(Qt.ApplicationModal)  # Set as modal window
        self.ui_Format_File.buttonBox.accepted.connect(self.ui_Format_File_buttonBoxAccepted)
        self.ui_Format_File.buttonBox.rejected.connect(self.ui_Format_File_buttonBoxRejected)
        self.ui_Format_File.show()
        my_emit(signal.lineEdit_System_Tips, 'Format setting.')

    def action_Save_Preparation(self):
        if not self.CheckExistence(shareInfo.menuPreparation.listResult, 'Preparation result isn\'t exist!'):
            return
        my_emit(signal.lineEdit_System_Tips, 'Selecting preparation save file.')
        fileNameTuple = QFileDialog().getSaveFileName(parent=self.ui_Main,
                                                      caption='Save preparation result',
                                                      directory='./',
                                                      filter="Fasta files (*.fasta);;Fasta files (*.fa)")
        if not fileNameTuple[0]:
            my_emit(signal.lineEdit_System_Tips, 'No save path selected!')
            return
        self.thread_Save_Preparation = Thread_Save_Preparation(fileNameTuple[0])
        self.thread_Save_Preparation.start()

    def ui_Length_Clipping_buttonBoxAccepted(self):
        self.action_Clear_Output()
        minimumLength = self.ui_Length_Clipping.spinBox_Minimum_Length.text()
        maximumLength = self.ui_Length_Clipping.spinBox_Maximum_Length.text()
        self.ui_Length_Clipping.close()
        self.thread_Length_Clipping = Thread_Length_Clipping(int(minimumLength), int(maximumLength))
        self.thread_Length_Clipping.start()

    def ui_Length_Clipping_buttonBoxRejected(self):
        self.ui_Length_Clipping.close()
        my_emit(signal.lineEdit_System_Tips, 'Length clipping setting is canceled!')

    def ui_CD_HIT_buttonBoxAccepted(self):
        self.action_Clear_Output()
        self.ui_CD_HIT.close()
        self.thread_CD_HIT = Thread_CD_HIT()
        self.thread_CD_HIT.start()

    def ui_CD_HIT_buttonBoxRejected(self):
        self.ui_CD_HIT.close()
        my_emit(signal.lineEdit_System_Tips, 'CD-HIT setting is canceled!')

    def ui_Format_File_buttonBoxAccepted(self):
        self.action_Clear_Output()
        label = self.ui_Format_File.spinBox_Label.text()
        dataType = self.ui_Format_File.comboBox_Type.currentText()
        self.ui_Format_File.close()
        self.thread_Format_File = Thread_Format_File(label, dataType)
        self.thread_Format_File.start()

    def ui_Format_File_buttonBoxRejected(self):
        self.ui_Format_File.close()
        my_emit(signal.lineEdit_System_Tips, 'Format file setting is canceled!')

    # Features
    def action_Feature_Extraction(self):
        if not self.CheckExistence(shareInfo.menuFile.trainFileData, 'Train file isn\'t imported!'):
            return
        self.ui_Feature_Extraction = uic.loadUi("ui/FeatureExtraction.ui")
        self.ui_Feature_Extraction.setWindowModality(Qt.ApplicationModal)  # Set as modal window
        self.ui_Feature_Extraction.comboBox_Select_Feature.currentIndexChanged.connect(
            self.ui_Feature_Extraction_comboBox_Select_Feature_currentIndexChanged)
        self.ui_Feature_Extraction.buttonBox.accepted.connect(self.ui_Feature_Extraction_buttonBoxAccepted)
        self.ui_Feature_Extraction.buttonBox.rejected.connect(self.ui_Feature_Extraction_buttonBoxRejected)
        self.ui_Feature_Extraction.show()
        my_emit(signal.lineEdit_System_Tips, 'Feature extraction setting.')

    def action_Save_Feature(self):
        if not self.CheckExistence(shareInfo.menuFeature.ndarrayResult, 'Feature result isn\'t exist!'):
            return
        my_emit(signal.lineEdit_System_Tips, 'Selecting feature save file.')
        fileNameTuple = QFileDialog().getSaveFileName(self.ui_Main, 'Save preparation result', './',
                                                      "Csv files (*.csv);;Text files (*.txt);;All files (*)")
        if not fileNameTuple[0]:
            my_emit(signal.lineEdit_System_Tips, 'No save path selected!')
            return
        self.thread_Save_Feature = Thread_Save_Feature(fileNameTuple[0])
        self.thread_Save_Feature.start()

    def ui_Feature_Extraction_comboBox_Select_Feature_currentIndexChanged(self):
        SelectFeature(self.ui_Feature_Extraction.comboBox_Select_Feature,
                      self.ui_Feature_Extraction.scrollAreaWidgetContents_Feature_Params.layout())

    def ui_Feature_Extraction_buttonBoxAccepted(self):
        self.action_Clear_Output()
        shareInfo.menuFeature.featureName = self.ui_Feature_Extraction.comboBox_Select_Feature.currentText()
        shareInfo.menuFeature.featureParams = GetLayoutItemValue(
            self.ui_Feature_Extraction.scrollAreaWidgetContents_Feature_Params.layout())
        self.ui_Feature_Extraction.close()
        self.thread_Feature_Extraction = Thread_Feature_Extraction(shareInfo.menuFeature.featureName,
                                                                   shareInfo.menuFeature.featureParams)
        self.thread_Feature_Extraction.start()

    def ui_Feature_Extraction_buttonBoxRejected(self):
        self.ui_Feature_Extraction.close()
        my_emit(signal.lineEdit_System_Tips, 'Feature extraction setting is canceled!')

    # Model
    def action_Start_Training(self):
        if not self.CheckExistence(shareInfo.menuFile.trainFileData, 'Train file isn\'t imported!'):
            return
        if not self.CheckExistence(shareInfo.menuFile.testFileData, 'Test file isn\'t imported!'):
            return
        if not self.CheckExistence(shareInfo.menuModel.encodingName, 'Encoding  isn\'t selected!'):
            return
        if not self.CheckExistence(shareInfo.menuModel.modelName, 'Model isn\'t selected!'):
            return
        self.action_Clear_Output()
        my_emit(signal.textBrowser_Message,
                '====Start training====\n'
                'Encoding method: {},\n'
                'Encoding params: {},\n'
                'Model name: {},\n'
                'Model params: {},\n'
                'Validation: {} fold.\n'
                .format(shareInfo.menuModel.encodingName, shareInfo.menuModel.encodingParams,
                        shareInfo.menuModel.modelName, shareInfo.menuModel.modelParams,
                        shareInfo.menuModel.validation))
        self.thread_Start_Training = Thread_Start_Training()
        self.thread_Start_Training.start()

    def action_Encoding_Method(self):
        self.ui_EncodingMethod = uic.loadUi("ui/EncodingMethod.ui")
        self.ui_EncodingMethod.setWindowModality(Qt.ApplicationModal)  # Set as modal window
        self.ui_EncodingMethod.comboBox_Select_Encoding.currentIndexChanged.connect(
            self.ui_EncodingMethod_comboBox_Select_Encoding_currentIndexChanged)
        self.ui_EncodingMethod.buttonBox.accepted.connect(self.ui_EncodingMethod_buttonBoxAccepted)
        self.ui_EncodingMethod.buttonBox.rejected.connect(self.ui_EncodingMethod_buttonBoxRejected)
        self.ui_EncodingMethod.show()
        my_emit(signal.lineEdit_System_Tips, 'Encoding method setting.')

    def ui_EncodingMethod_comboBox_Select_Encoding_currentIndexChanged(self):
        SelectFeature(self.ui_EncodingMethod.comboBox_Select_Encoding,
                      self.ui_EncodingMethod.scrollAreaWidgetContents_Encoding_Params.layout())

    def ui_EncodingMethod_buttonBoxAccepted(self):
        shareInfo.menuModel.encodingName = self.ui_EncodingMethod.comboBox_Select_Encoding.currentText()
        shareInfo.menuModel.encodingParams = GetLayoutItemValue(
            self.ui_EncodingMethod.scrollAreaWidgetContents_Encoding_Params.layout())
        self.ui_EncodingMethod.close()
        my_emit(signal.lineEdit_System_Tips, 'Encoding setting is OK!')

    def ui_EncodingMethod_buttonBoxRejected(self):
        self.ui_EncodingMethod.close()
        my_emit(signal.lineEdit_System_Tips, 'Encoding method setting is canceled!')

    def action_Select_Model(self):
        self.ui_SelectModel = uic.loadUi("ui/SelectModel.ui")
        self.ui_SelectModel.setWindowModality(Qt.ApplicationModal)  # Set as modal window
        self.ui_SelectModel.comboBox_Select_Model.currentIndexChanged.connect(
            self.ui_SelectModel_comboBox_Select_Model_currentIndexChanged)
        self.ui_SelectModel.buttonBox.accepted.connect(self.ui_SelectModel_buttonBoxAccepted)
        self.ui_SelectModel.buttonBox.rejected.connect(self.ui_SelectModel_buttonBoxRejected)
        self.ui_SelectModel.show()
        my_emit(signal.lineEdit_System_Tips, 'Model setting.')

    def ui_SelectModel_comboBox_Select_Model_currentIndexChanged(self):
        SelectModel(self.ui_SelectModel.comboBox_Select_Model,
                    self.ui_SelectModel.scrollAreaWidgetContents_Model_Params.layout())

    def ui_SelectModel_buttonBoxAccepted(self):
        shareInfo.menuModel.modelName = self.ui_SelectModel.comboBox_Select_Model.currentText()
        shareInfo.menuModel.modelParams = GetLayoutItemValue(
            self.ui_SelectModel.scrollAreaWidgetContents_Model_Params.layout())
        self.ui_SelectModel.close()
        my_emit(signal.lineEdit_System_Tips, 'Model setting is OK!')

    def ui_SelectModel_buttonBoxRejected(self):
        self.ui_SelectModel.close()
        my_emit(signal.lineEdit_System_Tips, 'Model setting is canceled!')

    def action_Validation(self):
        self.ui_Validation = uic.loadUi("ui/CrossValidation.ui")
        self.ui_Validation.setWindowModality(Qt.ApplicationModal)  # Set as modal window
        self.ui_Validation.buttonBox.accepted.connect(self.ui_Validation_buttonBoxAccepted)
        self.ui_Validation.buttonBox.rejected.connect(self.ui_Validation_buttonBoxRejected)
        self.ui_Validation.show()
        my_emit(signal.lineEdit_System_Tips, 'Cross validation setting.')

    def ui_Validation_buttonBoxAccepted(self):
        shareInfo.menuModel.validation = int(self.ui_Validation.spinBox.text())
        self.ui_Validation.close()
        my_emit(signal.lineEdit_System_Tips, 'Cross Validation setting is OK!')

    def ui_Validation_buttonBoxRejected(self):
        self.ui_Validation.close()
        my_emit(signal.lineEdit_System_Tips, 'Cross Validation setting is canceled!')

    def action_Save_Model(self):
        if not self.CheckExistence(shareInfo.menuModel.trainedModel, 'Model isn\'t exist!'):
            return
        my_emit(signal.lineEdit_System_Tips, 'Selecting model save file.')
        fileNameTuple = QFileDialog().getSaveFileName(self.ui_Main, 'Save model', './',
                                                      "Pickle files (*.pickle);;H5 files (*.h5);;All files (*)")
        if not fileNameTuple[0]:
            my_emit(signal.lineEdit_System_Tips, 'No save path selected!')
            return
        self.thread_Save_Model = Thread_Save_Model(fileNameTuple[0])
        self.thread_Save_Model.start()

    def action_Save_All_Metrics(self):
        self.ui_Main.textBrowser_Message.setText('action_Save_All_Metrics')

    def action_Save_Classification_Report(self):
        self.ui_Main.textBrowser_Message.setText('action_Save_Classification_Report')

    def action_Save_Confusion_Matrix(self):
        self.ui_Main.textBrowser_Message.setText('action_Save_Confusion_Matrix')

    def action_Save_Train_Loss(self):
        self.ui_Main.textBrowser_Message.setText('action_Save_Train_Loss')

    def action_Save_ROC_Curve(self):
        self.ui_Main.textBrowser_Message.setText('action_Save_ROC_Curve')

    def action_Save_PRC_Curve(self):
        self.ui_Main.textBrowser_Message.setText('action_Save_PRC_Curve')

    # Prediction
    def action_Start_Prediction(self):
        if not self.CheckExistence(shareInfo.menuFile.model, 'Model file isn\'t imported!'):
            return
        if not self.CheckExistence(shareInfo.menuFile.predictionFileData, 'Prediction file isn\'t imported!'):
            return
        self.thread_Start_Prediction = Thread_Start_Prediction()
        self.thread_Start_Prediction.start()

    def action_Save_Prediction_Result(self):
        if not self.CheckExistence(shareInfo.menuPrediction.listPredictionResult, 'Result isn\'t exist!'):
            return
        my_emit(signal.lineEdit_System_Tips, 'Selecting prediction result save file.')
        fileNameTuple = QFileDialog().getSaveFileName(self.ui_Main, 'Save prediction result file', './',
                                                      "Csv files (*.csv);;Text files (*.txt);;All files (*)")
        if not fileNameTuple[0]:
            my_emit(signal.lineEdit_System_Tips, 'No save path selected!')
            return
        self.thread_Save_Prediction = Thread_Save_Prediction(fileNameTuple[0])
        self.thread_Save_Prediction.start()

    # Visualization
    def action_Dimension_Reduction(self):
        self.ui_Main.textBrowser_Message.setText('action_Dimension_Reduction')

    def action_Feature_Ranking(self):
        self.ui_Main.textBrowser_Message.setText('action_Feature_Ranking')

    def action_Network_Visualization(self):
        self.ui_Main.textBrowser_Message.setText('action_Network_Visualization')

    # Help
    def action_Document(self):
        self.ui_Document = uic.loadUi("ui/About.ui")
        self.ui_Document.show()
        import webbrowser
        url = 'https://www.w3schools.com/'
        webbrowser.open_new_tab(url)
        my_emit(signal.lineEdit_System_Tips, 'Opened document!')

    def action_About(self):
        self.ui_About = uic.loadUi("ui/About.ui")
        self.ui_About.show()
        my_emit(signal.lineEdit_System_Tips, 'Show about information!')

    # Stop
    def action_Stop(self):
        self.ui_Main.textBrowser_Message.setText('stop')

    # 交互区域
    def comboBox_Select_Encoding(self):
        SelectFeature(self.ui_Main.comboBox_Select_Encoding,
                      self.ui_Main.scrollAreaWidgetContents_Encoding_Params.layout())
        shareInfo.menuModel.encodingName = self.ui_Main.comboBox_Select_Encoding.currentText()
        shareInfo.menuModel.encodingParams = GetLayoutItemValue(
            self.ui_Main.scrollAreaWidgetContents_Encoding_Params.layout())
        my_emit(signal.lineEdit_System_Tips, 'Encoding setting is OK!')

    def comboBox_Select_Model(self):
        SelectModel(self.ui_Main.comboBox_Select_Model,
                    self.ui_Main.scrollAreaWidgetContents_Model_Params.layout())
        shareInfo.menuModel.modelName = self.ui_Main.comboBox_Select_Model.currentText()
        shareInfo.menuModel.modelParams = GetLayoutItemValue(
            self.ui_Main.scrollAreaWidgetContents_Model_Params.layout())
        my_emit(signal.lineEdit_System_Tips, 'Model setting is OK!')

    ''' plot function'''

    # System Tips
    def set_lineEdit_System_Tips(self, text):
        self.ui_Main.lineEdit_System_Tips.setText(text)

    # Progress bar
    def set_progressBar(self, value):
        self.ui_Main.progressBar.setValue(value)

    # Params
    def set_widget_Params_Encoding_Model_CrossValidation_SuperParams(self, text1, text2, text3, text4):
        if text1:
            self.ui_Main.lineEdit_Encoding.setText(text1)
        if text2:
            self.ui_Main.lineEdit_Model.setText(text2)
        if text3:
            self.ui_Main.lineEdit_Cross_Validation.setText(text3)
        if text4:
            self.ui_Main.textBrowser_Super_Params.setText(text4)

    # File Display
    def set_tabWidget_File_TrainFile_TestFile_PreparationFile(self, text1, text2, text3, text4):
        if text1:
            self.ui_Main.textBrowser_Train_File.append(text1)
        if text2:
            self.ui_Main.textBrowser_Test_File.append(text2)
        if text3:
            self.ui_Main.textBrowser_Prediction_File.append(text3)
        if text4:
            self.ui_Main.textBrowser_Preparation_File.append(text4)

    # Metrics Display
    def set_tabWidget_Metrics_ROCCurve_ConfusionMatrix_ClassificationReport(self, roc, cm, cr):
        if roc:
            DeleteLayoutItem(self.ui_Main.tab_ROC_Curve.layout())
            canvasROC = MyCanvas()
            canvasROC.figure = roc.figure
            self.ui_Main.tab_ROC_Curve.layout().addWidget(canvasROC)
        if cm:
            DeleteLayoutItem(self.ui_Main.tab_Confusion_Matrix.layout())
            canvasCM = MyCanvas()
            canvasCM.figure = cm.figure
            self.ui_Main.tab_Confusion_Matrix.layout().addWidget(canvasCM)
        if cr:
            for row in range(2):
                for col, value in enumerate(cr.values()):
                    tableItem = MyQTableWidgetItem('{:.2f}%'.format(value[row] * 100))
                    tableItem.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
                    self.ui_Main.tableWidget_Classification_Report.setItem(row, col, tableItem)
            for col, value in enumerate(cr.values()):
                tableItem = MyQTableWidgetItem('{:.2f}%'.format(sum(value) / 2 * 100))
                tableItem.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
                self.ui_Main.tableWidget_Classification_Report.setItem(2, col, tableItem)

    # Message Display
    def set_textBrowser_Message(self, text):
        self.ui_Main.textBrowser_Message.append(text)

    # Features Display
    def set_graphicsView_Feature(self, value):
        pass

    '''General methods'''

    # Dialog
    def WarningDialog(self, title, content):
        QMessageBox.warning(self.ui_Main, title, content, QMessageBox.Ok)

    def CheckExistence(self, target, tipContent):
        if target is None:
            my_emit(signal.lineEdit_System_Tips, tipContent)
            self.WarningDialog('Warning', tipContent)
            return False
        return True
