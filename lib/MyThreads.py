import os
import pickle

import pandas as pd
from PyQt5.QtCore import QThread
from lib.DataProcessing import *
from lib.ShareInfo import *
from lib.MySignal import *
from lib.FeatureExtraction import *
from lib.Visualization import *
from lib.ModelDefinition import *

'''File'''


class Thread_Import_Train_File(QThread):
    def __init__(self, listFilePath):
        super(Thread_Import_Train_File, self).__init__()
        self.listFilePath = listFilePath

    def run(self):
        fasta_data = []
        for filePath in self.listFilePath:
            my_emit(signal.lineEdit_System_Tips, 'Loading train file: ' + filePath)
            tmp_data = LoadData(filePath, 0)
            fasta_data += tmp_data
        shareInfo.menuFile.trainFileData = fasta_data
        my_emit(signal.lineEdit_System_Tips, 'Loaded file!')


class Thread_Import_Test_File(QThread):
    def __init__(self, listFilePath):
        super(Thread_Import_Test_File, self).__init__()
        self.listFilePath = listFilePath

    def run(self):
        fasta_data = []
        for filePath in self.listFilePath:
            my_emit(signal.lineEdit_System_Tips, 'Loading test file: ' + filePath)
            tmp_data = LoadData(filePath, 1)
            fasta_data += tmp_data
        shareInfo.menuFile.testFileData = fasta_data
        my_emit(signal.lineEdit_System_Tips, 'Loaded file!')


class Thread_Import_Prediction_File(QThread):
    def __init__(self, listFilePath):
        super(Thread_Import_Prediction_File, self).__init__()
        self.listFilePath = listFilePath

    def run(self):
        fasta_data = []
        for filePath in self.listFilePath:
            my_emit(signal.lineEdit_System_Tips, 'Loading prediction file: ' + filePath)
            tmp_data = LoadData(filePath, 2)
            fasta_data += tmp_data
        shareInfo.menuFile.predictionFileData = fasta_data
        my_emit(signal.lineEdit_System_Tips, 'Loaded file!')


class Thread_Import_Preparation_File(QThread):
    def __init__(self, listFilePath):
        super(Thread_Import_Preparation_File, self).__init__()
        self.listFilePath = listFilePath

    def run(self):
        fasta_data = []
        for filePath in self.listFilePath:
            my_emit(signal.lineEdit_System_Tips, 'Loading preparation file: ' + filePath)
            tmp_data = LoadData(filePath, 3)
            fasta_data += tmp_data
        shareInfo.menuFile.preparationFileData = fasta_data
        my_emit(signal.lineEdit_System_Tips, 'Loaded file!')


class Thread_Import_Model(QThread):
    def __init__(self, modelPath):
        super(Thread_Import_Model, self).__init__()
        self.modelPath = modelPath

    def run(self):
        my_emit(signal.lineEdit_System_Tips, 'Loading model.')
        my_emit(signal.progressBar, 0)
        with open(self.modelPath, 'rb') as f:
            shareInfo.menuFile.model = pickle.load(f)
        my_emit(signal.progressBar, 100)
        my_emit(signal.lineEdit_System_Tips, 'Model file has loaded!')


class Thread_Import_Feature(QThread):
    def __init__(self, featurePath):
        super(Thread_Import_Feature, self).__init__()
        self.featurePath = featurePath

    def run(self):
        my_emit(signal.textBrowser_Message, 'Import feature isn\'t implemented')
        my_emit(signal.lineEdit_System_Tips, 'Import feature isn\'t implemented')


'''Preparation'''


class Thread_Duplication(QThread):
    def __init__(self):
        super(Thread_Duplication, self).__init__()

    def run(self):
        my_emit(signal.lineEdit_System_Tips, 'Duplicating preparation file.')
        setSequences = set()
        shareInfo.menuPreparation.listResult = []
        preparationLength = len(shareInfo.menuFile.preparationFileData)
        one_percent = max(1, preparationLength // 100)
        preparationIndex, newResultIndex = 0, 0
        for item in shareInfo.menuFile.preparationFileData:
            if item[1] not in setSequences:
                shareInfo.menuPreparation.listResult.append('{}\n{}'.format(item[0], item[1]))
                newResultIndex -= 1
            setSequences.add(item[1])
            preparationIndex += 1
            if preparationIndex % one_percent == 0 or preparationIndex == preparationLength:
                my_emit(signal.progressBar, 100 * preparationIndex / preparationLength)
                if newResultIndex < 0:
                    my_emit(signal.textBrowser_Message,
                            '\n'.join(shareInfo.menuPreparation.listResult[newResultIndex:]))
                    newResultIndex = 0
        my_emit(signal.textBrowser_Message, '\n' + '=' * 10 + '\nAfter processing, left data number is: {}'.format(
            len(shareInfo.menuPreparation.listResult)))
        my_emit(signal.lineEdit_System_Tips, 'Duplication is OK!')


class Thread_Length_Clipping(QThread):
    def __init__(self, minimumLength, maximumLength):
        super(Thread_Length_Clipping, self).__init__()
        self.minimumLength = minimumLength
        self.maximumLength = maximumLength

    def run(self):
        my_emit(signal.lineEdit_System_Tips, 'Length clipping preparation file.')
        shareInfo.menuPreparation.listResult = []
        preparationLength = len(shareInfo.menuFile.preparationFileData)
        one_percent = max(1, preparationLength // 100)
        preparationIndex, newResultIndex = 0, 0
        for item in shareInfo.menuFile.preparationFileData:
            if self.minimumLength <= len(item[1]) <= self.maximumLength:
                shareInfo.menuPreparation.listResult.append('{}\n{}'.format(item[0], item[1]))
                newResultIndex -= 1
            preparationIndex += 1
            if preparationIndex % one_percent == 0 or preparationIndex == preparationLength:
                my_emit(signal.progressBar, 100 * preparationIndex / preparationLength)
                if newResultIndex < 0:
                    my_emit(signal.textBrowser_Message,
                            '\n'.join(shareInfo.menuPreparation.listResult[newResultIndex:]))
                    newResultIndex = 0
        my_emit(signal.textBrowser_Message, '\n' + '=' * 10 + '\nAfter processing, left data number is: {}'.format(
            len(shareInfo.menuPreparation.listResult)))
        my_emit(signal.lineEdit_System_Tips, 'Length clipping is OK!')


class Thread_CD_HIT(QThread):
    def __init__(self):
        super(Thread_CD_HIT, self).__init__()

    def run(self):
        my_emit(signal.lineEdit_System_Tips, 'Not implemented!')
        my_emit(signal.textBrowser_Message, 'Not implemented!')


class Thread_Format_File(QThread):
    def __init__(self, label, dataType):
        super(Thread_Format_File, self).__init__()
        self.label = label
        self.dataType = dataType

    def run(self):
        my_emit(signal.lineEdit_System_Tips, 'Format preparation file.')
        shareInfo.menuPreparation.listResult = []
        preparationLength = len(shareInfo.menuFile.preparationFileData)
        one_percent = max(1, preparationLength // 100)
        preparationIndex, newResultIndex = 0, 0
        for item in shareInfo.menuFile.preparationFileData:
            shareInfo.menuPreparation.listResult.append(
                '{}|{}|{}\n{}'.format(item[0], self.label, self.dataType, item[1]))
            newResultIndex -= 1
            preparationIndex += 1
            if preparationIndex % one_percent == 0 or preparationIndex == preparationLength:
                my_emit(signal.progressBar, 100 * preparationIndex / preparationLength)
                if newResultIndex < 0:
                    my_emit(signal.textBrowser_Message,
                            '\n'.join(shareInfo.menuPreparation.listResult[newResultIndex:]))
                    newResultIndex = 0
        my_emit(signal.textBrowser_Message, '\n' + '=' * 10 + '\nAfter processing, left data number is: {}'.format(
            len(shareInfo.menuPreparation.listResult)))
        my_emit(signal.lineEdit_System_Tips, 'Format file is OK!')


class Thread_Save_Preparation(QThread):
    def __init__(self, fileName):
        super(Thread_Save_Preparation, self).__init__()
        self.fileName = fileName

    def run(self):
        my_emit(signal.lineEdit_System_Tips, 'Saving preparation result.')
        resultLength = len(shareInfo.menuPreparation.listResult)
        one_percent = max(1, resultLength // 100)
        preparationIndex = 0
        if os.path.exists(self.fileName):
            os.remove(self.fileName)
        with open(self.fileName, 'w+') as f:
            for item in shareInfo.menuPreparation.listResult:
                f.write(str(item) + '\n')
                preparationIndex += 1
                if preparationIndex % one_percent == 0 or preparationIndex == resultLength:
                    my_emit(signal.progressBar, 100 * preparationIndex / resultLength)
        my_emit(signal.lineEdit_System_Tips, 'Preparation file has saved!')


class Thread_Feature_Extraction(QThread):
    def __init__(self, featureName, dictParams):
        super(Thread_Feature_Extraction, self).__init__()
        self.featureName = featureName
        self.dictParams = dictParams

    def run(self):
        listSequences = [i[1] for i in shareInfo.menuFile.trainFileData]
        encodingFunc = globals()[self.featureName]
        encoded_data = encodingFunc(listSequences, **self.dictParams)
        shareInfo.menuFeature.ndarrayResult = encoded_data
        my_emit(signal.textBrowser_Message,
                'Feature name: {}\n'.format(self.featureName) +
                'Params: {}\n'.format(
                    '; '.join([k + '=' + str(v) for k, v in self.dictParams.items()]) if self.dictParams else 'None') +
                'Encoded data\'s shape: {}'.format(encoded_data.shape))


class Thread_Save_Feature(QThread):
    def __init__(self, fileName):
        super(Thread_Save_Feature, self).__init__()
        self.fileName = fileName

    def run(self):
        my_emit(signal.lineEdit_System_Tips, 'Saving feature result.')
        resultLength = len(shareInfo.menuFeature.ndarrayResult)
        one_percent = max(1, resultLength // 100)
        featureIndex = 0
        if os.path.exists(self.fileName):
            os.remove(self.fileName)
        with open(self.fileName, 'w+') as f:
            for rowItem in shareInfo.menuFeature.ndarrayResult:
                f.write(','.join([str(colItem).replace('\n', '') for colItem in rowItem]) + '\n')
                featureIndex += 1
                if featureIndex % one_percent == 0 or featureIndex == resultLength:
                    my_emit(signal.progressBar, 100 * featureIndex / resultLength)
        my_emit(signal.lineEdit_System_Tips, 'Feature file has saved!')


class Thread_Start_Training(QThread):
    def __init__(self):
        super(Thread_Start_Training, self).__init__()

    def run(self):
        StartTrain(shareInfo.menuFile.trainFileData, shareInfo.menuFile.testFileData,
                   shareInfo.menuModel.encodingName, shareInfo.menuModel.encodingParams, shareInfo.menuModel.modelName,
                   shareInfo.menuModel.modelParams, shareInfo.menuModel.validation)


class Thread_Save_Model(QThread):
    def __init__(self, fileName):
        super(Thread_Save_Model, self).__init__()
        self.fileName = fileName

    def run(self):
        my_emit(signal.lineEdit_System_Tips, 'Saving model.')
        if os.path.exists(self.fileName):
            os.remove(self.fileName)
        my_emit(signal.progressBar, 0)
        with open(self.fileName, 'wb') as f:
            pickle.dump(shareInfo.menuModel.trainedModel, f)
        my_emit(signal.progressBar, 100)
        my_emit(signal.lineEdit_System_Tips, 'Model file has saved!')


class Thread_Start_Prediction(QThread):
    def __init__(self):
        super(Thread_Start_Prediction, self).__init__()

    def run(self):
        my_emit(signal.lineEdit_System_Tips, 'Start prediction...')
        StartPrediction(shareInfo.menuFile.predictionFileData, shareInfo.menuFile.model,
                        shareInfo.menuModel.encodingName, shareInfo.menuModel.encodingParams)
        my_emit(signal.lineEdit_System_Tips, 'Prediction has finished.')


class Thread_Save_Prediction(QThread):
    def __init__(self, fileName):
        super(Thread_Save_Prediction, self).__init__()
        self.fileName = fileName

    def run(self):
        my_emit(signal.lineEdit_System_Tips, 'Saving prediction result.')
        if os.path.exists(self.fileName):
            os.remove(self.fileName)
        my_emit(signal.progressBar, 0)

        dictResult = dict(zip(['\n'.join(i) for i in shareInfo.menuFile.predictionFileData],
                              [str(i) for i in shareInfo.menuPrediction.listPredictionResult]))
        dataframe = pd.DataFrame({'Sequence': dictResult.keys(), 'Prediction': dictResult.values()})
        dataframe.to_csv(self.fileName, index=False, sep=',')
        my_emit(signal.progressBar, 100)
        my_emit(signal.lineEdit_System_Tips, 'Prediciton result file has saved!')


class Thread_Clear_Output(QThread):
    def __init__(self):
        super(Thread_Clear_Output, self).__init__()

    def run(self):
        pass


class Thread_Clear_Output(QThread):
    def __init__(self):
        super(Thread_Clear_Output, self).__init__()

    def run(self):
        pass


class Thread_Clear_Output(QThread):
    def __init__(self):
        super(Thread_Clear_Output, self).__init__()

    def run(self):
        pass


class Thread_Clear_Output(QThread):
    def __init__(self):
        super(Thread_Clear_Output, self).__init__()

    def run(self):
        pass


class Thread_Clear_Output(QThread):
    def __init__(self):
        super(Thread_Clear_Output, self).__init__()

    def run(self):
        pass


class Thread_Clear_Output(QThread):
    def __init__(self):
        super(Thread_Clear_Output, self).__init__()

    def run(self):
        pass


class Thread_Clear_Output(QThread):
    def __init__(self):
        super(Thread_Clear_Output, self).__init__()

    def run(self):
        pass


class Thread_Clear_Output(QThread):
    def __init__(self):
        super(Thread_Clear_Output, self).__init__()

    def run(self):
        pass


class Thread_Clear_Output(QThread):
    def __init__(self):
        super(Thread_Clear_Output, self).__init__()

    def run(self):
        pass


class Thread_Clear_Output(QThread):
    def __init__(self):
        super(Thread_Clear_Output, self).__init__()

    def run(self):
        pass


class Thread_Clear_Output(QThread):
    def __init__(self):
        super(Thread_Clear_Output, self).__init__()

    def run(self):
        pass


class Thread_Clear_Output(QThread):
    def __init__(self):
        super(Thread_Clear_Output, self).__init__()

    def run(self):
        pass


class Thread_Clear_Output(QThread):
    def __init__(self):
        super(Thread_Clear_Output, self).__init__()

    def run(self):
        pass


class Thread_Clear_Output(QThread):
    def __init__(self):
        super(Thread_Clear_Output, self).__init__()

    def run(self):
        pass


class Thread_Clear_Output(QThread):
    def __init__(self):
        super(Thread_Clear_Output, self).__init__()

    def run(self):
        pass


class Thread_Clear_Output(QThread):
    def __init__(self):
        super(Thread_Clear_Output, self).__init__()

    def run(self):
        pass


class Thread_Clear_Output(QThread):
    def __init__(self):
        super(Thread_Clear_Output, self).__init__()

    def run(self):
        pass


class Thread_Clear_Output(QThread):
    def __init__(self):
        super(Thread_Clear_Output, self).__init__()

    def run(self):
        pass


class Thread_Clear_Output(QThread):
    def __init__(self):
        super(Thread_Clear_Output, self).__init__()

    def run(self):
        pass


class Thread_Clear_Output(QThread):
    def __init__(self):
        super(Thread_Clear_Output, self).__init__()

    def run(self):
        pass
