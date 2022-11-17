import os
import random
import time
import math
from PySide2.QtCore import QThread
from PySide2.QtWidgets import QGraphicsView
from matplotlib import pyplot as plt
import pyqtgraph as pg
from matplotlib.backends.backend_template import FigureCanvas

from lib.DataProcessing import *
from lib.ShareInfo import *
from lib.MySignal import *
from lib.FeatureExtraction import *
from lib.Visualization import *
from lib.ModelDefinition import *
import numpy as np
import pandas as pd

'''File'''


class Thread_Import_Train_File(QThread):
    def __init__(self, listFilePath):
        super(Thread_Import_Train_File, self).__init__()
        self.listFilePath = listFilePath

    def run(self):
        fasta_data = []
        for filePath in self.listFilePath:
            my_emit(signal.lineEdit_System_Tips, 'Loading train file: ' + filePath)
            data = LoadData(filePath, 0)
            fasta_data += data
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
            data = LoadData(filePath, 1)
            fasta_data += data
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
            data = LoadData(filePath, 2)
            fasta_data += data
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
            data = LoadData(filePath, 3)
            fasta_data += data
        shareInfo.menuFile.preparationFileData = fasta_data
        my_emit(signal.lineEdit_System_Tips, 'Loaded file!')


class Thread_Import_Model(QThread):
    def __init__(self, modelPath):
        super(Thread_Import_Model, self).__init__()
        self.modelPath = modelPath

    def run(self):
        my_emit(signal.textBrowser_Message, 'Import model isn\'t implemented')
        my_emit(signal.lineEdit_System_Tips, 'Import model isn\'t implemented')


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
        encoded_data = None
        listSequences = [i[1] for i in shareInfo.menuFile.trainFileData]
        if self.featureName == 'AAC':
            encoded_data = AAC(listSequences)
        if self.featureName == 'CKSAAP':
            gap = self.dictParams['Gap:']
            encoded_data = CKSAAP(listSequences, int(gap))
        if self.featureName == 'CTriad':
            gap = self.dictParams['Gap:']
            encoded_data = CTriad(listSequences, int(gap))
        if self.featureName == 'DDE':
            encoded_data = DDE(listSequences)
        if self.featureName == 'KSCTriad':
            gap = self.dictParams['Gap:']
            encoded_data = KSCTriad(listSequences, int(gap))
        if self.featureName == 'TPC':
            encoded_data = TPC(listSequences)
        if self.featureName == 'Word2Vector':
            k = self.dictParams['K-mer:']
            vec_size = self.dictParams['Vector Size:']
            encoded_data = Word2Vector(listSequences, k_mer=int(k), vector_size=int(vec_size))
        shareInfo.menuFeature.ndarrayResult = encoded_data
        my_emit(signal.textBrowser_Message, str(encoded_data))


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
