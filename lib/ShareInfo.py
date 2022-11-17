class MenuFile:
    def __init__(self):
        self.trainFileData = None
        self.testFileData = None
        self.predictionFileData = None
        self.preparationFileData = None
        self.model = None
        self.feature = None


class MenuPreparation:
    def __init__(self):
        # self.lengthClipping = None
        # self.CD_HIT = None
        # self.formatFile = None
        self.listResult = None


class MenuFeature:
    def __init__(self):
        self.ndarrayResult = None


class MenuModel:
    def __init__(self):
        self.encodingName = None
        self.encodingParams = None
        self.modelName = None
        self.modelParams = None
        self.validation = 5

        self.trainedModel = None
        self.kFoldValidationROCFig = None
        self.dictMetrics = None


class MenuPrediction:
    def __init__(self):
        self.listPredictionResult = None


class MenuVisualization:
    def __init__(self):
        self.xxx = None
        self.xxxx = None


class ShareInfo:
    def __init__(self):
        self.menuFile = MenuFile()
        self.menuPreparation = MenuPreparation()
        self.menuFeature = MenuFeature()
        self.menuModel = MenuModel()

    def DefaultMenuFile(self):
        self.menuFile = MenuFile()


shareInfo = ShareInfo()
