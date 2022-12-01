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
        self.featureName = 'AAC'
        self.featureParams = {}
        self.ndarrayResult = None


class MenuModel:
    def __init__(self):
        self.encodingName = 'AAC'
        self.encodingParams = {}
        self.modelName = 'LR'
        self.modelParams = {}
        self.validation = 5

        self.trainedModel = None
        self.canvasROC = None
        self.canvasConfusionMatrix = None
        self.classificationReport = None


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
        self.menuPrediction = MenuPrediction()

    def DefaultMenuFile(self):
        self.menuFile = MenuFile()


shareInfo = ShareInfo()
