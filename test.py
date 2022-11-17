import matplotlib.pyplot as plt
from PySide2.QtWidgets import QGraphicsScene
from matplotlib.figure import Figure

from lib.DataProcessing import LoadData

roc_fig = plt.figure()
fig = plt.subplot(1, 1, 1)
# graphicsSceneROC = QGraphicsScene()
# graphicsSceneROC.addWidget(fig)
print(type(roc_fig))
