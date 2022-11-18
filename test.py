from sklearn.metrics import confusion_matrix

from lib.Visualization import draw_confusion_matrix

cm = confusion_matrix([1, 0, 1, 1], [0, 1, 1, 1])
can = draw_confusion_matrix(cm)
can.figure.savefig('result111.png')
