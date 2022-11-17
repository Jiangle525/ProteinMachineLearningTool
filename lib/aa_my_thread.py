import random
import time
import math
from PySide2.QtCore import QThread
from keras.callbacks import Callback
from lib.model.cnn import CNN_LOB
from lib.model.cnn_lstm import CNN_LSTM_LOB
from lib.model.lstm import LSTM_LOB
from lib.share import SI
from lib.signal import signal
from lib.draw import *


class Thread_Load_Model(QThread):
    '''
        处理加载模型的线程
    '''

    def __init__(self, model_name):
        super(Thread_Load_Model, self).__init__()
        self.model_name = model_name

    def run(self):
        signal.tips.emit('正在导入模型...')
        for i in range(100):
            signal.progress_bar.emit(i)
            time.sleep(random.randint(0, 10) / 400)
        SI.model.load_model(self.model_name)
        signal.progress_bar.emit(100)
        signal.tips.emit('模型导入成功！')


class Thread_Trian_File(QThread):
    '''
       处理导入训练文件的线程
    '''

    def __init__(self, file_names_lst, layout):
        super(Thread_Trian_File, self).__init__()
        self.file_names_lst = file_names_lst
        self.layout = layout

    def run(self):
        signal.tips.emit('正在加载训练文件...')
        for i in range(100):
            time.sleep((random.randint(0, 10) / 500) * len(self.file_names_lst))
            signal.progress_bar.emit(i)
        signal.tips.emit('训练文件加载成功！正在绘制曲线图.....')
        SI.x_train, SI.y_train = SI.model.LoadData(self.file_names_lst)
        signal.draw.emit(1)  # 发送显示训练文件股票走势的信号


class Thread_Predict_File(QThread):
    '''
        处理导入预测数据的文件
    '''

    def __init__(self, file_names_lst):
        super(Thread_Predict_File, self).__init__()
        self.file_names_lst = file_names_lst

    def run(self):
        signal.tips.emit('正在加载预测文件...')
        for i in range(100):
            time.sleep((random.randint(0, 10) / 500) * len(self.file_names_lst))
            signal.progress_bar.emit(i)
        signal.tips.emit('预测文件加载成功！正在绘制曲线图.....')
        SI.x_data, SI.y_true = SI.model.LoadData(self.file_names_lst)
        signal.draw.emit(0)  # 发送显示训练文件股票走势的信号


class Thread_Save_Result(QThread):
    '''
        处理保存预测结果的线程
    '''

    def __init__(self, file_name, result):
        super(Thread_Save_Result, self).__init__()
        self.file_name = file_name
        self.result = result

    def run(self):
        signal.tips.emit('正在保存预测结果...')
        result = np.argmax(self.result, axis=1)

        batch = math.ceil(len(result) / 100)
        with open(self.file_name, 'w', encoding='utf-8') as f:
            f.write('')
        with open(self.file_name, 'a', encoding='utf-8') as f:
            for i in range(101):
                time.sleep(0.025)
                signal.progress_bar.emit(i)
                for i in result[i * batch:(i + 1) * batch]:
                    f.write(str(i + 1) + ' ')

        signal.tips.emit('预测结果保存成功！')


class Thread_Save_Model(QThread):
    '''
        处理保存模型的线程
    '''

    def __init__(self, file_name, result):
        super(Thread_Save_Model, self).__init__()
        self.file_name = file_name
        self.result = result

    def run(self):
        signal.tips.emit('正在保存模型...')
        for i in range(101):
            signal.progress_bar.emit(i)
            time.sleep(random.randint(0, 10) / 400)
        signal.tips.emit('{}模型已保存！'.format(self.file_name.split('/')[-1]))
        SI.model.save_model(self.file_name)


class Train_CallBack(Callback):
    '''
        用于训练数据时的回调函数
        发送训练数据的信号
    '''

    # 在每一个epoch之后发送当前进度信号值
    def on_epoch_end(self, epoch, logs=None):
        signal.progress_bar.emit(round(epoch * 100 / self.params['epochs']))


class Predict_CallBack(Callback):
    '''
        用于预测数据时的回调函数，
        发送预测数据进度的信号
    '''

    # 在每一个epoch之后发送当前进度信号值params['steps']为总批量次数，batch为当前批量序号
    def on_predict_batch_end(self, batch, logs=None):
        signal.progress_bar.emit(round(batch * 100 / self.params['steps']))


class Thread_Start_Predict(QThread):
    '''
        预测数据的线程
    '''

    def __init__(self):
        super(Thread_Start_Predict, self).__init__()

    def run(self):
        signal.tips.emit('正在预测中...')
        signal.progress_bar.emit(0)

        # 创建一个Predict_CallBack的对象，用于回调
        predict_callback = Predict_CallBack()

        # 预测数据
        SI.model.predict(SI.x_data, batch_size=128, callbacks=[predict_callback])

        # 发送显示混淆矩阵和分类报告的信号
        signal.confusion_matrix.emit('1')
        signal.classification_report.emit('1')

        signal.progress_bar.emit(100)
        signal.tips.emit('预测成功！')


class Thread_Start_Train(QThread):

    def __init__(self):
        super(Thread_Start_Train, self).__init__()

    def run(self):
        signal.progress_bar.emit(0)
        signal.tips.emit('正在训练模型中...')
        if SI.option_setting['model_name'] == 'CNN':
            SI.model = CNN_LOB()
        elif SI.option_setting['model_name'] == 'LSTM':
            SI.model = LSTM_LOB()
        else:
            SI.model = CNN_LSTM_LOB()

        # 创建一个Train_CallBack的对象，用于回调
        train_callback = Train_CallBack()

        # 获取训练过程中的参数
        epochs = int(SI.option_setting['params'][0])
        batch_size = int(SI.option_setting['params'][2])

        # 训练模型
        SI.model.train(SI.x_train, SI.y_train, epochs=epochs, batch_size=batch_size, callbacks=[train_callback])

        signal.progress_bar.emit(100)
        signal.tips.emit('训练已完成，注意及时保存模型！')
