from lib.MyThreads import *
import pyqtgraph as pg
from PySide2.QtCore import Qt, QCoreApplication
from PySide2.QtWidgets import QApplication, QFileDialog, QMainWindow, QMessageBox, QSpinBox, QTableWidgetItem
from PySide2.QtUiTools import QUiLoader
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from sklearn.metrics import confusion_matrix, classification_report


class Win_Main(QMainWindow):

    def __init__(self):
        super().__init__()
        self.option_ui = None

        # 加载主窗口
        self.ui = QUiLoader().load("ui/main.ui")

        # 文件菜单栏
        self.ui.action_train_file.triggered.connect(self.handle_train_file)  # 多线程导入训练文件
        self.ui.action_predict_file.triggered.connect(self.handle_predict_file)  # 多线程导入预测文件
        self.ui.action_save_results.triggered.connect(self.handle_save_results)  # 多线程保存结果
        self.ui.action_load_model.triggered.connect(self.action_load_model)  # 多线程加载模型
        self.ui.action_exit.triggered.connect(self.handle_exit)

        # 选项菜单栏
        self.ui.action_option_setting.triggered.connect(self.handle_option_setting)
        self.ui.action_model_CNN.triggered.connect(self.handle_model_CNN)
        self.ui.action_model_LSTM.triggered.connect(self.handle_model_LSTM)
        self.ui.action_model_CNN_LSTM.triggered.connect(self.handle_model_CNN_LSTM)
        self.ui.action_use_gpu.triggered.connect(self.handle_use_gpu)
        self.ui.action_start_train.triggered.connect(self.handle_start_train)  # 多线程训练模型
        self.ui.action_start_predict.triggered.connect(self.handle_start_predict)  # 多线程预测结果
        self.ui.action_save_model.triggered.connect(self.handle_save_model)  # 多线程保存模型

        # 视图菜单栏
        ''' 暂未处理 '''

        # 帮助菜单栏
        self.ui.action_document.triggered.connect(self.handle_document)
        self.ui.action_about.triggered.connect(self.handle_about)

        # 接收进度条的信号，进行设置
        signal.progress_bar.connect(self.set_progress_bar)

        # 接收提示信息的信号，进行设置
        signal.tips.connect(self.set_process_tips)

        # 接收混淆矩阵的信号，进行显示
        signal.confusion_matrix.connect(self.show_confusion_matrix)

        # 接收分类报告的信号，进行显示
        signal.classification_report.connect(self.show_classification_report)

        # 接收绘图的信号，进行绘图
        signal.draw.connect(self.draw_stock)

    # 显示混淆矩阵的槽函数
    def show_confusion_matrix(self):
        y_true, y_pred = decode_one_hot(SI.y_true, SI.model.listResult)
        cm = confusion_matrix(y_true, y_pred)

        # 将混淆矩阵绘图，并得到figure对象
        fig = get_confusion_matrix_figure(cm)

        # 创建画布，用于后续添加到视图区
        self.canvas = FigureCanvas(fig)
        self.canvas.draw()

        # 删除占位控件
        for i in reversed(range(self.ui.verticalLayout_confusion_matrix.count())):
            self.ui.verticalLayout_confusion_matrix.itemAt(i).widget().setParent(None)

        # 添加绘图部件到布局
        self.ui.verticalLayout_confusion_matrix.addWidget(self.canvas)

        # 显示分类报告

    # 显示分类报告的槽函数
    def show_classification_report(self):
        y_true, y_pred = decode_one_hot(SI.y_true, SI.model.listResult)
        original_report = classification_report(y_true, y_pred, target_names=('(+1)', '(+0)', '(-1)'), digits=4)

        # 处理分类报告
        report = deal_classification_report(original_report)

        _translate = QCoreApplication.translate
        for i in range(5):
            for j in range(3):
                self.ui.tableWidget_metrics_analyse.setItem(i, j, QTableWidgetItem())
                self.ui.tableWidget_metrics_analyse.item(i, j).setText(_translate("widget", report[i][j]))

        # 合并第5行的单元格
        self.ui.tableWidget_metrics_analyse.setSpan(4, 0, 1, 3)
        self.ui.tableWidget_metrics_analyse.item(4, 0).setText(_translate("widget", report[3][2]))
        self.ui.tableWidget_metrics_analyse.item(4, 0).setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)

    # 设置进度条
    def set_progress_bar(self, value):
        self.ui.progressBar.setValue(value)

    # 设置过程提示文字
    def set_process_tips(self, value):
        self.ui.label_process_tips.setText(value)

    # 绘制股票走势曲线图
    def draw_stock(self, value):

        if value == 1:
            label = np.argmax(SI.y_train, axis=1) + 1
            last = SI.x_train[0, 0, 0, 0] * 100
        else:
            label = np.argmax(SI.y_true, axis=1) + 1
            last = SI.x_data[0, 0, 0, 0] * 100

        stocks = get_stock_data(label, last)


        pw = pg.PlotWidget()  # 实例化一个绘图部件
        pw.plot(stocks)  # 在绘图部件中绘制折线图

        # 删除占位控件
        for i in reversed(range(self.ui.verticalLayout_stock_data.count())):
            self.ui.verticalLayout_stock_data.itemAt(i).widget().setParent(None)

        self.ui.verticalLayout_stock_data.addWidget(pw)  # 添加绘图部件到布局

        # 发送绘图完成的信号
        signal.progress_bar.emit(100)
        signal.tips.emit('股票曲线图绘制完成！')

    # 选择训练文件
    def handle_train_file(self):
        # 获取训练文件名
        file_names_tuple = QFileDialog().getOpenFileNames(self.ui, '选择训练文件', './',
                                                          "所有文件 (*);;文本文件 (*.txt);;Excel文件 (*.xls);;Excel文件 (*.xlsx)")
        file_names_lst = file_names_tuple[0]

        if not file_names_lst:
            signal.progress_bar.emit(0)
            signal.tips.emit('未选择训练文件！')
            # 如果之前选择过训练文件
            if SI.train_file_lst:
                self.ui.action_train_file.setChecked(True)
                return
            # 如果之前没有选择过训练文件
            self.ui.action_train_file.setChecked(False)
            return

        SI.train_file_lst = file_names_lst
        self.ui.action_train_file.setChecked(True)

        # 启动加载训练文件线程
        self.thread_trian_file = Thread_Trian_File(file_names_lst, self.ui.verticalLayout_stock_data)
        self.thread_trian_file.start()

    # 选择预测文件
    def handle_predict_file(self):
        # 获取预测文件名
        file_names_tuple = QFileDialog().getOpenFileNames(self.ui, '选择预测文件', './',
                                                          "所有文件 (*);;文本文件 (*.txt);;Excel文件 (*.xls);;Excel文件 (*.xlsx)")
        file_names_lst = file_names_tuple[0]

        if not file_names_lst:
            signal.tips.emit('未选择预测文件！')
            # 之前选择过预测文件
            if SI.predict_file_lst:
                self.ui.action_predict_file.setChecked(True)
                return
            # 之前没有选择过预测文件
            self.ui.action_predict_file.setChecked(False)
            return

        SI.predict_file_lst = file_names_lst
        self.ui.action_predict_file.setChecked(True)

        # 启动加载训练文件线程
        self.thread_predict_file = Thread_Predict_File(file_names_lst)
        self.thread_predict_file.start()

    # 保存预测结果
    def handle_save_results(self):
        # 当前没有模型或者预测结果为空
        if not SI.model or not len(SI.model.listResult):
            QMessageBox.warning(self.ui, '提示', '当前没有预测结果')
            return

        # 设置文件默认名字为预测结果+当前时间
        default_file_name = time.strftime('预测结果%Y%m%d%H%M%S', time.localtime())
        file_name = QFileDialog().getSaveFileName(self.ui, '保存预测结果', './' + default_file_name, "文本文件 (*.txt)")[0]

        if file_name:
            # 如果点击了确认按钮，则启动线程进行保存预测结果
            self.thread_save_result = Thread_Save_Result(file_name, SI.model.listResult)
            self.thread_save_result.start()
        else:
            signal.progress_bar.emit(0)
            signal.tips.emit('您取消了文件选择，预测结果未保存！')

    # 加载已有模型
    def action_load_model(self):
        # 获取模型存储的位置
        model_name = QFileDialog().getOpenFileName(self.ui, '选择模型文件', './', "h5文件 (*.h5)")[0]
        if model_name:
            # 如果点击了确认按钮，则启动线程加载模型
            self.thread = Thread_Import_Preparation_File(model_name)
            self.thread.start()
        else:
            signal.progress_bar.emit(0)
            signal.tips.emit('您取消了加载模型！')

    # 退出
    def handle_exit(self):
        reply = QMessageBox.question(self.ui, '退出', '确认退出', QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
        if reply == QMessageBox.Yes:
            self.ui.close()

    # 选项设置
    def handle_option_setting(self):
        # 加载选项设置
        if not self.option_ui:
            self.option_ui = QUiLoader().load("ui/option_setting.ui")
            self.option_ui.setWindowModality(Qt.ApplicationModal)  # 设置为模态窗口

        # '''加载之前的设置'''
        # 模型架构
        if SI.option_setting['model_name'] == 'CNN':
            self.option_ui.radioButton_CNN.setChecked(True)
        elif SI.option_setting['model_name'] == 'LSTM':
            self.option_ui.radioButton_LSTM.setChecked(True)
        else:
            self.option_ui.radioButton_CNN_LSTM.setChecked(True)

        # 是否启用GPU
        if SI.option_setting['use_gpu']:
            self.option_ui.radioButton_use_gpu.setChecked(True)
        else:
            self.option_ui.radioButton_forbid_gpu.setChecked(True)

        # 训练参数 [epochs,h_value,batch_size]
        params = SI.option_setting['params']
        self.option_ui.epochs.setValue(int(params[0]))
        self.option_ui.h_value.setValue(int(params[1]))
        self.option_ui.batch_size.setValue(int(params[2]))

        # 按钮触发事件
        self.option_ui.option_setting_btn_ok.clicked.connect(self.handle_option_setting_ok)
        self.option_ui.option_setting_btn_cancel.clicked.connect(self.handle_option_setting_cancel)
        self.option_ui.option_setting_btn_default.clicked.connect(self.handle_option_setting_default)

        self.option_ui.show()

    # 选择CNN模型
    def handle_model_CNN(self):
        self.ui.action_model_CNN.setChecked(True)
        self.ui.action_model_LSTM.setChecked(False)
        self.ui.action_model_CNN_LSTM.setChecked(False)
        SI.option_setting['model_name'] = 'CNN'
        self.set_process_tips('模型架构已选择CNN')

    # 选择LSTM模型
    def handle_model_LSTM(self):
        self.ui.action_model_LSTM.setChecked(True)
        self.ui.action_model_CNN.setChecked(False)
        self.ui.action_model_CNN_LSTM.setChecked(False)
        SI.option_setting['model_name'] = 'LSTM'
        self.set_process_tips('模型架构已选择LSTM')

    # 选择CNN_LSTM模型
    def handle_model_CNN_LSTM(self):
        self.ui.action_model_CNN_LSTM.setChecked(True)
        self.ui.action_model_CNN.setChecked(False)
        self.ui.action_model_LSTM.setChecked(False)
        SI.option_setting['model_name'] = 'CNN_LSTM'
        self.set_process_tips('模型架构已选择CNN—LSTM')

    # 是否启用GPU
    def handle_use_gpu(self):
        SI.option_setting['use_gpu'] = self.ui.action_use_gpu.isChecked()
        if SI.option_setting['use_gpu']:
            self.set_process_tips('已启用GPU进行加速！')
        else:
            self.set_process_tips('已禁用CPU进行加速！')

    # 开始训练
    def handle_start_train(self):
        if SI.train_file_lst:
            if QMessageBox.question(self.ui, '提示', '训练时间可能比较长，确认是否要训练？') == QMessageBox.Yes:
                # 启动线程训练模型
                self.thread_start_train = Thread_Start_Train()
                self.thread_start_train.start()
        else:
            self.warnning_tips_dialog(self.ui, '提示', '请先选择训练文件！')

    # 开始预测
    def handle_start_predict(self):
        if SI.predict_file_lst:
            # 启动线程预测数据
            self.thread_start_predict = Thread_Start_Predict()
            self.thread_start_predict.start()
        else:
            self.warnning_tips_dialog(self.ui, '提示', '请先选择预测文件！')

    # 保存当前模型
    def handle_save_model(self):
        if SI.model == None:
            QMessageBox.warning(self.ui, '提示', '当前没有模型')
            return

        # 设置文件默认名字为模型+当前时间
        default_file_name = time.strftime('模型%Y%m%d%H%M%S', time.localtime())
        file_name = QFileDialog().getSaveFileName(self.ui, '选择训练文件', './' + default_file_name, "h5文件 (*.h5)")[0]

        if file_name:
            # 启动线程保存预测模型
            self.thread_save_model = Thread_Save_Model(file_name, SI.model.listResult)
            self.thread_save_model.start()
        else:
            signal.progress_bar.emit(0)
            signal.tips.emit('您取消了文件选择，模型未保存！')

    # 帮助文档
    def handle_document(self):
        import webbrowser

        s = ''' #####需要替换帮助文档的地址##### '''
        url = 'https://www.baidu.com'

        webbrowser.open_new_tab(url)
        self.set_process_tips('已打开帮助文档！')

    # 关于
    def handle_about(self):
        self.about_ui = QUiLoader().load("ui/about.ui")
        self.about_ui.show()

    # 处理《选项设置》的按钮事件——确认
    def handle_option_setting_ok(self):
        # 获取模型架构的名字
        # 注意：0号元素为QGroupBox的布局对象，1号元素为QRadioButton的列表
        for radiobtn in self.option_ui.groupBox_model_name.children()[1:]:
            if radiobtn.isChecked():
                model_name = radiobtn.text()
                break
        else:
            self.warnning_tips_dialog(self.option_ui, '提示', '请选择模型！')
            return

        # 获取是否选用GPU加速
        for radiobtn in self.option_ui.groupBox_use_gpu.children()[1:]:
            if radiobtn.isChecked():
                use_gpu = (True if radiobtn.text() == '启用' else False)
                break
        else:
            self.warnning_tips_dialog(self.option_ui, '提示', '是否启用GPU加速？')
            return

        # 获取训练参数
        epochs = self.option_ui.groupBox_train_params.findChild(QSpinBox, 'epochs').text()
        h_value = self.option_ui.groupBox_train_params.findChild(QSpinBox, 'h_value').text()
        batch_size = self.option_ui.groupBox_train_params.findChild(QSpinBox, 'batch_size').text()

        # 是否记住设置
        is_remember = self.option_ui.checkBox_remember.isChecked()

        # 更新设置
        SI.option_setting['model_name'] = model_name
        SI.option_setting['use_gpu'] = use_gpu
        SI.option_setting['params'] = [epochs, h_value, batch_size]
        SI.option_setting['metrics'] = 'accuracy'
        SI.option_setting['is_remember'] = is_remember

        # 更新菜单栏——模型框架
        if model_name == 'CNN':
            self.handle_model_CNN()
        elif model_name == 'LSTM':
            self.handle_model_LSTM()
        else:  # 注意：model_name == 'CNN-LSTM'，不是下划线CNN_LSTM
            self.handle_model_CNN_LSTM()

        # 更新菜单栏——是否启用GPU
        if use_gpu == True:
            self.ui.action_use_gpu.setChecked(True)
        else:
            self.ui.action_use_gpu.setChecked(False)

        # 如果记住设置，则写入设置文件
        if is_remember:
            with open('lib/settings.py', 'w') as f:
                f.write('option_setting = ')
                f.write(str(SI.option_setting))
                f.close()

        self.option_ui.hide()
        self.set_process_tips('已更新选项设置！')

    # 处理《选项设置》的按钮事件——取消
    def handle_option_setting_cancel(self):
        self.option_ui.close()
        self.option_ui = None
        self.set_process_tips('用户取消更新了选项设置！')

    # 处理《选项设置》的按钮事件——恢复默认
    def handle_option_setting_default(self):
        self.option_ui.radioButton_CNN.setChecked(True)
        self.option_ui.radioButton_forbid_gpu.setChecked(True)
        self.option_ui.epochs.setValue(SI.default_epochs)
        self.option_ui.h_value.setValue(SI.default_h_value)
        self.option_ui.batch_size.setValue(SI.default_batch_size)

    # 弹出警告对话框
    def warnning_tips_dialog(self, parent, title, content):
        QMessageBox.warning(parent, title, content, QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)


app = QApplication([])
SI.mainWin = Win_Main()
SI.mainWin.ui.show()
app.exec_()
