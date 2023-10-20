from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QMessageBox

import pandas as pd
import numpy as np
from tabulate import tabulate

from sklearn.model_selection import train_test_split

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

import time
class MatplotlibCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.setParent(parent)
        FigureCanvas.setSizePolicy(self, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        #region setup
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1144, 754)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.metric_combobox = QtWidgets.QComboBox(self.centralwidget)
        self.metric_combobox.setGeometry(QtCore.QRect(590, 50, 151, 41))
        self.metric_combobox.setObjectName("metric_combobox")
        self.metric_combobox.addItem("")
        self.metric_combobox.addItem("")
        self.metric_combobox.addItem("")
        self.metric_combobox.addItem("")
        self.main_textedit = QtWidgets.QTextEdit(self.centralwidget)
        self.main_textedit.setGeometry(QtCore.QRect(30, 270, 531, 411))
        self.main_textedit.setObjectName("main_textedit")
        self.Load_data_button = QtWidgets.QPushButton(self.centralwidget)
        self.Load_data_button.setGeometry(QtCore.QRect(30, 170, 531, 41))
        self.Load_data_button.setObjectName("Load_data_button")
        self.enter_console_button = QtWidgets.QPushButton(self.centralwidget)
        self.enter_console_button.setGeometry(QtCore.QRect(380, 220, 181, 41))
        self.enter_console_button.setObjectName("enter_console_button")
        self.graphicsView = QtWidgets.QGraphicsView(self.centralwidget)
        self.graphicsView.setGeometry(QtCore.QRect(580, 270, 541, 411))
        self.graphicsView.setObjectName("graphicsView")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(580, 160, 541, 20))
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.data_choice_combobox = QtWidgets.QComboBox(self.centralwidget)
        self.data_choice_combobox.setGeometry(QtCore.QRect(30, 220, 231, 41))
        self.data_choice_combobox.setObjectName("data_choice_combobox")
        self.data_choice_combobox.addItem("")
        self.data_choice_combobox.addItem("")
        self.data_choice_combobox.addItem("")
        self.data_choice_combobox.addItem("")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(590, 20, 151, 31))
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(260, 230, 121, 21))
        self.label_3.setAlignment(QtCore.Qt.AlignCenter)
        self.label_3.setObjectName("label_3")
        self.count_k_spinBox = QtWidgets.QSpinBox(self.centralwidget)
        self.count_k_spinBox.setGeometry(QtCore.QRect(780, 50, 151, 41))
        self.count_k_spinBox.setAlignment(QtCore.Qt.AlignCenter)
        self.count_k_spinBox.setMinimum(1)
        self.count_k_spinBox.setMaximum(49)
        self.count_k_spinBox.setObjectName("count_k_spinBox")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(780, 20, 151, 31))
        self.label_4.setAlignment(QtCore.Qt.AlignCenter)
        self.label_4.setObjectName("label_4")
        self.comboBox_3 = QtWidgets.QComboBox(self.centralwidget)
        self.comboBox_3.setGeometry(QtCore.QRect(590, 210, 161, 41))
        self.comboBox_3.setObjectName("comboBox_3")
        self.comboBox_3.addItem("")
        self.comboBox_3.addItem("")
        self.comboBox_3.addItem("")
        self.comboBox_3.addItem("")
        self.comboBox_4 = QtWidgets.QComboBox(self.centralwidget)
        self.comboBox_4.setGeometry(QtCore.QRect(780, 210, 161, 41))
        self.comboBox_4.setObjectName("comboBox_4")
        self.comboBox_4.addItem("")
        self.comboBox_4.addItem("")
        self.comboBox_4.addItem("")
        self.comboBox_4.addItem("")
        self.line = QtWidgets.QFrame(self.centralwidget)
        self.line.setGeometry(QtCore.QRect(560, 20, 21, 681))
        self.line.setFrameShape(QtWidgets.QFrame.VLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.draw_button = QtWidgets.QPushButton(self.centralwidget)
        self.draw_button.setGeometry(QtCore.QRect(960, 210, 161, 41))
        self.draw_button.setObjectName("draw_button")
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(590, 190, 161, 16))
        self.label_6.setAlignment(QtCore.Qt.AlignCenter)
        self.label_6.setObjectName("label_6")
        self.label_7 = QtWidgets.QLabel(self.centralwidget)
        self.label_7.setGeometry(QtCore.QRect(780, 190, 161, 16))
        self.label_7.setAlignment(QtCore.Qt.AlignCenter)
        self.label_7.setObjectName("label_7")
        self.opt_k_button = QtWidgets.QPushButton(self.centralwidget)
        self.opt_k_button.setGeometry(QtCore.QRect(960, 50, 161, 41))
        self.opt_k_button.setObjectName("opt_k_button")
        self.label_8 = QtWidgets.QLabel(self.centralwidget)
        self.label_8.setGeometry(QtCore.QRect(960, 20, 161, 31))
        self.label_8.setAlignment(QtCore.Qt.AlignCenter)
        self.label_8.setObjectName("label_8")
        self.line_2 = QtWidgets.QFrame(self.centralwidget)
        self.line_2.setGeometry(QtCore.QRect(20, 150, 1111, 16))
        self.line_2.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_2.setObjectName("line_2")
        self.label_9 = QtWidgets.QLabel(self.centralwidget)
        self.label_9.setGeometry(QtCore.QRect(30, 55, 61, 21))
        self.label_9.setObjectName("label_9")
        self.label_10 = QtWidgets.QLabel(self.centralwidget)
        self.label_10.setGeometry(QtCore.QRect(30, 90, 61, 21))
        self.label_10.setObjectName("label_10")
        self.label_11 = QtWidgets.QLabel(self.centralwidget)
        self.label_11.setGeometry(QtCore.QRect(294, 55, 61, 21))
        self.label_11.setObjectName("label_11")
        self.label_12 = QtWidgets.QLabel(self.centralwidget)
        self.label_12.setGeometry(QtCore.QRect(294, 95, 61, 21))
        self.label_12.setObjectName("label_12")
        self.predict_button = QtWidgets.QPushButton(self.centralwidget)
        self.predict_button.setGeometry(QtCore.QRect(260, 130, 93, 28))
        self.predict_button.setObjectName("predict_button")
        self.doubleSpinBox = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.doubleSpinBox.setGeometry(QtCore.QRect(100, 50, 171, 31))
        self.doubleSpinBox.setProperty("value", 5.0)
        self.doubleSpinBox.setObjectName("doubleSpinBox")
        self.doubleSpinBox_2 = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.doubleSpinBox_2.setGeometry(QtCore.QRect(100, 90, 171, 31))
        self.doubleSpinBox_2.setProperty("value", 4.0)
        self.doubleSpinBox_2.setObjectName("doubleSpinBox_2")
        self.doubleSpinBox_3 = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.doubleSpinBox_3.setGeometry(QtCore.QRect(360, 50, 171, 31))
        self.doubleSpinBox_3.setObjectName("doubleSpinBox_3")
        self.doubleSpinBox_4 = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.doubleSpinBox_4.setGeometry(QtCore.QRect(360, 90, 171, 31))
        self.doubleSpinBox_4.setObjectName("doubleSpinBox_4")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(30, 19, 531, 31))
        self.label_5.setAlignment(QtCore.Qt.AlignCenter)
        self.label_5.setObjectName("label_5")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1144, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        # endregion

        # region parameters

        self.matplotlib_canvas = MatplotlibCanvas(self.centralwidget)
        self.matplotlib_canvas.setGeometry(QtCore.QRect(580, 270, 541, 411))

        self.clear_df_iris = pd.DataFrame()
        self.mod_df_iris = pd.DataFrame()
        self.clear_df_dop = pd.DataFrame()
        self.mod_df_dop = pd.DataFrame()
        # endregion parameters

        # region alex
        self.Load_data_button .clicked.connect(self.load_main_data)
        self.enter_console_button.clicked.connect(self.show_df_data)
        self.draw_button.clicked.connect(self.draw_button_click)
        self.predict_button.clicked.connect(self.predict_button_click)
        self.opt_k_button.clicked.connect(self.opt_k_button_click)
        # endregion alex

        # region setup
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        MainWindow.setTabOrder(self.doubleSpinBox, self.doubleSpinBox_3)
        MainWindow.setTabOrder(self.doubleSpinBox_3, self.doubleSpinBox_2)
        MainWindow.setTabOrder(self.doubleSpinBox_2, self.doubleSpinBox_4)
        MainWindow.setTabOrder(self.doubleSpinBox_4, self.predict_button)
        MainWindow.setTabOrder(self.predict_button, self.Load_data_button)
        MainWindow.setTabOrder(self.Load_data_button, self.data_choice_combobox)
        MainWindow.setTabOrder(self.data_choice_combobox, self.enter_console_button)
        MainWindow.setTabOrder(self.enter_console_button, self.main_textedit)
        MainWindow.setTabOrder(self.main_textedit, self.metric_combobox)
        MainWindow.setTabOrder(self.metric_combobox, self.count_k_spinBox)
        MainWindow.setTabOrder(self.count_k_spinBox, self.opt_k_button)
        MainWindow.setTabOrder(self.opt_k_button, self.comboBox_3)
        MainWindow.setTabOrder(self.comboBox_3, self.comboBox_4)
        MainWindow.setTabOrder(self.comboBox_4, self.draw_button)
        MainWindow.setTabOrder(self.draw_button, self.graphicsView)
        # endregion

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.metric_combobox.setItemText(0, _translate("MainWindow", "Эвклидова"))
        self.metric_combobox.setItemText(1, _translate("MainWindow", "Хемминга"))
        self.metric_combobox.setItemText(2, _translate("MainWindow", "Чебышева"))
        self.metric_combobox.setItemText(3, _translate("MainWindow", "Косинусная"))
        self.main_textedit.setHtml(_translate("MainWindow",
                                              "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
                                              "<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
                                              "p, li { white-space: pre-wrap; }\n"
                                              "</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:7.8pt; font-weight:400; font-style:normal;\">\n"
                                              "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Console</p>\n"
                                              "<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><br /></p></body></html>"))
        self.Load_data_button.setText(_translate("MainWindow", "Загрузить данные"))
        self.enter_console_button.setText(_translate("MainWindow", "Вывести данные ниже"))
        self.label.setText(_translate("MainWindow", "Окно визуализации"))
        self.data_choice_combobox.setItemText(0, _translate("MainWindow", "Исходный iris.csv"))
        self.data_choice_combobox.setItemText(1, _translate("MainWindow", "Модифицированный df ирисов"))
        self.data_choice_combobox.setItemText(2, _translate("MainWindow", "Вывести исходный dop.csv"))
        self.data_choice_combobox.setItemText(3, _translate("MainWindow", "Вывести заполненный dop.csv"))
        self.label_2.setText(_translate("MainWindow", "Выбор метрики"))
        self.label_3.setText(_translate("MainWindow", "-------->"))
        self.label_4.setText(_translate("MainWindow", "Число соседей k"))
        self.comboBox_3.setItemText(0, _translate("MainWindow", "Длина чашелистника"))
        self.comboBox_3.setItemText(1, _translate("MainWindow", "Ширина чашелистника"))
        self.comboBox_3.setItemText(2, _translate("MainWindow", "Длина лепестка"))
        self.comboBox_3.setItemText(3, _translate("MainWindow", "Ширина лепестка"))
        self.comboBox_4.setItemText(0, _translate("MainWindow", "Длина чашелистника"))
        self.comboBox_4.setItemText(1, _translate("MainWindow", "Ширина чашелистника"))
        self.comboBox_4.setItemText(2, _translate("MainWindow", "Длина лепестка"))
        self.comboBox_4.setItemText(3, _translate("MainWindow", "Ширина лепестка"))
        self.draw_button.setText(_translate("MainWindow", "Отрисовать"))
        self.label_6.setText(_translate("MainWindow", "x axis"))
        self.label_7.setText(_translate("MainWindow", "y axis"))
        self.opt_k_button.setText(_translate("MainWindow", "Вывести информацию"))
        self.label_8.setText(_translate("MainWindow", "Оптимальный k"))
        self.label_9.setText(_translate("MainWindow", "Sepal.L"))
        self.label_10.setText(_translate("MainWindow", "Sepal.W"))
        self.label_11.setText(_translate("MainWindow", "Petal.L"))
        self.label_12.setText(_translate("MainWindow", "Petal.W"))
        self.predict_button.setText(_translate("MainWindow", "Предсказать"))
        self.label_5.setText(_translate("MainWindow", "Введите данные ириса"))

    code_name_dict = {'0': "Setosa", '1': "Versicolor", '2': "Virginica"}
    code_FULLname_dict = {'0': "Iris-Setosa", '1': "Iris-Versicolor", '2': "Iris-Virginica"}

    def show_df_data(self):
        number_data_frame = self.data_choice_combobox.currentText()
        match number_data_frame:
            case "Исходный iris.csv":
                self.main_textedit.setText(self.clear_df_iris.to_string())
            case "Модифицированный df ирисов":
                self.main_textedit.setText(tabulate(self.mod_df_iris, headers='keys', tablefmt='pretty', showindex=False))
            case "Вывести исходный dop.csv":
                self.main_textedit.setText(self.clear_df_dop.to_string())
            case "Вывести заполненный dop.csv":
                #self.main_textedit.setText(tabulate(self.mod_df_dop, headers='keys', tablefmt='pretty', showindex=False))
                pass
            case _:
                pass

    def load_main_data(self):
        file_path = "data/iris.csv"
        df = pd.read_csv(file_path, sep=",")
        self.clear_df_iris = df

        df2 = pd.read_csv(file_path, sep=",")
        df2['num_kind'] = pd.factorize(df2['Kind'])[0]
        self.mod_df_iris = df2

        file_path = "data/Dop.csv"
        df_dop = pd.read_csv(file_path, sep=",")


        model = self.fit_our_model()
        df_dop["Kind"]=model.predict(df_dop.values, 1)
        df_dop['Kind'] = df_dop['Kind'].astype(str).replace(self.code_FULLname_dict)

        self.clear_df_dop=df_dop

        message_box = QMessageBox()
        message_box.setIcon(QMessageBox.Information)
        message_box.setText("Данные успешно загружены!")
        message_box.setWindowTitle("Успех")
        message_box.setStandardButtons(QMessageBox.Ok)
        message_box.setDefaultButton(QMessageBox.Ok)
        message_box.exec_()

    def draw_button_click(self):
        if self.clear_df_iris.empty:
            self.show_popup_critical("Ошибка", "Загрузите данные прежде чем нажать на кнопку")

        else:
            text1 = self.comboBox_3.currentText()
            text2 = self.comboBox_4.currentText()

            if text1 == text2:
                self.show_popup_critical("Ошибка","Выберите разные признаки для сравнения")
            else:
                match text1:
                    case "Длина чашелистника":
                        num_choice1 = 0
                    case "Ширина чашелистника":
                        num_choice1 = 1
                    case "Длина лепестка":
                        num_choice1 = 2
                    case "Ширина лепестка":
                        num_choice1 = 3
                    case _:
                        pass

                match text2:
                    case "Длина чашелистника":
                        num_choice2 = 0
                    case "Ширина чашелистника":
                        num_choice2 = 1
                    case "Длина лепестка":
                        num_choice2 = 2
                    case "Ширина лепестка":
                        num_choice2 = 3
                    case _:
                        pass


                # Очистка предыдущего графика
                self.matplotlib_canvas.fig.clear()

                # Вызов вашей функции для построения графика
                self.plot_iris_scatter(num_choice1, num_choice2, text1, text2)

                # Обновление виджета с новым графиком
                self.matplotlib_canvas.draw()
                # self.plot_iris_scatter(num_choice1, num_choice2,text1,text2)

    def plot_iris_scatter(self, feature_x, feature_y, xlabel, ylabel):
        data = self.mod_df_iris.iloc[:, :4].copy()
        target = self.mod_df_iris.iloc[:, 5].to_frame().copy()

        data = np.array(data)
        target = np.array(target)
        target = target.flatten()

        # Создание графика
        ax = self.matplotlib_canvas.fig.add_subplot(111)

        # Построение точек для каждого класса
        for class_value in set(target):
            # Получение индексов точек, принадлежащих текущему классу
            indices = (target == class_value)

            # Извлечение признаков для текущего класса
            x = data[indices, feature_x]  # Первый признак
            y = data[indices, feature_y]  # Второй признак

            # Построение точек с цветом, соответствующим текущему классу
            ax.scatter(x, y, label=f"Iris-{self.code_name_dict[str(class_value)]}")

        ax.legend()
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title("Scatter Plot of Iris Dataset")

    def predict_button_click(self):
        if self.clear_df_iris.empty:
            self.show_popup_critical("Ошибка", "Загрузите данные прежде чем нажать на кнопку")
        else:

            new_X = [[self.doubleSpinBox.value(),
                      self.doubleSpinBox_2.value(),
                      self.doubleSpinBox_3.value(),
                      self.doubleSpinBox_4.value()]]

            model = self.fit_our_model()
            prediction = model.predict(new_X)

            type_iris = f"Iris-{self.code_name_dict[str(prediction[0])]}"

            self.show_popup_information("Шаманим","Спрашиваю у Ванги что за ирис вы записали, \nподождите немного.")
            time.sleep(3)
            self.show_popup_information("Предсказание",f"Ваш ирис это {type_iris}")

    def fit_our_model(self):
        X = self.mod_df_iris.iloc[:, :4].copy()
        y = self.mod_df_iris.iloc[:, 5].to_frame().copy()


        X = np.array(X)
        y = np.array(y)
        y = y.flatten()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        count_k=self.count_k_spinBox.value()

        custom_knn = CustomKNN(count_k)
        custom_knn.fit(X_train, y_train)

        return custom_knn

    def opt_k_button_click(self):
        if self.clear_df_iris.empty:
            self.show_popup_critical("Ошибка", "Загрузите данные прежде чем нажать на кнопку")
        else:
            X = self.mod_df_iris.iloc[:, :1].copy()
            y = self.mod_df_iris.iloc[:, 5].to_frame().copy()

            X = np.array(X)
            y = np.array(y)
            y = y.flatten()

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


            output_log=""
            maxim_value=0

            for key in range(1,150):
                custom_knn = CustomKNN(key)
                custom_knn.fit(X_train, y_train)

                y_pred = custom_knn.predict(X_test, 1)
                count = 0
                for i in range(len(y_pred)):
                    if y_pred[i] == y_test[i]:
                        count += 1

                tmp_value=round(count / len(y_pred) * 100,2)
                output_log+=f"Точность, при k={key}  >>> {tmp_value}%\n"
                if tmp_value >= maxim_value:
                    maxim_value = tmp_value
                    best_k = key
            messagebox_log = f"Самый оптимальный k={best_k}, с точностью {maxim_value}"
            output_log += messagebox_log
            self.main_textedit.setText(output_log)
            self.show_popup_information("Оптимальное количество соседей",messagebox_log)

    # region message_box
    @staticmethod
    def show_popup_information(messagebox_title_inner: str, content_text_inner: str):
        message_box = QMessageBox()
        message_box.setIcon(QMessageBox.Information)
        message_box.setText(content_text_inner)
        message_box.setWindowTitle(messagebox_title_inner)
        message_box.setStandardButtons(QMessageBox.Ok)
        message_box.setDefaultButton(QMessageBox.Ok)

        message_box.exec_()

    @staticmethod
    def show_popup_warning(messagebox_title_inner: str, content_text_inner: str):
        message_box = QMessageBox()
        message_box.setIcon(QMessageBox.Warning)
        message_box.setText(content_text_inner)
        message_box.setWindowTitle(messagebox_title_inner)
        message_box.setStandardButtons(QMessageBox.Ok)
        message_box.setDefaultButton(QMessageBox.Ok)

        message_box.exec_()

    @staticmethod
    def show_popup_critical(messagebox_title_inner: str, content_text_inner: str):
        message_box = QMessageBox()
        message_box.setIcon(QMessageBox.Critical)
        message_box.setText(content_text_inner)
        message_box.setWindowTitle(messagebox_title_inner)
        message_box.setStandardButtons(QMessageBox.Ok)
        message_box.setDefaultButton(QMessageBox.Ok)

        message_box.exec_()
    # endregion


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    import sys
    from CustomKNN import CustomKNN
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
