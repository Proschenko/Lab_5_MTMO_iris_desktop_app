from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QMessageBox

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

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
        MainWindow.resize(1687, 782)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.metric_combobox = QtWidgets.QComboBox(self.centralwidget)
        self.metric_combobox.setGeometry(QtCore.QRect(220, 50, 151, 31))
        self.metric_combobox.setObjectName("metric_combobox")
        self.metric_combobox.addItem("")
        self.metric_combobox.addItem("")
        self.metric_combobox.addItem("")
        self.metric_combobox.addItem("")
        self.main_textedit = QtWidgets.QTextEdit(self.centralwidget)
        self.main_textedit.setGeometry(QtCore.QRect(470, 180, 631, 541))
        self.main_textedit.setObjectName("main_textedit")
        self.Load_data_button = QtWidgets.QPushButton(self.centralwidget)
        self.Load_data_button.setGeometry(QtCore.QRect(470, 50, 631, 41))
        self.Load_data_button.setObjectName("Load_data_button")
        self.enter_console_button = QtWidgets.QPushButton(self.centralwidget)
        self.enter_console_button.setGeometry(QtCore.QRect(920, 130, 181, 41))
        self.enter_console_button.setObjectName("enter_console_button")
        self.graphicsView = QtWidgets.QGraphicsView(self.centralwidget)
        self.graphicsView.setGeometry(QtCore.QRect(1120, 290, 541, 431))
        self.graphicsView.setObjectName("graphicsView")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(1120, 180, 541, 31))
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.data_choice_combobox = QtWidgets.QComboBox(self.centralwidget)
        self.data_choice_combobox.setGeometry(QtCore.QRect(470, 130, 311, 41))
        self.data_choice_combobox.setObjectName("data_choice_combobox")
        self.data_choice_combobox.addItem("")

        self.data_choice_combobox.addItem("")
        self.data_choice_combobox.addItem("")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(70, 50, 141, 31))
        self.label_2.setAlignment(QtCore.Qt.AlignLeading | QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(790, 130, 131, 41))
        self.label_3.setAlignment(QtCore.Qt.AlignCenter)
        self.label_3.setObjectName("label_3")
        self.count_k_spinBox = QtWidgets.QSpinBox(self.centralwidget)
        self.count_k_spinBox.setGeometry(QtCore.QRect(220, 100, 151, 31))
        self.count_k_spinBox.setAlignment(QtCore.Qt.AlignCenter)
        self.count_k_spinBox.setMinimum(1)
        self.count_k_spinBox.setMaximum(149)
        self.count_k_spinBox.setObjectName("count_k_spinBox")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(70, 100, 141, 31))
        self.label_4.setAlignment(QtCore.Qt.AlignLeading | QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self.label_4.setObjectName("label_4")
        self.comboBox_3 = QtWidgets.QComboBox(self.centralwidget)
        self.comboBox_3.setGeometry(QtCore.QRect(1130, 240, 161, 41))
        self.comboBox_3.setObjectName("comboBox_3")
        self.comboBox_3.addItem("")
        self.comboBox_3.addItem("")
        self.comboBox_3.addItem("")
        self.comboBox_3.addItem("")
        self.comboBox_4 = QtWidgets.QComboBox(self.centralwidget)
        self.comboBox_4.setGeometry(QtCore.QRect(1320, 240, 161, 41))
        self.comboBox_4.setObjectName("comboBox_4")
        self.comboBox_4.addItem("")
        self.comboBox_4.addItem("")
        self.comboBox_4.addItem("")
        self.comboBox_4.addItem("")
        self.line = QtWidgets.QFrame(self.centralwidget)
        self.line.setGeometry(QtCore.QRect(1100, 20, 21, 711))
        self.line.setFrameShape(QtWidgets.QFrame.VLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.draw_button = QtWidgets.QPushButton(self.centralwidget)
        self.draw_button.setGeometry(QtCore.QRect(1500, 240, 161, 41))
        self.draw_button.setObjectName("draw_button")
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(1130, 220, 161, 16))
        self.label_6.setAlignment(QtCore.Qt.AlignCenter)
        self.label_6.setObjectName("label_6")
        self.label_7 = QtWidgets.QLabel(self.centralwidget)
        self.label_7.setGeometry(QtCore.QRect(1320, 220, 161, 16))
        self.label_7.setAlignment(QtCore.Qt.AlignCenter)
        self.label_7.setObjectName("label_7")
        self.opt_k_button = QtWidgets.QPushButton(self.centralwidget)
        self.opt_k_button.setGeometry(QtCore.QRect(220, 140, 151, 31))
        self.opt_k_button.setObjectName("opt_k_button")
        self.label_8 = QtWidgets.QLabel(self.centralwidget)
        self.label_8.setGeometry(QtCore.QRect(70, 140, 141, 31))
        self.label_8.setAlignment(QtCore.Qt.AlignLeading | QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self.label_8.setObjectName("label_8")
        self.line_2 = QtWidgets.QFrame(self.centralwidget)
        self.line_2.setGeometry(QtCore.QRect(1110, 170, 571, 20))
        self.line_2.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_2.setObjectName("line_2")
        self.label_9 = QtWidgets.QLabel(self.centralwidget)
        self.label_9.setGeometry(QtCore.QRect(1130, 55, 61, 21))
        self.label_9.setObjectName("label_9")
        self.label_10 = QtWidgets.QLabel(self.centralwidget)
        self.label_10.setGeometry(QtCore.QRect(1130, 90, 61, 21))
        self.label_10.setObjectName("label_10")
        self.label_11 = QtWidgets.QLabel(self.centralwidget)
        self.label_11.setGeometry(QtCore.QRect(1394, 55, 61, 21))
        self.label_11.setObjectName("label_11")
        self.label_12 = QtWidgets.QLabel(self.centralwidget)
        self.label_12.setGeometry(QtCore.QRect(1394, 95, 61, 21))
        self.label_12.setObjectName("label_12")
        self.predict_button = QtWidgets.QPushButton(self.centralwidget)
        self.predict_button.setGeometry(QtCore.QRect(1360, 140, 93, 28))
        self.predict_button.setObjectName("predict_button")
        self.doubleSpinBox = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.doubleSpinBox.setGeometry(QtCore.QRect(1200, 50, 171, 31))
        self.doubleSpinBox.setProperty("value", 5.0)
        self.doubleSpinBox.setObjectName("doubleSpinBox")
        self.doubleSpinBox_2 = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.doubleSpinBox_2.setGeometry(QtCore.QRect(1200, 90, 171, 31))
        self.doubleSpinBox_2.setProperty("value", 4.0)
        self.doubleSpinBox_2.setObjectName("doubleSpinBox_2")
        self.doubleSpinBox_3 = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.doubleSpinBox_3.setGeometry(QtCore.QRect(1460, 50, 171, 31))
        self.doubleSpinBox_3.setObjectName("doubleSpinBox_3")
        self.doubleSpinBox_4 = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.doubleSpinBox_4.setGeometry(QtCore.QRect(1460, 90, 171, 31))
        self.doubleSpinBox_4.setObjectName("doubleSpinBox_4")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(1130, 10, 531, 31))
        self.label_5.setAlignment(QtCore.Qt.AlignCenter)
        self.label_5.setObjectName("label_5")
        self.checkBox_1 = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_1.setEnabled(True)
        self.checkBox_1.setGeometry(QtCore.QRect(40, 300, 171, 21))
        self.checkBox_1.setChecked(True)
        self.checkBox_1.setObjectName("checkBox_1")
        self.checkBox_5 = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_5.setGeometry(QtCore.QRect(40, 460, 171, 21))
        self.checkBox_5.setObjectName("checkBox_5")
        self.checkBox_norm = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_norm.setGeometry(QtCore.QRect(70, 180, 301, 20))
        self.checkBox_norm.setTristate(False)
        self.checkBox_norm.setObjectName("checkBox_norm")
        self.checkBox_2 = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_2.setGeometry(QtCore.QRect(40, 340, 171, 20))
        self.checkBox_2.setObjectName("checkBox_2")
        self.checkBox_3 = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_3.setGeometry(QtCore.QRect(40, 380, 171, 20))
        self.checkBox_3.setObjectName("checkBox_3")
        self.checkBox_4 = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_4.setGeometry(QtCore.QRect(40, 420, 171, 20))
        self.checkBox_4.setObjectName("checkBox_4")
        self.label_14 = QtWidgets.QLabel(self.centralwidget)
        self.label_14.setGeometry(QtCore.QRect(40, 510, 391, 31))
        self.label_14.setAlignment(QtCore.Qt.AlignCenter)
        self.label_14.setObjectName("label_14")
        self.horizontalSlider = QtWidgets.QSlider(self.centralwidget)
        self.horizontalSlider.setEnabled(True)
        self.horizontalSlider.setGeometry(QtCore.QRect(210, 300, 160, 21))
        self.horizontalSlider.setMinimum(-5)
        self.horizontalSlider.setMaximum(5)
        self.horizontalSlider.setProperty("value", 1)
        self.horizontalSlider.setTracking(True)
        self.horizontalSlider.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider.setTickPosition(QtWidgets.QSlider.TicksAbove)
        self.horizontalSlider.setObjectName("horizontalSlider")
        self.horizontalSlider_2 = QtWidgets.QSlider(self.centralwidget)
        self.horizontalSlider_2.setEnabled(False)
        self.horizontalSlider_2.setGeometry(QtCore.QRect(210, 340, 160, 22))
        self.horizontalSlider_2.setMinimum(-5)
        self.horizontalSlider_2.setMaximum(5)
        self.horizontalSlider_2.setProperty("value", 1)
        self.horizontalSlider_2.setTracking(True)
        self.horizontalSlider_2.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_2.setTickPosition(QtWidgets.QSlider.TicksAbove)
        self.horizontalSlider_2.setObjectName("horizontalSlider_2")
        self.horizontalSlider_3 = QtWidgets.QSlider(self.centralwidget)
        self.horizontalSlider_3.setEnabled(False)
        self.horizontalSlider_3.setGeometry(QtCore.QRect(210, 380, 160, 22))
        self.horizontalSlider_3.setMinimum(-5)
        self.horizontalSlider_3.setMaximum(5)
        self.horizontalSlider_3.setProperty("value", 1)
        self.horizontalSlider_3.setTracking(True)
        self.horizontalSlider_3.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_3.setTickPosition(QtWidgets.QSlider.TicksAbove)
        self.horizontalSlider_3.setObjectName("horizontalSlider_3")
        self.horizontalSlider_4 = QtWidgets.QSlider(self.centralwidget)
        self.horizontalSlider_4.setEnabled(False)
        self.horizontalSlider_4.setGeometry(QtCore.QRect(210, 420, 160, 22))
        self.horizontalSlider_4.setMinimum(-5)
        self.horizontalSlider_4.setMaximum(5)
        self.horizontalSlider_4.setProperty("value", 1)
        self.horizontalSlider_4.setTracking(True)
        self.horizontalSlider_4.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_4.setTickPosition(QtWidgets.QSlider.TicksAbove)
        self.horizontalSlider_4.setObjectName("horizontalSlider_4")
        self.horizontalSlider_5 = QtWidgets.QSlider(self.centralwidget)
        self.horizontalSlider_5.setEnabled(False)
        self.horizontalSlider_5.setGeometry(QtCore.QRect(210, 460, 160, 22))
        self.horizontalSlider_5.setMinimum(-5)
        self.horizontalSlider_5.setMaximum(5)
        self.horizontalSlider_5.setProperty("value", 1)
        self.horizontalSlider_5.setTracking(True)
        self.horizontalSlider_5.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_5.setTickPosition(QtWidgets.QSlider.TicksAbove)
        self.horizontalSlider_5.setObjectName("horizontalSlider_5")
        self.horizontalSlider_6 = QtWidgets.QSlider(self.centralwidget)
        self.horizontalSlider_6.setGeometry(QtCore.QRect(200, 570, 160, 22))
        self.horizontalSlider_6.setMinimum(-5)
        self.horizontalSlider_6.setMaximum(5)
        self.horizontalSlider_6.setProperty("value", 1)
        self.horizontalSlider_6.setTracking(True)
        self.horizontalSlider_6.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_6.setTickPosition(QtWidgets.QSlider.TicksAbove)
        self.horizontalSlider_6.setObjectName("horizontalSlider_6")
        self.label_value_interp_1 = QtWidgets.QLabel(self.centralwidget)
        self.label_value_interp_1.setGeometry(QtCore.QRect(380, 300, 51, 21))
        self.label_value_interp_1.setAlignment(QtCore.Qt.AlignCenter)
        self.label_value_interp_1.setObjectName("label_value_interp_1")
        self.label_value_interp_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_value_interp_2.setGeometry(QtCore.QRect(380, 340, 51, 21))
        self.label_value_interp_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_value_interp_2.setObjectName("label_value_interp_2")
        self.label_value_interp_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_value_interp_3.setGeometry(QtCore.QRect(380, 380, 51, 21))
        self.label_value_interp_3.setAlignment(QtCore.Qt.AlignCenter)
        self.label_value_interp_3.setObjectName("label_value_interp_3")
        self.label_value_interp_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_value_interp_4.setGeometry(QtCore.QRect(380, 420, 51, 21))
        self.label_value_interp_4.setAlignment(QtCore.Qt.AlignCenter)
        self.label_value_interp_4.setObjectName("label_value_interp_4")
        self.label_value_interp_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_value_interp_5.setGeometry(QtCore.QRect(380, 460, 51, 21))
        self.label_value_interp_5.setAlignment(QtCore.Qt.AlignCenter)
        self.label_value_interp_5.setObjectName("label_value_interp_5")
        self.label_value_interp_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_value_interp_6.setGeometry(QtCore.QRect(380, 570, 51, 21))
        self.label_value_interp_6.setAlignment(QtCore.Qt.AlignCenter)
        self.label_value_interp_6.setObjectName("label_value_interp_6")
        self.horizontalSlider_7 = QtWidgets.QSlider(self.centralwidget)
        self.horizontalSlider_7.setGeometry(QtCore.QRect(199, 610, 161, 22))
        self.horizontalSlider_7.setMinimum(-5)
        self.horizontalSlider_7.setMaximum(5)
        self.horizontalSlider_7.setProperty("value", 1)
        self.horizontalSlider_7.setTracking(True)
        self.horizontalSlider_7.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_7.setTickPosition(QtWidgets.QSlider.TicksAbove)
        self.horizontalSlider_7.setObjectName("horizontalSlider_7")
        self.label_value_interp_7 = QtWidgets.QLabel(self.centralwidget)
        self.label_value_interp_7.setGeometry(QtCore.QRect(380, 610, 51, 21))
        self.label_value_interp_7.setAlignment(QtCore.Qt.AlignCenter)
        self.label_value_interp_7.setObjectName("label_value_interp_7")
        self.horizontalSlider_8 = QtWidgets.QSlider(self.centralwidget)
        self.horizontalSlider_8.setGeometry(QtCore.QRect(200, 650, 160, 22))
        self.horizontalSlider_8.setMinimum(-5)
        self.horizontalSlider_8.setMaximum(5)
        self.horizontalSlider_8.setProperty("value", 1)
        self.horizontalSlider_8.setTracking(True)
        self.horizontalSlider_8.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_8.setTickPosition(QtWidgets.QSlider.TicksAbove)
        self.horizontalSlider_8.setObjectName("horizontalSlider_8")
        self.label_value_interp_8 = QtWidgets.QLabel(self.centralwidget)
        self.label_value_interp_8.setGeometry(QtCore.QRect(380, 650, 51, 21))
        self.label_value_interp_8.setAlignment(QtCore.Qt.AlignCenter)
        self.label_value_interp_8.setObjectName("label_value_interp_8")
        self.horizontalSlider_9 = QtWidgets.QSlider(self.centralwidget)
        self.horizontalSlider_9.setGeometry(QtCore.QRect(200, 690, 160, 22))
        self.horizontalSlider_9.setMinimum(-5)
        self.horizontalSlider_9.setMaximum(5)
        self.horizontalSlider_9.setProperty("value", 1)
        self.horizontalSlider_9.setTracking(True)
        self.horizontalSlider_9.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_9.setTickPosition(QtWidgets.QSlider.TicksAbove)
        self.horizontalSlider_9.setObjectName("horizontalSlider_9")
        self.label_value_interp_9 = QtWidgets.QLabel(self.centralwidget)
        self.label_value_interp_9.setGeometry(QtCore.QRect(380, 690, 51, 21))
        self.label_value_interp_9.setAlignment(QtCore.Qt.AlignCenter)
        self.label_value_interp_9.setObjectName("label_value_interp_9")
        self.label_15 = QtWidgets.QLabel(self.centralwidget)
        self.label_15.setGeometry(QtCore.QRect(40, 571, 151, 20))
        self.label_15.setObjectName("label_15")
        self.label_16 = QtWidgets.QLabel(self.centralwidget)
        self.label_16.setGeometry(QtCore.QRect(40, 610, 151, 21))
        self.label_16.setObjectName("label_16")
        self.label_17 = QtWidgets.QLabel(self.centralwidget)
        self.label_17.setGeometry(QtCore.QRect(40, 690, 151, 21))
        self.label_17.setObjectName("label_17")
        self.label_18 = QtWidgets.QLabel(self.centralwidget)
        self.label_18.setGeometry(QtCore.QRect(40, 650, 151, 21))
        self.label_18.setObjectName("label_18")
        self.label_19 = QtWidgets.QLabel(self.centralwidget)
        self.label_19.setGeometry(QtCore.QRect(40, 250, 391, 31))
        self.label_19.setAlignment(QtCore.Qt.AlignCenter)
        self.label_19.setObjectName("label_19")
        self.line_3 = QtWidgets.QFrame(self.centralwidget)
        self.line_3.setGeometry(QtCore.QRect(440, 20, 21, 711))
        self.line_3.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_3.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_3.setObjectName("line_3")
        self.label_20 = QtWidgets.QLabel(self.centralwidget)
        self.label_20.setGeometry(QtCore.QRect(30, 10, 391, 31))
        self.label_20.setAlignment(QtCore.Qt.AlignCenter)
        self.label_20.setObjectName("label_20")
        self.line_4 = QtWidgets.QFrame(self.centralwidget)
        self.line_4.setGeometry(QtCore.QRect(0, 490, 451, 20))
        self.line_4.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_4.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_4.setObjectName("line_4")
        self.line_5 = QtWidgets.QFrame(self.centralwidget)
        self.line_5.setGeometry(QtCore.QRect(0, 240, 451, 20))
        self.line_5.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_5.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_5.setObjectName("line_5")
        self.label_21 = QtWidgets.QLabel(self.centralwidget)
        self.label_21.setGeometry(QtCore.QRect(480, 10, 611, 31))
        self.label_21.setAlignment(QtCore.Qt.AlignCenter)
        self.label_21.setObjectName("label_21")
        self.label_13 = QtWidgets.QLabel(self.centralwidget)
        self.label_13.setGeometry(QtCore.QRect(470, 106, 231, 20))
        self.label_13.setAlignment(QtCore.Qt.AlignCenter)
        self.label_13.setObjectName("label_13")
        self.checkBox_golos = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_golos.setGeometry(QtCore.QRect(70, 210, 301, 20))
        self.checkBox_golos.setObjectName("checkBox_golos")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1687, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        # endregion
        # region parameters

        self.matplotlib_canvas = MatplotlibCanvas(self.centralwidget)
        self.matplotlib_canvas.setGeometry(QtCore.QRect(1120, 290, 541, 431))

        self.clear_df_iris = pd.DataFrame()
        self.mod_df_iris = pd.DataFrame()
        self.clear_df_dop = pd.DataFrame()
        self.mod_df_dop = pd.DataFrame()
        # endregion parameters

        # region alex
        self.Load_data_button.clicked.connect(self.load_main_data)
        self.enter_console_button.clicked.connect(self.show_df_data)
        self.draw_button.clicked.connect(self.draw_button_click)
        self.predict_button.clicked.connect(self.predict_button_click)
        self.opt_k_button.clicked.connect(self.opt_k_button_click)

        #checkboxes
        self.checkBox_1.stateChanged.connect(self.checkBox_1_state_changed)
        self.checkBox_2.stateChanged.connect(self.checkBox_2_state_changed)
        self.checkBox_3.stateChanged.connect(self.checkBox_3_state_changed)
        self.checkBox_4.stateChanged.connect(self.checkBox_4_state_changed)
        self.checkBox_5.stateChanged.connect(self.checkBox_5_state_changed)

        #horizontal
        self.horizontalSlider.valueChanged.connect(self.horizontalSlider_1_value_changed)
        self.horizontalSlider_2.valueChanged.connect(self.horizontalSlider_2_value_changed)
        self.horizontalSlider_3.valueChanged.connect(self.horizontalSlider_3_value_changed)
        self.horizontalSlider_4.valueChanged.connect(self.horizontalSlider_4_value_changed)
        self.horizontalSlider_5.valueChanged.connect(self.horizontalSlider_5_value_changed)
        self.horizontalSlider_6.valueChanged.connect(self.horizontalSlider_6_value_changed)
        self.horizontalSlider_7.valueChanged.connect(self.horizontalSlider_7_value_changed)
        self.horizontalSlider_8.valueChanged.connect(self.horizontalSlider_8_value_changed)
        self.horizontalSlider_9.valueChanged.connect(self.horizontalSlider_9_value_changed)
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
        # self.data_choice_combobox.setItemText(1, _translate("MainWindow", "Заполненный dop.csv"))
        self.data_choice_combobox.setItemText(1, _translate("MainWindow", "Ансамбль классификаторов для iris.csv"))
        self.data_choice_combobox.setItemText(2, _translate("MainWindow",
                                                            "Использовать настройки приложения для dop.csv"))

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
        self.label_5.setText(_translate("MainWindow", "Введите свободные веса"))
        self.checkBox_1.setText(_translate("MainWindow", "Logistic Regression"))
        self.checkBox_5.setText(_translate("MainWindow", "GaussianNB"))
        self.checkBox_norm.setText(_translate("MainWindow", "Использовать нормализованные данные"))
        self.checkBox_2.setText(_translate("MainWindow", "Decision Tree Classifier"))
        self.checkBox_3.setText(_translate("MainWindow", "Random Forest Classifier"))
        self.checkBox_4.setText(_translate("MainWindow", "SVC"))
        self.label_14.setText(_translate("MainWindow", "Коэфиценты значимости перед признаками"))
        self.label_value_interp_1.setText(_translate("MainWindow", "1"))
        self.label_value_interp_2.setText(_translate("MainWindow", "None"))
        self.label_value_interp_3.setText(_translate("MainWindow", "None"))
        self.label_value_interp_4.setText(_translate("MainWindow", "None"))
        self.label_value_interp_5.setText(_translate("MainWindow", "None"))
        self.label_value_interp_6.setText(_translate("MainWindow", "1"))
        self.label_value_interp_7.setText(_translate("MainWindow", "1"))
        self.label_value_interp_8.setText(_translate("MainWindow", "1"))
        self.label_value_interp_9.setText(_translate("MainWindow", "1"))
        self.label_15.setText(_translate("MainWindow", "Sepal Lenght"))
        self.label_16.setText(_translate("MainWindow", "Sepal Widht"))
        self.label_17.setText(_translate("MainWindow", "Petal Widht"))
        self.label_18.setText(_translate("MainWindow", "Petal Lenght"))
        self.label_19.setText(_translate("MainWindow", "Настройка классфикаторов"))
        self.label_20.setText(_translate("MainWindow", "Основные настройки"))
        self.label_21.setText(_translate("MainWindow", "Область работы с данными"))
        self.label_13.setText(_translate("MainWindow", "Выберите данные:"))
        self.checkBox_golos.setText(_translate("MainWindow", "Использовать взвешенное голосование"))

    code_name_dict = {'0': "Setosa", '1': "Versicolor", '2': "Virginica"}
    code_FULLname_dict = {'0': "Iris-Setosa", '1': "Iris-Versicolor", '2': "Iris-Virginica"}

    #region simple functions
    def checkBox_1_state_changed(self):
        self.horizontalSlider.setEnabled(self.checkBox_1.isChecked())
        self.label_value_interp_1.setText( str(self.horizontalSlider.value()) if self.checkBox_1.isChecked() else "None")

    def checkBox_2_state_changed(self):
        self.horizontalSlider_2.setEnabled(self.checkBox_2.isChecked())
        self.label_value_interp_2.setText( str(self.horizontalSlider_2.value()) if self.checkBox_2.isChecked() else "None")

    def checkBox_3_state_changed(self):
        self.horizontalSlider_3.setEnabled(self.checkBox_3.isChecked())
        self.label_value_interp_3.setText( str(self.horizontalSlider_3.value()) if self.checkBox_3.isChecked() else "None")

    def checkBox_4_state_changed(self):
        self.horizontalSlider_4.setEnabled(self.checkBox_4.isChecked())
        self.label_value_interp_4.setText( str(self.horizontalSlider_4.value()) if self.checkBox_4.isChecked() else "None")

    def checkBox_5_state_changed(self):
        self.horizontalSlider_5.setEnabled(self.checkBox_5.isChecked())
        self.label_value_interp_5.setText( str(self.horizontalSlider_5.value()) if self.checkBox_5.isChecked() else "None")



    def horizontalSlider_1_value_changed(self):
        self.label_value_interp_1.setText(str(self.horizontalSlider.value()))

    def horizontalSlider_2_value_changed(self):
        self.label_value_interp_2.setText(str(self.horizontalSlider_2.value()))

    def horizontalSlider_3_value_changed(self):
        self.label_value_interp_3.setText(str(self.horizontalSlider_3.value()))

    def horizontalSlider_4_value_changed(self):
        self.label_value_interp_4.setText(str(self.horizontalSlider_4.value()))
    def horizontalSlider_5_value_changed(self):
        self.label_value_interp_5.setText(str(self.horizontalSlider_5.value()))
    def horizontalSlider_6_value_changed(self):
        self.label_value_interp_6.setText(str(self.horizontalSlider_6.value()))
    def horizontalSlider_7_value_changed(self):
        self.label_value_interp_7.setText(str(self.horizontalSlider_7.value()))
    def horizontalSlider_8_value_changed(self):
        self.label_value_interp_8.setText(str(self.horizontalSlider_8.value()))
    def horizontalSlider_9_value_changed(self):
        self.label_value_interp_9.setText(str(self.horizontalSlider_9.value()))

    # endregion

    def show_df_data(self):
        number_data_frame = self.data_choice_combobox.currentText()
        match number_data_frame:
            case "Исходный iris.csv":
                self.main_textedit.setText(self.clear_df_iris.to_string())

            case "Ансамбль классификаторов для iris.csv":
                some_str = str(self.fit_any_classificators())
                if len(some_str) < 10:
                    self.main_textedit.setText("Некая ошибка")
                else:
                    some_str += "\n"
                    some_str += self.clear_df_iris.to_string()

                    self.main_textedit.setText(some_str)
            case "Использовать настройки приложения для dop.csv":
                output_str_tmp = "Убедитесь что вы поставили правильные настройки:\n"
                model1 = self.fit_our_model()
                df_dop = self.clear_df_dop.copy()
                df_dop["Kind"] = model1.predict(df_dop.values, 1, self.checkBox_golos.isChecked())
                df_dop['Kind'] = df_dop['Kind'].astype(str).replace(self.code_FULLname_dict)
                self.mod_df_dop = df_dop
                output_str_tmp += df_dop.to_string()
                self.main_textedit.setText(output_str_tmp)
            case _:
                self.show_popup_critical("Ошибка с файлом")

    def load_main_data(self):
        file_path = "data/iris.csv"
        df = pd.read_csv(file_path, sep=",")
        self.clear_df_iris = df

        df2 = pd.read_csv(file_path, sep=",")
        df2['num_kind'] = pd.factorize(df2['Kind'])[0]
        self.mod_df_iris = df2

        file_path = "data/Dop.csv"
        df_dop = pd.read_csv(file_path, sep=",")

        self.clear_df_dop = df_dop

        self.show_popup_information("Успех", "Данные успешно загружены!")

    def draw_button_click(self):
        if self.clear_df_iris.empty:
            self.show_popup_critical("Ошибка", "Загрузите данные прежде чем нажать на кнопку")

        else:
            text1 = self.comboBox_3.currentText()
            text2 = self.comboBox_4.currentText()

            if text1 == text2:
                self.show_popup_critical("Ошибка", "Выберите разные признаки для сравнения")
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
            prediction = model.predict(new_X, 1)

            type_iris = f"Iris-{self.code_name_dict[str(prediction[0])]}"

            self.show_popup_information("Шаманим", "Спрашиваю у Ванги что за ирис вы записали, \nподождите немного.")
            time.sleep(3)
            self.show_popup_information("Предсказание", f"Ваш ирис это {type_iris}")

    @staticmethod
    def calculate_accuracy(true_labels, predicted_labels):

        if len(true_labels) != len(predicted_labels):
            raise ValueError("Длины массивов должны совпадать.")

        correct = 0
        total = len(true_labels)

        for true, predicted in zip(true_labels, predicted_labels):
            if true == predicted:
                correct += 1

        accuracy = correct / total * 100
        return accuracy

    def fit_any_classificators(self):
        if ((self.checkBox_1.isChecked() + self.checkBox_2.isChecked() + self.checkBox_3.isChecked() +
             self.checkBox_4.isChecked() + self.checkBox_5.isChecked()) == 0):
            self.show_popup_critical("Ошибка", "Выберите хотя бы один классификатор")
        else:
            models_arr = []
            my_weights = []
            output_str = "Выбраны следующие классификаторы:"
            if self.checkBox_1.isChecked():
                models_arr.append(('model1', LogisticRegression()))
                my_weights.append(self.horizontalSlider.value())
                output_str += "\nLogisticRegression"
            if self.checkBox_2.isChecked():
                models_arr.append(('model2', DecisionTreeClassifier()))
                my_weights.append(self.horizontalSlider_2.value())
                output_str += "\nDecisionTreeClassifier"
            if self.checkBox_3.isChecked():
                models_arr.append(('model3', RandomForestClassifier()))
                my_weights.append(self.horizontalSlider_3.value())
                output_str += "\nRandomForestClassifier"
            if self.checkBox_4.isChecked():
                models_arr.append(('model4', SVC(probability=True)))
                my_weights.append(self.horizontalSlider_4.value())
                output_str += "\nSVC"
            if self.checkBox_5.isChecked():
                models_arr.append(('model5', GaussianNB()))
                my_weights.append(self.horizontalSlider_5.value())
                output_str += "\nGaussianNB"

            # Получаем данные и делаем выборки
            X = self.mod_df_iris.iloc[:, :4].copy()
            y = self.mod_df_iris.iloc[:, 5].to_frame().copy()
            X = np.array(X)
            y = np.array(y)
            y = y.flatten()

            # Нормализируем если нужно
            scaler = StandardScaler()
            if self.checkBox_norm.isChecked():
                output_str += "\nДанные нормализованы"
                scaler.fit(X)
                X = scaler.transform(X)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            voting_classifier = VotingClassifier(estimators=models_arr, voting='soft', weights=my_weights)

            voting_classifier.fit(X_train, y_train)

            y_pred = voting_classifier.predict(X_test)

            accuracy = self.calculate_accuracy(y_test, y_pred)
            output_str += f"\nТочность = {accuracy}"
            return output_str

    def fit_our_model(self):
        """
        :return: получаем обученную модель  с которой можно все делать
        """

        # Получаем данные и делаем выборки
        X = self.mod_df_iris.iloc[:, :4].copy()
        y = self.mod_df_iris.iloc[:, 5].to_frame().copy()
        X = np.array(X)
        y = np.array(y)
        y = y.flatten()

        # Нормализируем если нужно
        scaler = StandardScaler()
        if self.checkBox_3.isChecked():
            scaler.fit(X)
            X = scaler.transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        weights = np.array([
            self.horizontalSlider_6.value(),
            self.horizontalSlider_7.value(),
            self.horizontalSlider_8.value(),
            self.horizontalSlider_9.value()
        ])

        count_k = self.count_k_spinBox.value()

        custom_knn = CustomKNN(count_k)
        custom_knn.fit(X_train, y_train, weights)

        return custom_knn

    def plot_k_drawing(self, k_history_inner, accuracy_history):
        # Получение объекта Axes
        ax =self.matplotlib_canvas.fig.add_subplot(111)

        title_text = "График зависимости на основе метрики: "
        choice_metric_param = self.metric_combobox.currentIndex() + 1
        if choice_metric_param == 1:
            title_text += "Евклидова"
        elif choice_metric_param == 2:
            title_text += "Хемминга"
        elif choice_metric_param == 3:
            title_text += "Чебышева"
        elif choice_metric_param == 4:
            title_text += "Косинусная"

        ax.set_title(title_text)
        # Построение графика
        ax.plot(k_history_inner, accuracy_history)

        # Установка меток для осей (по желанию)
        ax.set_xlabel('Количество соседей')
        ax.set_ylabel('Точность')


    def opt_k_button_click(self):
        if self.clear_df_iris.empty:
            self.show_popup_critical("Ошибка", "Загрузите данные прежде чем нажать на кнопку")
        else:
            X = self.mod_df_iris.iloc[:, :4].copy()
            y = self.mod_df_iris.iloc[:, 5].to_frame().copy()

            X = np.array(X)
            y = np.array(y)
            y = y.flatten()

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            output_log = "Выбранная метрика:"
            choice_metric_param =self.metric_combobox.currentIndex()+1
            if choice_metric_param==1:
                output_log +="\nЕвклидова\n"
            elif choice_metric_param==2:
                output_log +="\nХемминга\n"
            elif choice_metric_param==3:
                output_log +="\nЧебышева\n"
            elif choice_metric_param==4:
                output_log +="\nКосинусная\n"
            maxim_value = 0

            weights_tmp = np.array([
                self.horizontalSlider_6.value(),
                self.horizontalSlider_7.value(),
                self.horizontalSlider_8.value(),
                self.horizontalSlider_9.value()
            ])
            accuracy_history=[]
            k_history =[]
            for key in range(1, 150):
                custom_knn = CustomKNN(key)
                custom_knn.fit(X_train, y_train, weights_tmp)

                y_pred = custom_knn.predict(X_test, choice_metric_param, self.checkBox_golos.isChecked())
                count = 0
                for i in range(len(y_pred)):
                    if y_pred[i] == y_test[i]:
                        count += 1

                tmp_value = round(count / len(y_pred) * 100, 2)
                accuracy_history.append(tmp_value)
                k_history.append(key)
                output_log += f"Точность, при k={key}  >>> {tmp_value}%\n"
                if tmp_value >= maxim_value:
                    maxim_value = tmp_value
                    best_k = key


            messagebox_log = f"Самый оптимальный k={best_k}, с точностью {maxim_value}"
            output_log += messagebox_log
            self.main_textedit.setText(output_log)
            self.show_popup_information("Оптимальное количество соседей", messagebox_log)

            # Очистка предыдущего графика
            self.matplotlib_canvas.fig.clear()

            # Вызов вашей функции для построения графика
            self.plot_k_drawing(k_history,accuracy_history)
            # Обновление виджета с новым графиком
            self.matplotlib_canvas.draw()

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
