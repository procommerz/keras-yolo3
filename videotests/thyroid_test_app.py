from PyQt5 import QtGui
import numpy as np
import math, random, threading, time, os, sys, queue, _thread

from PyQt5 import QtCore, QtGui
from PyQt5.QtCore import pyqtSlot, pyqtSignal, QThread, Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QMainWindow, QVBoxLayout, QLabel, QHBoxLayout, QWidget
import pyqtgraph as pg
from threading import Thread

class ThyroidTestAppThread(QThread):
    main_video_update = pyqtSignal(QImage)
    feature_frame_update = pyqtSignal(str, QImage, QImage)

    def __init__(self, parent=None, app=None):
        super(ThyroidTestAppThread, self).__init__(parent)
        self.app = app

    def run(self):
        self.app.testapp.load_everything()

        self.app.testapp.qt_main_video_update_signal = self.main_video_update
        self.app.testapp.qt_feature_frame_update_signal = self.feature_frame_update

        self.app.testapp.run()

class ThyroidTestApp(QMainWindow):
    def __init__(self, parent=None, app=None, testapp=None):
        super(ThyroidTestApp, self).__init__(parent)
        self.setStyleSheet("background-color: #2a2a2a; color: #d1d1d1;")

        self.testapp = testapp

        self.qt_main = QtGui.QWidget()

        screen_resolution = app.desktop().screenGeometry()
        #self.move(150, 150)
        self.setFixedSize(screen_resolution.width(), int(screen_resolution.height() * 0.936)) # 0.636))
        self.setCentralWidget(self.qt_main)
        self.qt_main.setLayout(QHBoxLayout())

        self.image_label = QLabel(self)
        self.image_label.setText("Loading may take a few seconds...")
        # self.image_label.move(280, 120)
        # self.image_label.resize(320, 240)
        self.qt_main.layout().addWidget(self.image_label, alignment=QtCore.Qt.AlignTop)

        self.qt_main.setStyleSheet("border: 1px solid #515151; vertical-align: top;")

        self.feature_images = dict() # placeholder for QLabel items for original feature images
        self.processed_feature_images = dict() # placeholder for QLabel items for processed feature images, that are displayed below originals

        n = 1
        for key in self.testapp.track_classes:
            vbox = QVBoxLayout()
            vbox.addStretch(1)

            self.feature_images[key] = QLabel(self)
            self.feature_images[key].setText("...")
            self.feature_images[key].resize(320, 240)

            self.processed_feature_images[key] = QLabel(self)
            self.processed_feature_images[key].setText("...")
            self.processed_feature_images[key].resize(320, 240)
            # self.qt_main.layout().addWidget(self.feature_images[key], 0, n)
            label = QLabel(self)
            label.setText(key)
            label.setStyleSheet("vertical-align: top;")
            vbox.addWidget(label, alignment=QtCore.Qt.AlignTop)
            vbox.addWidget(self.feature_images[key], alignment=QtCore.Qt.AlignTop)
            vbox.addWidget(self.processed_feature_images[key], alignment=QtCore.Qt.AlignTop)
            vbox.setAlignment(QtCore.Qt.AlignTop)

            widget = QWidget()
            widget.setLayout(vbox)

            # self.qt_main.layout().addLayout(vbox) # , alignment = QtCore.Qt.AlignTop
            self.qt_main.layout().addWidget(widget, alignment=QtCore.Qt.AlignTop)
            n += 1

        self.qt_main.layout().setAlignment(QtCore.Qt.AlignTop)

        self.testapp_thread = ThyroidTestAppThread(self, app=self)
        self.testapp_thread.main_video_update.connect(self.set_image)
        self.testapp_thread.feature_frame_update.connect(self.set_feature_image)
        self.testapp_thread.start()

        self.setWindowTitle("Thyroid Metrics Test")
        self.update()

    def update(self):
        QtCore.QTimer.singleShot(100, self.update)

    def keyPressEvent(self, event):
        # Exit on "q"
        if not event.isAutoRepeat():
            if event.text() == 'q':
                sys.exit(-1)

    def set_image(self, image):
        if self.windowState() & Qt.WindowMinimized: # Do not update minimized
            return

        self.image_label.setPixmap(QPixmap.fromImage(image))

    def set_feature_image(self, class_name, image, processed_image):
        if self.windowState() & Qt.WindowMinimized: # Do not update minimized
            return

        self.feature_images[class_name].setPixmap(QPixmap.fromImage(image))
        self.processed_feature_images[class_name].setPixmap(QPixmap.fromImage(processed_image))