# Due to incorrect subprocess termination, start with:
# pkill -f live.py; sleep 0.5; python3.6 live.py; sleep 1; pkill -f live.py; echo "Killed procs"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.fftpack
from scipy import signal
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Button
from matplotlib.widgets import TextBox
from matplotlib.widgets import CheckButtons
import pdb
from pdb import set_trace
# import pty
# from tkinter import *
import math, random, threading, time, os, sys, queue, _thread
import multiprocessing as mp
import queue
from pyqtgraph.Qt import QtCore, QtGui
from PyQt5.QtCore import pyqtSlot
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from threading import Thread
import struct
import array

import platform
import serial

# def set_qt_trace():
#     QtCore.pyqtRemoveInputHook()
#     set_trace()

# suppresses the use of scientific notation for small numbers when printing np.array
np.set_printoptions(suppress=True)

if platform.system() == "Darwin":
    default_usb_port = "/dev/cu.usbmodem7913201"
else:
    default_usb_port = "COM5"

# baudrate = 46080004
baudrate = 2000000

# simulate = True
simulate = False

class App(QtGui.QMainWindow):
    num_updates_after_launch = 0
    data_gyro_x = None
    data_gyro_y = None
    data_gyro_z = None

    data_acc_x = None
    data_acc_y = None
    data_acc_z = None

    gyro_point_x = None
    gyro_point_y = None
    gyro_point_z = None

    def __init__(self, parent=None):
        super(App, self).__init__(parent)

        #### Create Gui Elements ###########
        self.mainbox = QtGui.QWidget()
        screen_resolution = app.desktop().screenGeometry()
        self.setFixedSize(screen_resolution.width(), int(screen_resolution.height() * 0.936)) # 0.636))
        self.setCentralWidget(self.mainbox)
        self.mainbox.setLayout(QtGui.QGridLayout())

        # QGridLayout#addWidget(): (QWidget * widget, int fromRow, int fromColumn, int rowSpan, int columnSpan, Qt::Alignment alignment = 0)

        self.canvas = pg.GraphicsLayoutWidget()
        self.mainbox.layout().addWidget(self.canvas, 0, 0, 0, 3) # last param = number of buttons + 1

        # Gyro plots
        self.gyro_plot = self.canvas.addPlot(row=0, col=0, colspan=2)
        self.gyro_x_series = self.gyro_plot.plot(pen='r')
        self.gyro_y_series = self.gyro_plot.plot(pen='g')
        self.gyro_z_series = self.gyro_plot.plot(pen='b')

        self.acc_plot = self.canvas.addPlot(row=1, col=0, colspan=2)
        self.acc_x_series = self.acc_plot.plot(pen='r')
        self.acc_y_series = self.acc_plot.plot(pen='g')
        self.acc_z_series = self.acc_plot.plot(pen='b')

        self.data_gyro_x = []
        self.data_gyro_y = []
        self.data_gyro_z = []

        self.data_acc_x = []
        self.data_acc_y = []
        self.data_acc_z = []

        # self.label = QtGui.QLabel()
        # self.mainbox.layout().addWidget(self.label, 0, 0, 1, 1, QtCore.Qt.AlignTop)
        # self.label.setFixedWidth(250)

        self.canvas.nextRow()

        self.init3d_view()

        # self.canvas.nextRow()

        self.save_button = QtGui.QPushButton("Save")
        self.mainbox.layout().addWidget(self.save_button, 2, 1)

        self.zoom_button = QtGui.QPushButton("Zoom")
        self.mainbox.layout().addWidget(self.zoom_button, 2, 2)

        self.plot_all_button = QtGui.QPushButton("Plot all")
        self.mainbox.layout().addWidget(self.plot_all_button, 2, 3)

        self.pause_button = QtGui.QPushButton("Pause")
        self.mainbox.layout().addWidget(self.pause_button, 2, 4)

        # Start
        self.update()
        self.update_slower()

        # Start reading the serial port
        self.init_serial()

    def init3d_view(self):
        view = gl.GLViewWidget()
        # view.show()
        self.mainbox.layout().addWidget(view, 0, 3, 2, 3) # last param = number of buttons + 1

        ## create three grids, add each to the view
        xgrid = gl.GLGridItem()
        ygrid = gl.GLGridItem()
        zgrid = gl.GLGridItem()
        view.addItem(xgrid)
        view.addItem(ygrid)
        view.addItem(zgrid)

        ## rotate x and y grids to face the correct direction
        xgrid.rotate(90, 0, 1, 0)
        ygrid.rotate(90, 1, 0, 0)

        ## scale each grid differently
        xgrid.scale(0.2, 0.1, 0.1)
        ygrid.scale(0.2, 0.1, 0.1)
        zgrid.scale(0.1, 0.2, 0.1)

        self.gyro_point = gl.GLMeshItem(meshdata=gl.MeshData.sphere(radius=0.25, rows=16, cols=16), color=(0.0, 0.7, 0.0, 1.0))
        self.gyro_point.translate(0, 0, 0)
        view.addItem(self.gyro_point)

        self.acc_point = gl.GLMeshItem(meshdata=gl.MeshData.sphere(radius=0.25, rows=16, cols=16), color=(0.7, 0.0, 0.0, 1.0))
        self.acc_point.translate(0, 0, 0)
        view.addItem(self.acc_point)
        # gx.rotate(90, 0, 1, 0)


    def init_serial(self):
        self.serial = serial.Serial(default_usb_port, baudrate=baudrate)
        self.serial_thread = ReadSerialThread(self)
        self.serial_thread.start()

    def before_close(self):
        self.serial_thread.is_running = False
        if self.serial.is_open:
            self.serial.close()
        #self.serial_thread.join()

    def closeEvent(self, event):
        print("Exiting gracefully...")

        try:
            self.before_close()
        except BaseException as e:
            print(e)

        event.accept()

    def keyPressEvent(self, event):
        # Exit on "q"
        if not event.isAutoRepeat():
            if event.text() == 'q':
                print("Quitting...")
                self.before_close()
                sys.exit(-1)

    def update(self):
        try:
            x_limit = 1000
            timescale = np.arange(0, np.min([x_limit, len(self.data_gyro_x)]))
            # print(len(timescale))
            self.gyro_x_series.setData(timescale, self.data_gyro_x[-x_limit:])
            self.gyro_y_series.setData(timescale, self.data_gyro_y[-x_limit:])
            self.gyro_z_series.setData(timescale, self.data_gyro_z[-x_limit:])

            self.acc_x_series.setData(timescale, self.data_acc_x[-x_limit:])
            self.acc_y_series.setData(timescale, self.data_acc_y[-x_limit:])
            self.acc_z_series.setData(timescale, self.data_acc_z[-x_limit:])

            # Gyro updatre
            self.gyro_point.resetTransform()

            self.gyro_point_x = self.data_gyro_x[-1] * 0.2
            self.gyro_point_y = self.data_gyro_y[-1] * 0.2
            self.gyro_point_z = self.data_gyro_z[-1] * 0.2

            self.gyro_point.translate(self.gyro_point_x, self.gyro_point_y, self.gyro_point_z)

            # Acc update            
            self.acc_point.resetTransform()
            
            acc_scale = 2 # scale of the original vector when displayed

            # self.acc_point_x = (self.data_acc_x[-1] - 512) / 512 * 6
            # self.acc_point_y = (self.data_acc_y[-1] - 512) / 512 * 6
            # self.acc_point_z = (self.data_acc_z[-1] - 512) / 512 * 6
            self.acc_point_x = self.data_acc_x[-1] * acc_scale
            self.acc_point_y = self.data_acc_y[-1] * acc_scale
            self.acc_point_z = self.data_acc_z[-1] * acc_scale            

            self.acc_point.translate(self.acc_point_x, self.acc_point_y, self.acc_point_z)
            # self.gyro_point.translate(0.05, 0, 0)

            self.num_updates_after_launch += 1
            QtCore.QTimer.singleShot(1, self.update)
        except KeyboardInterrupt:
            print("Exiting gracefully...")
            # self.decode_thread.join()
            # self.filter_thread.join()
            # self.update_process.terminate()
            # self.multi_pulse_process.terminate()
        except BaseException as e:
            print("update thread: %s" % str(e))
            QtCore.QTimer.singleShot(1, self.update)

    def update_slower(self):
        try:
            pass
        except BaseException as e:
            print(e)
        finally:
            QtCore.QTimer.singleShot(33, self.update_slower)


class ReadSerialThread(Thread):
    is_running = False

    def __init__(self, app):
        Thread.__init__(self)
        self.app = app

    def run(self):
        self.is_running = True

        # Data row format:
        #    0       1     2      3      4     5     6    7
        # CONTROL GYRO_X GYRO_Y GYRO_Z ACC_X ACC_Y ACC_Z EOL
        struct_format_string = "cffffff"
        rows_in_buffer = 1

        max_records_before_reset = 10000

        while self.is_running is True and main_thread.is_alive():
            try:
                if len(self.app.data_gyro_x) > max_records_before_reset:
                    self.app.data_gyro_x = []
                    self.app.data_gyro_y = []
                    self.app.data_gyro_z = []
                    self.app.data_acc_x = []
                    self.app.data_acc_y = []
                    self.app.data_acc_z = []
            except BaseException as e:
                print(str(e))

            try:
                line = self.app.serial.readline()

                if len(line) >= 26:
                    if line[0:1] is b'D':
                        # Decode binary
                        data = struct.unpack_from("=" + (struct_format_string * rows_in_buffer), line)
                        self.app.data_gyro_x.append(data[1])
                        self.app.data_gyro_y.append(data[2])
                        self.app.data_gyro_z.append(data[3])

                        self.app.data_acc_x.append(data[4])
                        self.app.data_acc_y.append(data[5])
                        self.app.data_acc_z.append(data[6])
                        # print(data)
                    elif line[0:1] == b'S':
                        # Decode string (for debugging byte-encoding)
                        # Sending of 'S' lines must be disabled in device code
                        print(str(line))
            except BaseException as e:
                print("[ERROR]: %s" % (str(e)))
                print(e)

            time.sleep(0.01)

        self.is_running = False

main_thread = threading.currentThread()

if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    thisapp = App()
    thisapp.show()
    sys.exit(app.exec_())

# TODO: Properly exit subprocs