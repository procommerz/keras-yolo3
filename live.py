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
import serial
import pty
# from tkinter import *
import math, random, threading, time, os, sys, queue, _thread
import multiprocessing as mp
import queue
from pyqtgraph.Qt import QtCore, QtGui
from PyQt5.QtCore import pyqtSlot
import pyqtgraph as pg
import struct
import array

# def set_qt_trace():
#     QtCore.pyqtRemoveInputHook()
#     set_trace()

# suppresses the use of scientific notation for small numbers when printing np.array
np.set_printoptions(suppress=True)

default_usb_port = "/dev/cu.usbmodem7913201"
# default_usb_port = "/dev/cu.usbmodem53923101"

SIGNAL_KEY = 'data'
ECHO_KEY = 'data2'

# baudrate = 46080004
baudrate = 2000000

# Some constants for sampling frequency (and pulse frequency) from the microchip sketch
reference_sf = 206651
reference_pf = reference_sf / 2

speed_of_sound_meters_per_sec = 355.0
# speed_of_sound_meters_per_sec = 1455.0
speed_of_sound_mm_per_sec = speed_of_sound_meters_per_sec * 1000.0
speed_of_sound_mm_per_usec = speed_of_sound_mm_per_sec / (10**6)

normalize_max_distance_mm = 300

print("Speed of sound, mm per us: %f" % speed_of_sound_mm_per_usec)
ten_cm_time_us = 100 / speed_of_sound_mm_per_usec
normalize_max_distance_time_us = normalize_max_distance_mm / speed_of_sound_mm_per_usec
print("10 cm ~ %f us (%f ms)" % (ten_cm_time_us, ten_cm_time_us / 1000))
print("%d mm ~ %f us (%f ms)" % (normalize_max_distance_mm, normalize_max_distance_time_us, normalize_max_distance_time_us / 1000))

# simulate = True
simulate = False

class App(QtGui.QMainWindow):
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
        self.mainbox.layout().addWidget(self.canvas, 0, 1, 1, 6) # last param = number of buttons + 1


        # self.view = self.canvas.addViewBox()
        # self.view.setAspectLocked(True)
        # self.view.setRange(QtCore.QRectF(0,0, 100, 100))

        #  line plot
        self.signal_plot_root = self.canvas.addPlot(row=0, col=0, colspan=2)
        self.signal_plot = self.signal_plot_root.plot(pen='g')
        # self.signal_plot_root.setYRange(-2048, 2048, padding=0)

        self.signal_plot2 = None
        self.signal_plot2_root = self.canvas.addPlot(row=1, col=0, colspan=2)
        self.signal_plot2 = self.signal_plot2_root.plot(pen='FF00FF')
        self.pulse_on_echo_plot = self.signal_plot2_root.plot(pen='00FF00')
        # self.signal_plot2_root.setYRange(-2048, 2048, padding=0) # leave autopadding for echoes

        self.echo_peak_amp_line = pg.InfiniteLine(pos=0, movable=False,label="peak lvl",angle=0)
        self.signal_plot2_root.addItem(self.echo_peak_amp_line)
        # self.echo_lines.append(line)
        # self.signal_plot2_root.addItem(line)

        # self.fft_plot = self.canvas.addPlot(row=2, col=0, colspan=2)
        # self.fft_plot = self.fft_plot.plot(pen='b')

        # ftt plot
        self.filtered_plot_root = self.canvas.addPlot(row=3, col=0, colspan=2)
        self.filtered_plot = self.filtered_plot_root.plot(pen='y')
        self.echo_filt_peak_amp_line = pg.InfiniteLine(pos=0, movable=False, label="peak lvl", angle=0)
        self.filtered_plot_root.addItem(self.echo_filt_peak_amp_line)

        self.label = QtGui.QLabel()
        self.mainbox.layout().addWidget(self.label, 0, 0, 1, 1, QtCore.Qt.AlignTop)
        self.label.setFixedWidth(250)

        # self.canvas.nextRow()

        self.save_button = QtGui.QPushButton("Save")
        self.mainbox.layout().addWidget(self.save_button, 1, 1)

        self.zoom_button = QtGui.QPushButton("Zoom")
        self.mainbox.layout().addWidget(self.zoom_button, 1, 2)

        self.plot_all_button = QtGui.QPushButton("Plot all")
        self.mainbox.layout().addWidget(self.plot_all_button, 1, 3)

        self.pause_button = QtGui.QPushButton("Pause")
        self.mainbox.layout().addWidget(self.pause_button, 1, 4)

        self.toggle_pulse_button = QtGui.QPushButton("Tog. Pulse")
        self.mainbox.layout().addWidget(self.toggle_pulse_button, 1, 5)

        self.one_pulse_button = QtGui.QPushButton("Pulse")
        self.mainbox.layout().addWidget(self.one_pulse_button, 1, 6)

        # btn = pg.ColorButton()
        # self.view.add

        #### Image view ####################

        self.init_img()

        # QGridLayout#addWidget(): (QWidget * widget, int fromRow, int fromColumn, int rowSpan, int columnSpan, Qt::Alignment alignment = 0)
        # self.mainbox.layout().addWidget(self.canvas, 0, 1, 1, 6)  # last param = number of buttons + 1

        #### Set Data  #####################
        self.initial_padding_size = 6000
        # plt.style.use('ggplot')
        # data = np.array([512] * self.initial_padding_size)
        # data2 = np.array([512] * self.initial_padding_size)
        # # timestamps = np.linspace(0, 5999 * 636, 6000)
        # timestamps = np.linspace(-6000 * 40, 0, self.initial_padding_size)
        data = np.array([])
        data2 = np.array([])
        timestamps = np.array([])

        self.data_dict = {SIGNAL_KEY: data, ECHO_KEY: data2, 'timestamps': timestamps }
        self.data_dict['ftt'] = np.array([0] * 6000)
        self.data_dict['interp1'] = []
        self.data_dict['interp2'] = []

        self.echo_lines = []
        self.collected_echoes = []

        self.metrics_established = False # will become true when signal metrics are established at the start of the app

        self.signal_queue = mp.Queue()
        self.control_queue = mp.Queue()
        self.info_queue = mp.Queue()
        self.signal_pipe_recv, self.signal_pipe_snd = mp.Pipe()

        self.liveplot = LivePlotter(self.data_dict, self.signal_queue, self.control_queue, self.info_queue, self)

        self.counter = 0
        self.fps = 0.
        self.lastupdate = time.time()

        # Button events:
        self.pause_button.clicked.connect(self.liveplot.on_pause_click)
        self.zoom_button.clicked.connect(self.liveplot.on_func3_click)
        self.plot_all_button.clicked.connect(self.liveplot.on_plot_all)
        self.save_button.clicked.connect(self.liveplot.on_save_click)
        self.toggle_pulse_button.clicked.connect(self.liveplot.on_toggle_pulse_click)
        self.one_pulse_button.clicked.connect(self.liveplot.on_one_pulse_click)
        
        #### Start Subprocesses #####################
        if simulate is not True:
            self.liveplot.serial_proc = self.update_thread = LivePlotProc(self.liveplot, self.signal_queue, self.control_queue, self.info_queue, self.signal_pipe_snd)
            self.decode_thread = LivePlotDecodeThread(self.signal_plot, self.data_dict, self.signal_queue, self.info_queue, self.signal_pipe_recv, self.liveplot)
        else:
            self.decode_thread = None
            self.simfeed_thread = SimFeedThread(self.data_dict, "data/liverec14_gel.csv", batchsize = 1000, delay = 0.0005) # For simulated feed from a recording

        self.filter_thread = FilterThread(self.data_dict, self.liveplot)

        self.ftt_update_counter = 0
        self.base_update_counter = 0

        self.start_time = time.time()

        #### Start  #####################
        self.update()
        self.update_slower()

    def init_img(self):
        self.img_scanlines = pg.ImageView()
        self.mainbox.layout().addWidget(self.img_scanlines, 2, 0, 1, 7)

        self.image_width = 500
        self.image_height = 500
        # self.transposed_scanlines = np.zeros([1, 300, 300])
        self.transposed_scanlines = np.full([1, self.image_width, self.image_height], 0.0)

        self.img_scanlines.setImage(self.transposed_scanlines,
                                    xvals=np.linspace(1., 3., self.transposed_scanlines.shape[0]))

        self.img_scanlines.autoRange()

        self.current_scanline = 0
        self.max_scanline = self.transposed_scanlines.shape[1]

    def update_charts(self):
        # Wait for the lock
        if self.decode_thread is not None:
            while self.decode_thread.insert_locked is True:
                pass

            self.decode_thread.insert_locked = True

        new_x = self.data_dict['timestamps'][-self.liveplot.signal_zoom:]
        # line1.set_xdata(new_x)
        # signal_plot.set_xlim([new_x[0], new_x[-1]])

        self.timeline_fx = int(new_x[0])
        self.timeline_lx = int(new_x[-1])

        # TX line
        new_y = self.data_dict[SIGNAL_KEY][-self.liveplot.signal_zoom:]
        self.signal_plot.setData(new_x, new_y)

        # if liveplot.auto_y:
        #     signal_plot.set_ylim(
        #         [round(min(new_y / 50)) * 50 - 25, round(max(new_y / 50)) * 50 + 25])  # update limit channel
        # else:
        #     signal_plot.set_ylim([0, 4096])

        if self.signal_plot2 is not None:
            # RX line
            if len(self.data_dict[ECHO_KEY]) > self.liveplot.signal_zoom:
                new_y2 = self.data_dict[ECHO_KEY][-self.liveplot.signal_zoom:]
                self.signal_plot2.setData(new_x, new_y2)

                # Transform the pulse signal to match the echo range,
                # since the chart series will be combined on one scale
                pulses_on_echoes = np.copy(new_y)
                pulses_on_echoes[np.where(pulses_on_echoes == 0)] = self.liveplot.min_echo
                pulses_on_echoes[np.where(pulses_on_echoes == 1)] = self.liveplot.max_echo
                # self.pulse_on_echo_plot.setData(new_x, pulses_on_echoes)

        if self.decode_thread is not None:
            self.decode_thread.insert_locked = False

        # FTT
        # if self.liveplot.current_ftt_abs is not None:
        #     self.fft_plot.setData(self.liveplot.current_freqs_range, self.liveplot.current_ftt_abs)
        #
        if self.filtered_plot is not None:
            filtered_y2 = self.filter_echoes(new_x, new_y2)
            self.filtered_plot.setData(new_x, filtered_y2)
        # self.signal_plot.setData(self.xdata, self.ydata)
        # self.fft_plot.setData(self.xdata, self.ydata)

    def update_charts_after_image_update(self, timestamp_window):
        print("Echoes %d" % len(self.collected_echoes))
        if (hasattr(self, 'timeline_fx')):
            earliest_visible_t = self.timeline_fx
        else:
            earliest_visible_t = self.data_dict['timestamps'][-self.liveplot.signal_zoom:][0]

        # for l in self.echo_lines:
        #     self.signal_plot2_root.removeItem(l)
        #
        # for echo in self.collected_echoes[-100:]:
        #     # print("fwt: %s, et: %s" % (earliest_visible_t, echo))
        #     line = pg.InfiniteLine(pos=timestamp_window[echo], movable=False)
        #     self.echo_lines.append(line)
        #     self.signal_plot2_root.addItem(line)

        # self.signal_plot2_root.addItem(pg.InfiniteLine(pos=self.data_dict['timestamps'][-1], movable=False))

        if (hasattr(self, 'echo_peak_amp')) and self.echo_peak_amp is not None:
            self.echo_peak_amp_line.setPos(self.echo_peak_amp)
            self.echo_filt_peak_amp_line.setPos(self.echo_peak_amp)

    def establish_metrics(self):
        self.metrics_established = True

        window = 24000
        # TODO: Check the min and max of the pulse
        signal_data = self.data_dict[ECHO_KEY][-window:]
        self.signal_range = np.max(signal_data) * 0.733
        self.normal_range = np.median(signal_data)

        print("Signal range: %s, normal_range: %s" % (self.signal_range, self.normal_range))
        # TODO: Check the min and max of the echo

        echo_data = self.data_dict[SIGNAL_KEY][-window:]
        self.echo_range = np.percentile(echo_data, 96) # np.max(echo_data) * 0.5
        self.echo_normal_range = np.median(echo_data)

        print("Echo range: %s, normal_range: %s" % (self.signal_range, self.normal_range))

        # Tell the controller to stop auto calibration pulses
        # self.control_queue.put('disable_pulses')
        self.control_queue.put('slow_pulses')

        print("Calibration complete")

        # self.multi_pulse_thread = MultiPulseThread(self.liveplot, self.control_queue, 3, 0.5)

        pass

    def estimate_distances(self, pulses_filtered, echoes_filtered, mean_dt, scanline):
        # TODO: Find the first pulse, for which there's at least one echo withing 500us range
        scan_range_us = normalize_max_distance_time_us
        scan_range_mm = scan_range_us / (ten_cm_time_us) * 10
        scan_range_samples = scan_range_us // mean_dt
        # print(
        #     "Will use a sample range of %d samples to find the first pulse \nwith an echo within %dus \nor ~%.2fmm" % (
        #     scan_range_samples, scan_range_us, scan_range_mm))

        pulse_index = 0
        valid_echo_found = False
        valid_echoes = None  # placeholder for valid echoes

        signal_echoes = []

        if len(pulses_filtered) > 0:
            while pulse_index < len(pulses_filtered):
                pt = pulses_filtered[pulse_index]
                if pulse_index + 1 < len(pulses_filtered) - 1:
                    ve = echoes_filtered[(echoes_filtered > pt) & (echoes_filtered < pulses_filtered[pulse_index + 1])]
                else:
                    ve = echoes_filtered[(echoes_filtered > pt)]

                # ve = echoes_filtered[(echoes_filtered > pt) & (echoes_filtered < pt + scan_range_samples)]
                #ve = echoes_filtered[(echoes_filtered > pt)]

                if len(ve) > 0:
                    # print(len(ve))
                    valid_echoes = ve
                    signal_echoes.append((pt, ve))

                pulse_index += 1

        if len(signal_echoes) > 0:
            for pair in signal_echoes:
                # print("Found healthy pulse at %d us with echoes: " % (pair[0]))
                list(map(lambda i: self._process_echo(i, pair[0], scanline), pair[1]))
            pass
        else:
            pass
            # print("> None found :(")

        # TODO

        # self.data_dict['interp1'].append(dist1)
        # self.data_dict['interp2'].append(amp1)

        # return [[dist1, amp1]]

    def _process_echo(self, e, pulse_time, scanline):
        dist = (e - pulse_time) * speed_of_sound_mm_per_usec  # time * speed, half, because had to go there and back
        maxlen = len(scanline) - 1

        r = dist / normalize_max_distance_mm

        if r > 1.0:
            # Out of range
            pass
            # r = 1.0
        else:
            pos = int(r * maxlen)

            # Adjust scanline at position
            scanline[pos] = scanline[pos] + 1
            # if scanline[pos] > 20:
            #     scanline[pos] = 20

    def update_text(self):
        text = ""
        run_time = time.time() - self.start_time

        if len(self.data_dict['timestamps']) > 100:
            data_time = self.data_dict['timestamps'][-1] / 10 ** 6
            text += "Run time: %.2f\n" % (run_time)
            text += "Data time: %.2f\n" % (data_time)
            text += "Time lag: %.2f\n" % (run_time - data_time)
            text += "Datapoints: %d\n" % (len(self.data_dict[SIGNAL_KEY]) - self.initial_padding_size)
            # text += "S/Queue size: %d\n" % (signal_queue.qsize()) # not supported on macs
            text += "Mean DT micros: %.4f\n" % (np.mean(np.diff(self.data_dict['timestamps'][-10000:])))
            text += "Zoom: %d\n" % (self.liveplot.signal_zoom)

        # if len(self.liveplot.lines_per_read) > 1:
        #     mean_lines_per_read = np.mean(self.liveplot.lines_per_read)
        #     text += "Mean lines/read: %.1f\n" % (mean_lines_per_read)

        #
        # mean_time_per_read = np.mean(liveplot.time_per_read) / 10**6
        # text += "Mean time/read: %.6f s\n" % (mean_time_per_read)

        # if mean_time_per_read > 0:
        #     text += "Sampling hz: %.4f\n" % (mean_lines_per_read / mean_time_per_read)

        if len(self.data_dict['timestamps']) > 5100:
            self.liveplot.sampling_rate = (5000 / (self.data_dict['timestamps'][-1] - self.data_dict['timestamps'][-5000])) * 10**6

        text += "Sampling: %.1f Khz\n" % (self.liveplot.sampling_rate / 1000)

        # FPS counting and update
        now = time.time()
        dt = (now-self.lastupdate)
        if dt <= 0:
            dt = 0.000000000001
        fps2 = 1.0 / dt
        self.lastupdate = now
        self.fps = self.fps * 0.9 + fps2 * 0.1

        text += 'Mean Frame Rate:  {fps:.3f} FPS\n'.format(fps=self.fps)

        # if self.liveplot.strong_freqs is not None and len(self.liveplot.strong_freqs) > 0:
        #     text += "============\n"
        #
        #     strong_freqs = np.array(self.liveplot.strong_freqs)
        #
        #     # strong_freqs shape: ([i, self.liveplot.current_freqs_range[i], freq_amp])
        #     i = 0
        #     # print("==========")
        #     for freq in reversed(strong_freqs[strong_freqs[:, 2].argsort()][0:25]):
        #         # print(freq)
        #         text += "F%d) %s: %dHz\n" % (i, freq[2], freq[1])
        #         i += 1

        self.label.setText(text)

    def closeEvent(self, event):
        print("Exiting gracefully...")
        self.update_thread.terminate()
        event.accept()

    def update(self):
        try:
            self.update_text()

            if self.liveplot.paused is False and self.metrics_established:
                self.update_charts()

            if self.metrics_established is not True and len(self.data_dict[SIGNAL_KEY]) > 16000:
                self.establish_metrics()
                pass

            # self.xdata = np.linspace(self.counter, self.counter + 100, 6000)
            # self.ydata = np.sin(self.xdata)

            # self.img.setImage(self.data)
            # self.signal_plot.setData(self.xdata, self.ydata)
            # self.fft_plot.setData(self.xdata, self.ydata)


            # print("_update")

            QtCore.QTimer.singleShot(1, self.update)

            self.counter += 1
        except KeyboardInterrupt:
            print("Exiting gracefully...")
            self.decode_thread.join()
            self.filter_thread.join()
            self.update_thread.terminate()
            self.multi_pulse_thread.terminate()
        except BaseException as e:
            print("update thread: %s" % str(e))
            QtCore.QTimer.singleShot(1, self.update)

    def update_slower(self):
        try:
            if self.metrics_established is True:
                timestamp_window, signal_window, echo_window, filtered_echo_window = self.update_image()
                self.update_charts_after_image_update(timestamp_window)
        except BaseException as e:
            print(e)
        finally:
            QtCore.QTimer.singleShot(33, self.update_slower)

    def update_image(self):
        # print(self.img_scanlines.getImageItem().image)

        if self.current_scanline > self.max_scanline - 1:
            self.current_scanline = 0

        scanline = None

        window = 30000

        if self.liveplot.paused is False and self.metrics_established:
            timestamp_window = self.data_dict['timestamps'][-window:]
            signal_window = self.data_dict[SIGNAL_KEY][-window:] # TX
            echo_window = self.data_dict[ECHO_KEY][-window:] # RX

            self.liveplot.mean_echo = np.mean(echo_window)
            self.liveplot.min_echo = np.min(echo_window)
            self.liveplot.max_echo = np.max(echo_window)

            filtered_echo_window = self.filter_echoes(timestamp_window, echo_window)

            mean_dt = np.mean(np.diff(timestamp_window))

            # if np.max(signal_window) > self.signal_range * 0.8: # simple check for if there are any pulses at all
            if np.max(signal_window) > 0:  # simple check for if there are any pulses at all
                img = self.img_scanlines.getImageItem()
                pulses = self.collect_pulses(timestamp_window, signal_window)
                # pulses = self.collect_pulses(timestamp_window, echo_window)
                self.collected_echoes = echoes = self.collect_echoes(timestamp_window, filtered_echo_window)
                # self.collected_echoes = echoes = self.collect_echoes(timestamp_window, echo_window)

                if len(pulses >= 1) and len(echoes > 1):
                    scanline = img.image[self.current_scanline]
                    self.estimate_distances(pulses, echoes, mean_dt, scanline)

                    d = 0.25

                    if self.current_scanline > 0:
                        # Decay
                        # prev_scanline = img.image[self.current_scanline - 1]
                        # prev_scanline[prev_scanline > 0] -= d # decay
                        img.image[:][:] -= d
                        # prev_scanline1 = img.image[self.current_scanline - 2]
                        # prev_scanline1[prev_scanline1 > 0] -= d  # decay
                        # prev_scanline2 = img.image[self.current_scanline - 3]
                        # prev_scanline2[prev_scanline2 > 0] -= d  # decay
                        # prev_scanline3 = img.image[self.current_scanline - 4]
                        # prev_scanline3[prev_scanline3 > 0] -= d  # decay
                        # prev_scanline4 = img.image[self.current_scanline - 5]
                        # prev_scanline4[prev_scanline4 > 0] -= d  # decay

                        # scanline += prev_scanline

                        # img.image[self.current_scanline + 1] = scanline.copy()
                        # img.image[self.current_scanline + 2] = scanline.copy()
                        # img.image[self.current_scanline + 3] = scanline.copy()
                        # img.image[self.current_scanline + 4] = scanline.copy()

                    self.img_scanlines.setImage(img.image)
                    # self.img_scanlines.autoRange()
                    # img.updateImage()

                    # self.current_scanline += 5
                    self.current_scanline += 1
                    if self.current_scanline >= self.image_width:
                        # Clear image
                        img.image[:][:] = 0
                        self.current_scanline = 0

                # print(scanline)

            # if scanline is not None:
            # len(img.image[0][self.current_scanline])
            # img.image[self.current_scanline] = np.full([1], 0.0) # reset scanline

            # for point in scanline:
            #     img.image[self.current_scanline, max([self.image_height - 1, int(point[0])])] = point * 255 # brightness voltage / 255
            return (timestamp_window, signal_window, echo_window, filtered_echo_window)

        return (None, None, None)

    def collect_pulses(self, timestamp_window, pulses_window):
        # print("Pulse detection range: %f - %f" % (timestamp_window[0], timestamp_window[-1]))
        peaks = signal.find_peaks(np.diff(pulses_window), height=1, distance=10)
        # peaks = signal.find_peaks(pulses_window, threshold=10, height=150, distance=350)
        print("%d pulses" % len(peaks[0]))
        # print("RAW PEAKS (%d): %s" % (len(peaks[0]), peaks[0]))
        pulses = peaks[0]

        return np.array(pulses)

    def collect_echoes(self, timestamp_window, filtered_echo_window):
        self.echo_peak_amp = peak_amp = np.percentile(filtered_echo_window, 99.5)
        # self.echo_peak_amp = peak_amp = self.echo_range
        #     display(snapshot_echoes_filtered, peak_amp)
        peaks = signal.find_peaks(filtered_echo_window, height=peak_amp, distance=10)
        # print("RAW ECHO PEAKS: %d at amp %f" % (len(peaks[0]), peak_amp))
        # base_voltage = 625 # it's not in volts but in sampling resolution (12-bit currently, i.e. total range would be -2048..2048)
        # WARNING: Default voltage bias might differ in different connection scenarios. Ideally it must be sampled and calculated dynamically

        return np.array(peaks[0])

    def filter_echoes(self, timestamp_window, echo_window):
        if not hasattr(self, "echo_filter_butter1"):
            decay_steps = 1
            filter_freq = 0.9995 # 0.97
            print("Setting up filter with decay: %d, freqpoint (normalized): %f" % (decay_steps, filter_freq))
            self.echo_filter_butter1 = signal.butter(decay_steps, filter_freq, 'low', analog=False)

        if not hasattr(self, "echo_filter_ellip1"):
            filt2_N, filt2_Wn = signal.ellipord(0.1, 0.15, 3, 90, False)
            self.echo_filter_ellip1 = signal.ellip(filt2_N, 3, 90, filt2_Wn, 'low', False)

        # filtered_signal = signal.filtfilt(self.echo_filter_butter1[0], self.echo_filter_butter1[1], echo_window)
        # # filtered_signal = signal.filtfilt(self.echo_filter_ellip1[0], self.echo_filter_ellip1[1], echo_window)
        # denorm_ratio = np.max(echo_window) / np.max(filtered_signal)
        #
        # # filtered_norm_abs_echoes = np.abs((filtered_signal - np.median(filtered_signal)) * denorm_ratio)
        #
        # filtered_norm_abs_echoes = (filtered_signal - np.median(filtered_signal)) * denorm_ratio
        # filtered_norm_abs_echoes = np.abs(signal.hilbert(filtered_norm_abs_echoes))
        #
        # # return np.array(filtered_signal)
        return np.abs(echo_window)
        # return filtered_norm_abs_echoes

class LivePlotter():
    def __init__(self, data_dict, signal_queue, control_queue, info_queue, qtwindow):
        self.data_dict = data_dict

        self.paused = False
        self.auto_y = True

        self.pulsed_paused = False

        self.infotext = None # TextBox for data display

        self.current_ftt_abs = None
        self.current_freqs_range = None
        self.strong_freqs = None

        self.time_per_read = []
        self.lines_per_read = []

        self.mean_dt = 4
        self.sampling_rate = 0
        self.mean_echo = 0
        self.min_echo = 0
        self.max_echo = 1

        self.first_timestamp = 0 # first recorded serial data timestamp

        self.signal_zoom = 30000
        self.initial_padding = 6000

        self.qtwindow = qtwindow

        self.serial_proc = None

        self.signal_queue = signal_queue
        self.control_queue = control_queue
        self.info_queue = info_queue

    def on_pause_click(self, event):
        self.paused = not self.paused
        self.control_queue.put("toggle_pause")

    def on_toggle_pulse_click(self, event):
        self.control_queue.put("toggle_slow_pulse")

    def on_one_pulse_click(self, event):
        self.control_queue.put("one_pulse")

    def on_func1_click(self, event):
        self.auto_y = not self.auto_y

    def on_func2_click(self, event):
        pass

    def on_func3_click(self):
        if self.signal_zoom > 30000:
            self.signal_zoom = 30000
        elif self.signal_zoom == 30000:
            self.signal_zoom = 12000
        elif self.signal_zoom == 12000:
            self.signal_zoom = 6000
        elif self.signal_zoom == 6000:
            self.signal_zoom = 3000
        elif self.signal_zoom == 3000:
            self.signal_zoom = 1500
        elif self.signal_zoom == 1500:
            self.signal_zoom = 750
        elif self.signal_zoom == 750:
            self.signal_zoom = 325
        elif self.signal_zoom == 325:
            self.signal_zoom = 30000


        self.qtwindow.update_charts()
        self.qtwindow.update_text()

    def on_plot_all(self):
        self.signal_zoom = len(self.data_dict['timestamps'])
        self.qtwindow.update_charts()
        self.qtwindow.update_text()

    def on_ltrim_click(self, event):
        self.data_dict['timestamps'] = self.data_dict['timestamps'][-(self.signal_zoom):]
        self.data_dict[SIGNAL_KEY] = self.data_dict[SIGNAL_KEY][-(self.signal_zoom):]
        self.data_dict[ECHO_KEY] = self.data_dict[ECHO_KEY][-(self.signal_zoom):]

    def on_save_click(self):
        r = self.initial_padding - 1
        rows = []
        for timestamp in self.data_dict['timestamps'][r:]:
            rows.append(",".join([str(int(self.data_dict['timestamps'][r])), str(self.data_dict[SIGNAL_KEY][r]), str(self.data_dict[ECHO_KEY][r])]))
            r += 1

        f = open("data/liverec.csv", "w")
        f.write("\n".join(rows))
        f.close()

    def on_signal_checkbox_click(self, label):
        pass

    def read_info_queue(self):
        try:
            data = self.info_queue.get(False)
            if data[0:2] == 'RL':
                self.lines_per_read.append(int(data[2:]))
                self.lines_per_read = self.lines_per_read[-100:]
            elif data[0:2] == 'FT':
                self.first_timestamp = int(data[2:])
        except queue.Empty as error:
            pass


class FftThread(threading.Thread):
    def __init__(self, plot, series, data_dict, state_dict):
        threading.Thread.__init__(self)
        self.deamon = True
        self.plot = plot
        self.series = series
        self.data_dict = data_dict
        self.start()

    def run(self):
        while 1 and main_thread.isAlive():
            self.data_dict['ftt'] = scipy.fftpack.fft(self.data_dict[SIGNAL_KEY][-1500:])
            time.sleep(0.01)

class LivePlotDecodeThread(threading.Thread):
    def __init__(self, series, data_dict, signal_queue, info_queue, signal_pipe_recv, liveplot):
        threading.Thread.__init__(self)
        self.deamon = True
        self.series = series
        self.data_dict = data_dict
        self.liveplot = liveplot
        self.signal_queue = signal_queue
        self.signal_pipe_recv = signal_pipe_recv
        self.info_queue = info_queue
        self.byte_row_size = 10
        self.rows_per_read = 100000

        self.insert_locked = False # IMPORTANT: Thread-lock for updating the data_dict

        self.start()

    def run(self):
        skipped_first = False

        first_timestamp = 0
        last_timestamp = 0

        times = 0

        struct_format_string = "cLhhc"

        while 1 and main_thread.isAlive():
            try:
                # try:
                #     self.liveplot.read_info_queue()
                # except error:
                #     print(error)

                # try:
                #     str_line = self.signal_queue.get(False)
                # except BaseException as error:
                #     # print(error)
                #     time.sleep(0.0001)
                #     continue
                try:
                    # row_buffer = self.signal_pipe_recv.recv()
                    # row_buffer = self.signal_pipe_recv.recv_bytes(self.rows_per_read * self.byte_row_size * 1000)
                    # row_buffer = array.array('i', [0]*500000)
                    # print("Received")

                    times += 1

                    rows_in_buffer = 0

                    if self.signal_pipe_recv.poll(0.1):
                        # print("Receiving...")
                        row_buffer = self.signal_pipe_recv.recv_bytes()
                        buffer_length = len(row_buffer)
                        # buffer_length = self.signal_pipe_recv.recv_bytes_into(row_buffer)
                        rows_in_buffer = int(buffer_length / self.byte_row_size)

                        # print("Got %d bytes buffer, %d rows" % (buffer_length, rows_in_buffer))
                    else:
                        # print("Poll: no data")
                        rows_in_buffer = 0

                    if rows_in_buffer >= 1:
                        # pass
                        flattened_rows = struct.unpack_from("=" + (struct_format_string * rows_in_buffer), row_buffer)
                        # print("Unpacked %d cells, %d rows" % (len(flattened_rows), len(flattened_rows) / self.byte_row_size))
                        rows_in_buffer -= 1  ### padding!!!!

                        new_timestamps = [None] * rows_in_buffer
                        new_vals1 = np.array([None] * rows_in_buffer)
                        new_vals2 = np.array([None] * rows_in_buffer)
                        # new_vals2 = [None] * rows_in_buffer
                    else:
                        # time.sleep(0.01)
                        continue

                    # Each row is 10 bytes, starting with 84 ('T')
                    # First four bytes is an unsigned long timestamp
                    # print(d)
                    # row = struct.unpack_from("=cLhhc", data)

                    # line_buffer = line_buffer.split("\n")
                    # print("Reading %d lines" % buffer_length)
                    # line_buffer = []
                    # print(line_buffer)
                except EOFError as error:
                    print(error)
                    continue
                    # raise error
                except UnicodeDecodeError as error:
                    print(error)
                    continue
                except BaseException as error:
                    print("BaseException: " + str(error))
                    # QtCore.pyqtRemoveInputHook()
                    # pdb.set_trace()
                    time.sleep(0.25)
                    continue

                r = 0
                i = 0
                errors = 0

                # Each row has 5 values: ['T', timestamp, signal1, signal2, 255]
                while r < rows_in_buffer:
                    if (r > 1 and flattened_rows[i + 1] < new_timestamps[r-1]) or (r == 0 and len(self.data_dict['timestamps']) > 0 and flattened_rows[i + 1] < self.data_dict['timestamps'][-1]):
                        print("INVALID TIME: now: %d, was %d, before: %d" % (new_timestamps[r] + first_timestamp, new_timestamps[r-1] + first_timestamp, new_timestamps[r-2] + first_timestamp))
                        errors += 1
                        r += 1
                        continue
                        # self.liveplot.paused = True
                        # self.liveplot.serial_proc.terminate()
                        # QtCore.pyqtRemoveInputHook()
                        # pdb.set_trace()
                        # _thread.interrupt_main()
                        # raise "Invalid time"

                    new_timestamps[r] = flattened_rows[i + 1]

                    if first_timestamp == 0:
                        first_timestamp = self.liveplot.first_timestamp = new_timestamps[r]

                    new_timestamps[r] = new_timestamps[r] - first_timestamp


                    new_vals1[r] = flattened_rows[i + 2]
                    new_vals2[r] = flattened_rows[i + 3]
                    # print("r: %d, i: %i, %d, %d, %d" % (r, i, new_timestamps[r], new_vals1[r], new_vals2[r]))
                    i += 5
                    r += 1

                if rows_in_buffer - errors >= 1:
                    while self.insert_locked is True:
                        pass

                    self.insert_locked = True # set thread-lock
                    try:
                        self.data_dict[SIGNAL_KEY] = np.append(self.data_dict[SIGNAL_KEY], new_vals1)
                        # Replace zeroes with mean signal in the echo during pulse times
                        new_vals2[np.where(new_vals2 == 0)] = self.liveplot.mean_echo
                        self.data_dict[ECHO_KEY] = np.append(self.data_dict[ECHO_KEY], new_vals2)
                        self.data_dict['timestamps'] = np.append(self.data_dict['timestamps'], new_timestamps)
                    except BaseException as e:
                        print("DecodeError: %s" % str(e))
                    finally:
                        self.insert_locked = False # release lock
            except BaseException as e:
                print("Generic decode error: " % str(e))

class LivePlotProc(mp.Process):
    def __init__(self, liveplot, signal_queue, control_queue, info_queue, signal_pipe_snd):
        mp.Process.__init__(self)
        self.deamon = True
        self.signal_queue = signal_queue
        self.control_queue = control_queue
        self.info_queue = info_queue
        self.signal_pipe_snd = signal_pipe_snd
        self.liveplot = liveplot
        self.paused = False
        self.sending_buffer = False
        self.send_counter = 0

        time.sleep(1.0)
        # self.serial.reset_output_buffer()
        # self.serial.reset_input_buffer()

        try:
            self.pytty = None
            self.serial = serial.Serial(default_usb_port, baudrate=baudrate)
            self.pytty_desc = pty.slave_open(default_usb_port)
            self.pytty = os.fdopen(self.pytty_desc)

            time.sleep(2.0)

            self.serial.reset_output_buffer()
            self.process_command("enable_pulses") # force enable constant pulsing for calibration

            time.sleep(1.0)

            # Truncate garbage buffer that was collected without enabled pulses
        except BaseException as error:
            print(error)
            pass

        self.start()

    def run(self):
        skipped_first = False
        bytes_per_row = 10
        rows_per_read = 20000
        # os.nice(-19)

        # buffer = [None] * buffer_size

        total_lines_read = 0
        total_lines_processed = 0

        send_buffer = []

        self.command_read_cycle = 0

        self.serial.reset_input_buffer()

        while 1 and main_thread.is_alive():
            if self.paused is True or self.pytty is None: # do pause
                time.sleep(0.05)
                self.read_control_queue()
                continue

            # if not self.serial.is_open:
            #     # wait for serial
            #     time.sleep(0.01)
            #     pass

            # print("Reading...")
            # last_lines = self.serial.readlines(max_bytes_per_read)
            try:
                # new_bytes = self.pytty.read(max_bytes_per_read)

                data = self.serial.read(rows_per_read * bytes_per_row)

                # Start byte must be 84 ('T'), if it's not, we need to pad left
                # until the first start byte is reached.
                start_pos = 0
                end_pos = len(data) - 2 # one last row will be padded off

                while data[start_pos] != 84:
                    start_pos += 1

                # Pad one row off the very end, for safety
                while data[end_pos] != 255 and data[end_pos + 1] != 84 and end_pos > 9:
                    end_pos -= 1

                if end_pos - start_pos > 9:
                    buffer = data[start_pos:(end_pos + 1)] # after fixing and cleaning the data
                    # print("Piping %d bytes" % len(buffer))
                    self.send_counter += 1

                    # Sleep a bit every N cycles to let the
                    # receiving pipe unclog the buffer
                    # if self.send_counter % 20 == 0:
                    #     time.sleep(0.1)

                    send_buffer.extend(buffer)
                    send_buffer_length = len(send_buffer)

                    if send_buffer_length > 500 and self.sending_buffer is False:
                        tmp_buffer = bytes(send_buffer.copy())
                        t = threading.Thread(target=self.send_signal_buffer, args=(tmp_buffer, ))
                        t.start()
                        send_buffer.clear()
                    # t = threading.Thread(target=self.send_signal_buffer, args=(buffer,))
                    # t.start()
                    # time.sleep(0.01)

                # last_lines = self.pytty.readlines(max_bytes_per_read)
            except BaseException as error:
                print("Serial IO error: %s; waiting 3 sec" % error)

                time.sleep(3)

                try:
                    print("Closing old ports...")
                    self.serial.close()
                    self.pytty.close()
                    time.sleep(1.5)
                    print("Reopening")
                    self.serial = serial.Serial(default_usb_port, baudrate=baudrate)
                    self.serial.reset_output_buffer()
                    self.serial.reset_input_buffer()
                    self.pytty_desc = pty.slave_open(default_usb_port)
                    self.pytty = os.fdopen(self.pytty_desc)
                    print("Device reconnected")
                except BaseException as error:
                    print("Serial reopen error: %s; waiting +1 sec more" % error)
                    time.sleep(1)

                continue

            # Read control queue
            self.read_control_queue()

    def send_signal_buffer(self, buffer):
        self.sending_buffer = True
        try:
            send_length = len(buffer)
            self.signal_pipe_snd.send_bytes(buffer, 0, send_length)
            # print("Sent %d b" % (send_length))
        except BaseException as e:
            print(e)
        finally:
            self.sending_buffer = False

    def read_control_queue(self):
        # Read control queue
        self.command_read_cycle += 1

        if self.command_read_cycle > 20:
            self.command_read_cycle = 0
            try:
                command = self.control_queue.get(False)
                self.process_command(command)
            except BaseException as error:
                pass


    def process_command(self, command):
        if command == 'toggle_pause':
            self.paused = not self.paused
        elif command == 'toggle_pulse':
            print("toggle_pulse")
            try:
                self.serial.write(b't')
            except BaseException as e:
                print(e)
        elif command == 'one_pulse':
            print("one_pulse")
            try:
                self.serial.write(b'o')
            except BaseException as e:
                print(e)
        elif command == 'enable_pulses':
            print("enable_pulses")
            try:
                self.serial.write(b'e')
            except BaseException as e:
                print(e)
        elif command == 'disable_pulses':
            print("disable_pulses")
            try:
                self.serial.write(b'd')
            except BaseException as e:
                print(e)
        elif command == 'slow_pulses':
            print("slow_pulses")
            try:
                self.serial.write(b'p')
            except BaseException as e:
                print(e)
        elif command == 'toggle_slow_pulse':
            print("toggle_slow_pulse")
            try:
                self.serial.write(b'b')
            except BaseException as e:
                print(e)


    def send_command(self, command):
        self.control_queue.put(command)

class SimFeedThread(threading.Thread):
    def __init__(self, data_dict, filename, batchsize = 1000, delay = 0.005, loop = True):
        threading.Thread.__init__(self)
        self.deamon = True
        self.filename = filename
        self.paused = False
        self.batchsize = batchsize
        self.data_dict = data_dict
        self.delay = delay
        self.loop = loop

        time.sleep(2.0)

        self.load_data()

        self.start()

    def load_data(self):
        csv = pd.read_csv("data/liverec14_gel.csv")

        self.timings = np.array(csv.iloc[:, 0:1]).flatten()
        self.echoes = np.array(csv.iloc[:, 1:2]).flatten()
        self.signal = np.array(csv.iloc[:, 2:3]).flatten()

    def run(self):
        self.pos = 0
        pos = 0
        batchsize = self.batchsize
        maxlen = len(self.timings)
        finished = False

        while 1 and finished is not True and main_thread.is_alive():
            if self.paused is True: # do pause
                time.sleep(0.05)

            self.data_dict['timestamps'] = np.concatenate((self.data_dict['timestamps'], self.timings[pos:(pos+batchsize)]))
            self.data_dict[SIGNAL_KEY] = np.concatenate((self.data_dict[SIGNAL_KEY], self.echoes[pos:(pos+batchsize)]))
            self.data_dict[ECHO_KEY] = np.concatenate((self.data_dict[ECHO_KEY], self.signal[pos:(pos+batchsize)]))

            pos += batchsize

            if pos > maxlen:
                if self.loop is True:
                    pos = 0
                else:
                    finished = True

            self.pos = pos

            time.sleep(self.delay)



class OnePulseThread(threading.Thread):
    def __init__(self, control_queue, delay):
        threading.Thread.__init__(self)
        self.control_queue = control_queue
        self.delay = delay
        self.start()

    def run(self):
        time.sleep(self.delay)
        self.control_queue.put("one_pulse")

class MultiPulseThread(threading.Thread):
    def __init__(self, liveplot, control_queue, delay, cycle):
        threading.Thread.__init__(self)
        self.control_queue = control_queue
        self.delay = delay
        self.cycle = cycle
        self.liveplot = liveplot
        self.start()

    def run(self):
        time.sleep(self.delay)
        while 1:
            if self.liveplot.pulsed_paused:
                time.sleep(0.1)
                continue

            try:
                self.control_queue.put("one_pulse")
            except BaseException as e:
                print(e)

            print("Pulsed")
            time.sleep(self.cycle)

class FilterThread(threading.Thread):
    def __init__(self, data_dict, liveplot):
        threading.Thread.__init__(self)
        self.deamon = True
        self.data_dict = data_dict
        self.liveplot = liveplot
        self.iw = None
        self.iw_ftt = None
        self.initial_window_prepared = False
        self.start()

    def prepare_initial_window(self):
        self.iw = self.data_dict[ECHO_KEY]
        self.iw_ftt = scipy.fftpack.fft(self.iw)
        sampling_rate = np.mean(np.diff(self.data_dict['timestamps'])) / 1000000

        raw_freqs = scipy.fftpack.fftfreq(len(self.iw_ftt), sampling_rate)
        i = round(len(raw_freqs) / 2) + 1
        self.iw_freqs = raw_freqs[0:(i - 1)]
        self.iw_ftt_abs = np.abs(self.iw_ftt[(i - 1):])

        # TODO: Detect initial frequencies

        self.initial_window = True

    def run(self):
        if self.initial_window_prepared is False and len(self.data_dict['timestamps']) > 2000:
            self.prepare_initial_window()

        while 1 and main_thread.isAlive():
            if len(self.data_dict['timestamps']) < 2000:
                time.sleep(0.1)
                continue

            # ftt_data = self.data_dict['ftt'] = scipy.fftpack.fft(self.data_dict[SIGNAL_KEY][-2000:])

            self.liveplot.mean_dt = round(np.mean(np.diff(self.data_dict['timestamps'][-2000:])), 1)
            sampling_rate = self.liveplot.sampling_rate = self.liveplot.mean_dt / 1000000

            # raw_freqs = scipy.fftpack.fftfreq(len(ftt_data), sampling_rate)
            # iis = round(len(raw_freqs) / 2) + 1
            # self.liveplot.current_freqs_range = raw_freqs[0:(iis - 1)]
            # ftt_abs = self.liveplot.current_ftt_abs = np.abs(ftt_data[(iis - 1):])

            # max_amp = np.max(ftt_abs)
            # median_amp = np.median(ftt_abs)
            # mean_amp = np.mean(ftt_abs)
            # stdev_amp = np.std(ftt_abs)
            # min_amp = np.min(ftt_abs)

            # self.liveplot.strong_freqs = []
            # i = 0
            #
            # for freq_amp in ftt_abs:
            #     # Highpass
            #     if freq_amp > mean_amp + stdev_amp * 4:
            #         self.liveplot.strong_freqs.append([i, self.liveplot.current_freqs_range[i], int(freq_amp)])
            #         # continue
            #     # elif freq > min_amp * 1.1:
            #     #     selected_freqs.append([i, freq])
            #
            #     i = i + 1
            #
            # do_filter = True

            time.sleep(1)

class Sonic(object):
    def __init__(self):
        pass

    def distance(self, t1, t2, spd = 0.00035):
        # time is in microseconds,
        # speed must be in microseconds
        return (t2 - t1) * spd

main_thread = threading.currentThread()

if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    thisapp = App()
    thisapp.show()
    sys.exit(app.exec_())

# TODO: Properly exit subprocs