import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.fftpack
from scipy import signal
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Button
from matplotlib.widgets import TextBox
from matplotlib.widgets import CheckButtons
# import seaborn as sns
import pdb
import serial
# from tkinter import *
import math, random, threading, time, os
import multiprocessing as mp
import queue

# suppresses the use of scientific notation for small numbers when printing np.array
np.set_printoptions(suppress=True)

# default_usb_port = "/dev/cu.usbmodem144201"
default_usb_port = "/dev/cu.usbmodem53923101"

def lineplot(data):
    plot = sns.lineplot(data=np.array(data))
    plot.get_figure().show()
    return plot

def record_serial(path=None, csv=None):
    if path is None:
        path = default_usb_port

    print("Recording %s" % (path))

    data = [] # array of csv data lines
    ser = serial.Serial(path)

    while 1:
        try:
            data.extend(ser.readlines(10000))
        except KeyboardInterrupt:
            print("Interrupted by user!")
            ser.close()
            break

    if ser.is_open:
        ser.close()

    data = [i.decode().rstrip() for i in data] # strip lines

    if csv is not None:
        f = open(csv, "w")
        f.write("\n".join(data))
        f.close()
        pass

    return data

def plot_data(path=None):
    if path is None:
        path = "data/rec1.csv"

    df = pd.read_csv(path)

    plot_count = 3

    plt.ion()
    fig = plt.figure(figsize=(13, 9))
    fig.tight_layout()
    signal_plot = fig.add_subplot(plot_count, 1, 1)
    fft_plot = fig.add_subplot(plot_count, 1, 2)
    # freqs_plot = fig.add_subplot(plot_count, 1, 3)
    # freqdom_plot = fig.add_subplot(plot_count, 1, 3)
    filtered_plot = fig.add_subplot(plot_count, 1, 3)

    timings = df[df.columns[0]]
    raw_signal = df[df.columns[1]]

    timings_win = timings[0:400]
    raw_signal_win = raw_signal[0:400]

    sin_a = np.sin(np.linspace(np.pi * (3/4), 2200, len(raw_signal))) * 20 + 300

    # line1, = signal_plot.plot(timings_win, raw_signal_win, linewidth=0.5)
    line1, = signal_plot.plot(timings, raw_signal, linewidth=0.5)
    # line_sina, = signal_plot.plot(timings, sin_a, linewidth=0.5)
    # signal_plot.set_ylim([0, 1024])

    delta_time = dt_micros = np.mean(np.diff(timings)) # micros of delta time between recorded cycles
    dt_millis = dt_micros / 1000
    dt = dt_micros / 1000
    sampling_rate = dt_micros / 1000000

    # line_ftt, = fft_plot.plot(raw_freqs, np.abs(ftt_data), linewidth=0.5)  # FTT plot

    # fft_plot.set_xlim([(len(ftt_data) / -2) - 10, len(ftt_data) / 2 + 10])

    do_filter = False

    if do_filter is True:
        ftt_data = scipy.fftpack.fft(raw_signal)

        raw_freqs = scipy.fftpack.fftfreq(len(ftt_data), sampling_rate)
        iis = round(len(raw_freqs) / 2) + 1
        freqs = raw_freqs[0:(iis - 1)]
        ftt_abs = np.abs(ftt_data[(iis - 1):])
        print("Freqs min/max: %s / %s" % (np.min(freqs), np.max(freqs)))

        line_ftt, = fft_plot.plot(np.abs(raw_freqs), np.abs(ftt_data), linewidth=0.5)  # FTT plot

        fft_plot.set_ylim([np.min(ftt_data), np.max(ftt_data)])

        # freqs_plot, = freqs_plot.plot(np.arange(0, len(freqs), 1), freqs, linewidth=0.5) # Freqs plot
        # freqdom_plot.loglog(freqs, ftt_abs, linewidth=0.5) # Freqs plot
        # pdb.set_trace()
        # freqdom_plot.plot(raw_freqs, np.abs(ftt_data), linewidth=0.5)  # Freqs plot
        # freqdom_plot.set_title('Frequency domain')

        # Filtering:

        off = 200

        # Find a non-core high frequency, with a very small amplitude relative
        max_amp = np.max(ftt_abs)
        median_amp = np.median(ftt_abs)
        mean_amp = np.mean(ftt_abs)
        stdev_amp = np.std(ftt_abs)
        min_amp = np.min(ftt_abs)

        print("Windowed signal FTT: max: %s, min: %s, median: %s, mean: %s, std: %s" % (
        max_amp, min_amp, median_amp, mean_amp, stdev_amp))

        i = 0
        selected_freqs = []

        for freq_amp in ftt_abs:
            # Highpass
            if freq_amp < max_amp * 0.5 and freq_amp > median_amp:
                selected_freqs.append([i, freq_amp])
                # continue
            # elif freq > min_amp * 1.1:
            #     selected_freqs.append([i, freq])

            i = i + 1

        filtered_signal = np.array(raw_signal.copy())

        for freq in selected_freqs[0:20]:
            i = freq[0]
            f = abs(freqs[i])
            # wc = abs(freqs[np.argmax(ftt_abs)]) / (0.5 / dt)
            wc = f / (0.5 / sampling_rate)

            print("Filtering for %shz (norm: %s)" % (f, wc))

            wp = [wc * 0.9, min(0.999999, wc / 0.9)]
            ws = [wc * 0.95, min(0.999999, wc / 0.95)]
            # pdb.set_trace()
            try:
                # att = 10**6 / dt_micros / f
                # print("%s %s" % (f, att))
                b, a = signal.iirdesign(wp, ws, 1, 33)
                filtered_signal = signal.filtfilt(b, a, filtered_signal)
            except BaseException as error:
                print(error)
                # pdb.set_trace()

        raw_line_filtered, = filtered_plot.plot(timings, raw_signal, 'r', linewidth=0.5)
        line_filtered, = filtered_plot.plot(timings, filtered_signal, 'g', linewidth=0.5)
        filtered_plot.set_ylim([0, 1024])

    plt.show()

    pdb.set_trace()

def liveplot_serial(path=None, filter=None):
    if path is None:
        path = default_usb_port

    initial_padding_size = 6000
    # plt.style.use('ggplot')
    data = np.array([512] * initial_padding_size)
    data2 = np.array([512] * initial_padding_size)
    # timestamps = np.linspace(0, 5999 * 636, 6000)
    timestamps = np.linspace(-6000 * 40, 0, initial_padding_size)
    data_dict = {'data': data, 'data2': data2, 'timestamps': timestamps, 'ftt': np.array([0] * 6000)}

    signal_queue = mp.Queue()
    control_queue = mp.Queue()
    info_queue = mp.Queue()

    liveplot = LivePlotter(data_dict, signal_queue, control_queue, info_queue)

    plt.ion()
    fig = plt.figure(figsize=(16, 7))
    grid = GridSpec(3, 3, figure=fig)
    signal_plot = fig.add_subplot(grid[0, :])
    fft_plot = fig.add_subplot(grid[1, :])
    filtered_plot = fig.add_subplot(grid[2, :])
    # create a variable for the line so we can later update it
    line1, = signal_plot.plot(timestamps, data, linewidth=0.5)
    line2, = signal_plot.plot(timestamps, data2, 'r', linewidth=0.5)
    line_ftt, = fft_plot.plot([0] * round(len(data) / 2), linewidth=0.5)
    # update plot label/title
    plt.show()

    signal_plot.set_ylim([0, 4096])
    fft_plot.set_ylim([-1000, 1000])

    #
    # UI ELEMENTS
    #

    # [left, bottom, width, height]
    button_clear = Button(plt.axes([0.5, 0.92, 0.08, 0.045]), 'ltrim')
    button_clear.on_clicked(liveplot.on_ltrim_click)

    button_save = Button(plt.axes([0.6, 0.92, 0.08, 0.045]), 'Save CSV')
    button_save.on_clicked(liveplot.on_save_click)

    button_autoy = Button(plt.axes([0.7, 0.92, 0.08, 0.045]), 'Auto-Y')
    button_autoy.on_clicked(liveplot.on_func1_click)

    button_zoom = Button(plt.axes([0.8, 0.92, 0.08, 0.045]), 'Zoom')
    button_zoom.on_clicked(liveplot.on_func3_click)

    button_pause = Button(plt.axes([0.9, 0.92, 0.08, 0.045]), 'Pause')
    button_pause.on_clicked(liveplot.on_pause_click)

    # liveplot.infotext = TextBox(plt.axes([0.01, 0.535, 0.14, 0.45]), "", "Foo\nBar\nKay", color="1")
    liveplot.infotext = fig.text(0.01, 0.01, "Loading...")

    plt.subplots_adjust(left=0.185, right=0.99, top=0.9, bottom=0.05)

    #
    # Threads
    #

    update_thread = LivePlotProc(signal_queue, control_queue, info_queue)
    decode_thread = LivePlotDecodeThread(plt, signal_plot, data_dict, signal_queue, info_queue, liveplot)
    # ftt_thread = FttThread(plt, data_dict, liveplot)
    filter_thread = FilterThread(plt, data_dict, liveplot)

    ftt_update_counter = 0
    base_update_counter = 0

    start_time = time.time()

    while 1:
        try:
            if liveplot.paused is False:
                if base_update_counter > 3:
                    new_x = data_dict['timestamps'][-liveplot.signal_zoom:]
                    line1.set_xdata(new_x)
                    signal_plot.set_xlim([new_x[0], new_x[-1]])

                    # Memorize the first ever real-data timestamp:
                    # if liveplot.first_timestamp == 0 and initial_padding_size in data_dict['timestamps']:
                    #     liveplot.first_timestamp = data_dict['timestamps'][initial_padding_size]

                    new_y = data_dict['data'][-liveplot.signal_zoom:]

                    if liveplot.auto_y:
                        signal_plot.set_ylim([round(min(new_y / 50)) * 50 - 25, round(max(new_y / 50)) * 50 + 25]) # update limit channel
                    else:
                        signal_plot.set_ylim([0, 4096])

                    line1.set_ydata(new_y)

                    # TX line
                    new_y2 = data_dict['data2'][-liveplot.signal_zoom:]
                    line2.set_ydata(new_y2)
                    line2.set_xdata(new_x)

                    base_update_counter = 0

                if ftt_update_counter > 5:
                    min_freq = min(liveplot.current_freqs_range)
                    max_freq = max(liveplot.current_freqs_range)

                    # if not math.isnan(min_freq) and not math.isnan(max_freq):
                    line_ftt.set_xdata(liveplot.current_freqs_range)
                    line_ftt.set_ydata(liveplot.current_ftt_abs)
                    fft_plot.set_xlim([min_freq, max_freq])
                    fft_plot.set_ylim([-10, round(max(liveplot.current_ftt_abs) / 50) * 50 + 25])  # update limit channel

                    liveplot.mean_dt = round(np.mean(np.diff(data_dict['timestamps'][-3000:])), 1)
                    if liveplot.mean_dt > 0:
                        liveplot.sampling_rate = 10**6 / liveplot.mean_dt

                    # INFOTEXT
                    text = ""
                    text += "Run time: %.2f\n" % (time.time() - start_time)
                    text += "Data time: %.2f\n" % ((new_x[-1]) / 10**6)
                    text += "Datapoints: %d\n" % (len(data_dict['data']))
                    # text += "S/Queue size: %d\n" % (signal_queue.qsize()) # not supported on macs
                    text += "Mean DT: %s\n" % (liveplot.mean_dt)

                    if len(liveplot.lines_per_read) > 1:
                        mean_lines_per_read = np.mean(liveplot.lines_per_read)
                        text += "Mean lines/read: %.1f\n" % (mean_lines_per_read)

                    #
                    # mean_time_per_read = np.mean(liveplot.time_per_read) / 10**6
                    # text += "Mean time/read: %.6f s\n" % (mean_time_per_read)

                    # if mean_time_per_read > 0:
                    #     text += "Sampling hz: %.4f\n" % (mean_lines_per_read / mean_time_per_read)

                    text += "Sampling hz: %.4f\n" % (liveplot.sampling_rate)

                    if liveplot.strong_freqs is not None and len(liveplot.strong_freqs) > 0:
                        strong_freqs = np.array(liveplot.strong_freqs)

                        # strong_freqs shape: ([i, self.liveplot.current_freqs_range[i], freq_amp])
                        i = 0
                        # print("==========")
                        for freq in reversed(strong_freqs[strong_freqs[:,2].argsort()][0:25]):
                            # print(freq)
                            text += "F%d) %.1f: %d Hz\n" % (i, freq[2], freq[1])
                            i += 1

                    liveplot.infotext.set_text(text)
                    ftt_update_counter = 0

                ftt_update_counter += 1
                base_update_counter += 1

            # print(data[-1])
            time.sleep(0.03)
            plt.pause(0.03)
        except KeyboardInterrupt:
            print("Interrupted by user!")
            return data

    return data

class LivePlotter():
    def __init__(self, data_dict, signal_queue, control_queue, info_queue):
        self.data_dict = data_dict

        self.paused = False
        self.auto_y = True

        self.infotext = None # TextBox for data display

        self.current_ftt_abs = None
        self.current_freqs_range = None
        self.strong_freqs = None

        self.time_per_read = []
        self.lines_per_read = []

        self.mean_dt = 0
        self.sampling_rate = 0

        self.first_timestamp = 0 # first recorded serial data timestamp

        self.signal_zoom = 3000
        self.initial_padding = 6000

        self.signal_queue = signal_queue
        self.control_queue = control_queue
        self.info_queue = info_queue

    def on_pause_click(self, event):
        self.paused = not self.paused
        self.control_queue.put("toggle_pause")

    def on_func1_click(self, event):
        self.auto_y = not self.auto_y

    def on_func2_click(self, event):
        pass

    def on_func3_click(self, event):
        if self.signal_zoom == 3000:
            self.signal_zoom = 1500
        elif self.signal_zoom == 1500:
            self.signal_zoom = 750
        elif self.signal_zoom == 750:
            self.signal_zoom = 325
        elif self.signal_zoom == 325:
            self.signal_zoom = 3000

    def on_ltrim_click(self, event):
        self.data_dict['timestamps'] = self.data_dict['timestamps'][-(self.signal_zoom):]
        self.data_dict['data'] = self.data_dict['data'][-(self.signal_zoom):]
        self.data_dict['data2'] = self.data_dict['data2'][-(self.signal_zoom):]

    def on_save_click(self, event):
        r = self.initial_padding - 1
        rows = []
        for timestamp in self.data_dict['timestamps'][r:]:
            rows.append(",".join([str(int(self.data_dict['timestamps'][r])), str(self.data_dict['data'][r]), str(self.data_dict['data2'][r])]))
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
                print(int(data[2:]))
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
            self.data_dict['ftt'] = scipy.fftpack.fft(self.data_dict['data'][-1500:])
            time.sleep(0.01)

# class LivePlotThread(threading.Thread):
#     def __init__(self, plot, series, data_dict, serial, liveplot):
#         threading.Thread.__init__(self)
#         self.deamon = True
#         self.plot = plot
#         self.series = series
#         self.data_dict = data_dict
#         self.liveplot = liveplot
#         self.serial = serial
#         self.serial.reset_input_buffer()
#         self.start()
#
#     def run(self):
#         skipped_first = False
#         read_size_lines = 50000
#
#         while 1 and main_thread.isAlive() and self.serial.is_open:
#             last_lines = self.serial.readlines(read_size_lines)
#
#             # Skip the very first line, because it's often truncated
#             if skipped_first is False:
#                 last_lines = last_lines[1:read_size_lines]
#                 skipped_first = True
#
#             new_vals_count = len(last_lines)
#             new_vals1 = [None] * new_vals_count
#             new_vals2 = [None] * new_vals_count
#             new_timestamps = [None] * new_vals_count
#             # for _ in range(new_vals_count): self.data.pop(0)
#             i = 0
#             last_timestamp = 0
#             last_val = 0
#             last_val2 = 0
#
#             trim_length = 0
#
#             for line in last_lines:
#                 str_line = str(line, 'utf-8')
#                 # print("> " + str_line)
#
#                 # Check integrity
#                 if str_line[0] != "T" or str_line[-2:] != "\r\n":
#                     trim_length += 1
#                     continue
#
#                 str_line = str_line.rstrip()
#
#                 # Check integrity
#                 if "\r" in str_line:
#                     trim_length += 1
#                     continue
#
#                 # Skip check byte:
#                 str_line = str_line[1:]
#
#                 # try:
#                 parts = str_line.split(",")
#
#                 if len(parts) < 3:
#                     trim_length += 1
#                     continue
#
#                 try:
#                     parts[1] = parts[1].rstrip()
#                     parts[2] = parts[2].rstrip()
#
#                     if parts[0] != '':
#                         new_timestamps[i] = int(parts[0])
#                     else:
#                         trim_length += 1
#                         continue
#                         # new_timestamps[i] = int(last_timestamp)
#
#                     # Skip broken timestamps
#                     if new_timestamps[i] < last_timestamp: #or (last_timestamp != 0 and new_timestamps[i] - last_timestamp > 10000):
#                         trim_length += 1
#                         continue
#
#                     if parts[1] != '':
#                         new_vals1[i] = int(parts[1])
#                     else:
#                         trim_length += 1
#                         continue
#                         # new_vals1[i] = int(last_val)
#
#                     if parts[2] != '':
#                         new_vals2[i] = int(parts[2])
#                     else:
#                         trim_length += 1
#                         continue
#                         # new_vals2[i] = int(last_val2)
#
#                     # Debug output
#                     if new_timestamps[i] - last_timestamp > 1000:
#                         print("%s, with prev ts: %d, diff of: %d" % (str_line, last_timestamp, new_timestamps[i] - last_timestamp))
#
#                     last_timestamp = new_timestamps[i]
#                     last_val = new_vals1[i]
#                     last_val2 = new_vals2[i]
#
#                 except BaseException as error:
#                     raise error
#                     # new_timestamps[i] = last_timestamp
#                     # new_vals1[i] = last_val
#                     # new_vals2[i] = last_val2
#
#                     print(str(error))
#                     print("Skipped line %s" % line)
#                     trim_length += 1
#                     continue
#
#                 i += 1
#
#             # print("%d / %d" % (len(new_timestamps), len(new_vals)))
#
#             new_timestamps = new_timestamps[:new_vals_count - trim_length]
#             new_vals1 = new_vals1[:new_vals_count - trim_length]
#             new_vals2 = new_vals2[:new_vals_count - trim_length]
#
#             if trim_length > 0:
#                 print("Skipped %d malformed buffer lines" % trim_length)
#
#             self.liveplot.lines_per_read.append(len(last_lines))
#             self.liveplot.lines_per_read = self.liveplot.lines_per_read[-100:]
#
#             if len(new_timestamps) > 2:
#                 self.liveplot.time_per_read.append(new_timestamps[-1] - new_timestamps[0])
#                 self.liveplot.time_per_read = self.liveplot.time_per_read[-100:]
#
#             self.data_dict['data'] = np.append(self.data_dict['data'], new_vals1)
#             self.data_dict['data2'] = np.append(self.data_dict['data2'], new_vals2)
#             self.data_dict['timestamps'] = np.append(self.data_dict['timestamps'], new_timestamps)
#             # pdb.set_trace()
#             time.sleep(1/(57600 * 1))
#
#         self.serial.close()

# Decodes the multiprocessing signal_queue into the data_dict
class LivePlotDecodeThread(threading.Thread):
    def __init__(self, plot, series, data_dict, signal_queue, info_queue, liveplot):
        threading.Thread.__init__(self)
        self.deamon = True
        self.plot = plot
        self.series = series
        self.data_dict = data_dict
        self.liveplot = liveplot
        self.signal_queue = signal_queue
        self.info_queue = info_queue
        self.start()

    def run(self):
        skipped_first = False
        read_size_lines = 50000

        first_timestamp = 0
        last_timestamp = 0

        while 1 and main_thread.isAlive():
            self.liveplot.read_info_queue()

            try:
                str_line = self.signal_queue.get(False)
            except BaseException as error:
                # print(error)
                time.sleep(0.0001)
                continue

            # Sanity-check for length, a normal payload will be between 20 and 22 bytes (less \r\n)
            if len(str_line) < 16 or len(str_line) > 20:
                continue

            # try:
            parts = str_line.split(",")

            new_timestamp = 0
            new_val1 = 0
            new_val2 = 0

            if len(parts) < 3:
                continue

            try:
                new_timestamp = int(parts[0])

                if first_timestamp == 0:
                    first_timestamp = self.liveplot.first_timestamp = new_timestamp

                new_timestamp = new_timestamp - first_timestamp

                # Skip broken timestamps
                # if new_timestamp < last_timestamp: #or (last_timestamp != 0 and new_timestamps[i] - last_timestamp > 10000):
                #     continue

                new_val1 = int(parts[1])
                new_val2 = int(parts[2])

                # # Debug output
                # if new_timestamps[i] - last_timestamp > 1000:
                #     print("%s, with prev ts: %d, diff of: %d" % (str_line, last_timestamp, new_timestamps[i] - last_timestamp))

                last_timestamp = new_timestamp

            except BaseException as error:
                # print(error)
                raise error
                print("Skipped line %s" % str_line)
                continue

            # print("%d / %d" % (len(new_timestamps), len(new_vals)))

            # self.liveplot.lines_per_read.append(len(last_lines))
            # self.liveplot.lines_per_read = self.liveplot.lines_per_read[-100:]
            #
            # if len(new_timestamps) > 2:
            #     self.liveplot.time_per_read.append(new_timestamps[-1] - new_timestamps[0])
            #     self.liveplot.time_per_read = self.liveplot.time_per_read[-100:]

            self.data_dict['data'] = np.append(self.data_dict['data'], [new_val1])
            self.data_dict['data2'] = np.append(self.data_dict['data2'], [new_val2])
            self.data_dict['timestamps'] = np.append(self.data_dict['timestamps'], [new_timestamp])
            # pdb.set_trace()
            # time.sleep(1/(57600 * 1))

class LivePlotProc(mp.Process):
    def __init__(self, signal_queue, control_queue, info_queue):
        mp.Process.__init__(self)
        self.deamon = True
        self.signal_queue = signal_queue
        self.control_queue = control_queue
        self.info_queue = info_queue
        self.paused = False
        self.serial = serial.Serial(default_usb_port, baudrate=3686400)
        self.serial.reset_output_buffer()
        self.serial.reset_input_buffer()
        self.start()

    def run(self):
        skipped_first = False
        max_bytes_per_read = 1000000
        # os.nice(-19)

        while 1:
            if self.paused is True: # do pause
                time.sleep(0.01)

            # if not self.serial.is_open:
            #     # wait for serial
            #     time.sleep(0.01)
            #     pass

            last_lines = self.serial.readlines(max_bytes_per_read)

            self.info_queue.put_nowait("RL%d" % len(last_lines))

            # Skip the very first line, because it's often truncated
            if skipped_first is False:
                last_lines = last_lines[1:]
                skipped_first = True

            new_vals_count = len(last_lines)
            new_vals1 = [None] * new_vals_count
            new_vals2 = [None] * new_vals_count
            new_timestamps = [None] * new_vals_count
            # for _ in range(new_vals_count): self.data.pop(0)
            i = 0
            last_timestamp = 0
            last_val = 0
            last_val2 = 0

            trim_length = 0

            for line in last_lines:
                str_line = str(line, 'utf-8')
                # print("> " + str_line)

                # Check integrity
                if str_line[0] != "T" or str_line[-2:] != "\r\n":
                    continue

                str_line = str_line[1:].rstrip()

                # Check integrity
                if "\r" in str_line or 'T' in str_line:
                    continue

                # print(str_line)
                self.signal_queue.put_nowait(str_line)
                i += 1

            try:
                command = self.control_queue.get(False)
                self.process_command(command)
            except BaseException as error:
                pass

    def process_command(self, command):
        if command == 'toggle_pause':
            self.paused = not self.paused

    def send_command(self, command):
        self.control_queue.put(command)

class FilterThread(threading.Thread):
    def __init__(self, plot, data_dict, liveplot):
        threading.Thread.__init__(self)
        self.deamon = True
        self.plot = plot
        self.data_dict = data_dict
        self.liveplot = liveplot
        self.iw = None
        self.iw_ftt = None
        self.initial_window_prepared = False
        self.start()

    def prepare_initial_window(self):
        self.iw = self.data_dict['data']
        self.iw_ftt = scipy.fftpack.fft(self.iw)
        sampling_rate = 636 / 1000000 # TODO

        raw_freqs = scipy.fftpack.fftfreq(len(self.iw_ftt), sampling_rate)
        i = round(len(raw_freqs) / 2) + 1
        self.iw_freqs = raw_freqs[0:(i - 1)]
        self.iw_ftt_abs = np.abs(self.iw_ftt[(i - 1):])

        # TODO: Detect initial frequencies

        self.initial_window = True

    def run(self):
        if self.initial_window_prepared is False:
            self.prepare_initial_window()

        while 1 and main_thread.isAlive():
            ftt_data = self.data_dict['ftt'] = scipy.fftpack.fft(self.data_dict['data'][-6000:])

            self.liveplot.mean_dt = round(np.mean(np.diff(self.data_dict['timestamps'][-3000:])), 1)
            sampling_rate = self.liveplot.sampling_rate = self.liveplot.mean_dt / 1000000

            raw_freqs = scipy.fftpack.fftfreq(len(ftt_data), sampling_rate)
            iis = round(len(raw_freqs) / 2) + 1
            self.liveplot.current_freqs_range = raw_freqs[0:(iis - 1)]
            ftt_abs = self.liveplot.current_ftt_abs = np.abs(ftt_data[(iis - 1):])

            max_amp = np.max(ftt_abs)
            median_amp = np.median(ftt_abs)
            mean_amp = np.mean(ftt_abs)
            stdev_amp = np.std(ftt_abs)
            min_amp = np.min(ftt_abs)

            self.liveplot.strong_freqs = []
            i = 0

            for freq_amp in ftt_abs:
                # Highpass
                if freq_amp < max_amp * 0.5 and freq_amp > median_amp:
                    self.liveplot.strong_freqs.append([i, self.liveplot.current_freqs_range[i], freq_amp])
                    # continue
                # elif freq > min_amp * 1.1:
                #     selected_freqs.append([i, freq])

                i = i + 1

            do_filter = True

            time.sleep(636 / 1000000 * 100)

def test_sine_filter():
    # create data
    N = 4097
    T = 100.0
    t = np.linspace(-T / 2, T / 2, N)
    f = np.sin(50.0 * 2.0 * np.pi * t)# + 0.5 * np.sin(80.0 * 2.0 * np.pi * t)

    # plot function
    plt.ion()

    fig = plt.figure(figsize=(13, 6))

    signal_plot = fig.add_subplot(3, 1, 1)
    fft_plot = fig.add_subplot(3, 1, 2)
    freqs_plot = fig.add_subplot(3, 1, 3)

    signal_plot.plot(t, f, 'r', linewidth=0.5)
    plt.show()

    # pdb.set_trace()

    # perform FT and multiply by dt
    dt = t[1] - t[0]
    ft = np.fft.fft(f) * dt
    freqs = np.fft.fftfreq(N, dt)
    # pdb.set_trace()
    ind = round(N / 2 + 1)
    freq = freqs[:ind]
    amplitude = np.abs(ft[:ind])
    # plot results
    # plt.plot(freq, amplitude)
    # plt.legend(('numpy fft * dt'), loc='upper right')
    # plt.xlabel('f')
    # plt.ylabel('amplitude')
    signal_plot.set_xlim([0, 1.4])
    signal_plot.set_ylim([-1.5, 1.5])

    fft_plot.plot(freqs, ft, linewidth=0.5)
    freqs_plot.plot(freq, linewidth=0.5)

    # pdb.set_trace()
    wc = freq[np.argmax(amplitude)] / (0.5 / dt)
    # wp = [wc * 0.9, wc / 0.9]
    # ws = [wc * 0.95, wc / 0.95]
    wp = [wc * 0.9, wc / 0.9]
    ws = [wc * 0.95, wc / 0.95]
    b, a = signal.iirdesign(wp, ws, 1, 40)
    f2 = signal.filtfilt(b, a, f)

    signal_plot.plot(t, f2, 'g')
    plt.show()

    # plt.plot(freq[np.argmax(amplitude)], max(amplitude), 'ro')
    # print("Amplitude: %s Frequency: %s" % (str(max(amplitude)), str(freq[np.argmax(amplitude)])))

    plt.show()
    pdb.set_trace()


main_thread = threading.currentThread()

# liveplot_serial()
# record_serial(csv="data/rec1.csv")
plot_data(path="data/liverec.csv")
# test_sine_filter()
# pdb.set_trace()
