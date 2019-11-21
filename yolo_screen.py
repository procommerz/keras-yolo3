import sys
import argparse
import numpy as np
from yolo import YOLO, timer
from functools import reduce
from PIL import Image
from pdb import set_trace
import cv2
from mss import mss
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
from pyqtgraph.Qt import QtCore, QtGui
from PyQt5.QtCore import pyqtSlot, Qt
import pyqtgraph as pg
import math, random, threading, time, os, sys, queue, _thread
import scipy.misc

FLAGS = None

if __name__ == '__main__':
    # class YOLO defines the default value, so suppress any default here
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    '''
    Command line options
    '''
    parser.add_argument(
        '--model', type=str, default="checkpoints/trained_weights_final.h5",
        help='path to model weight file, default ' + YOLO.get_defaults("model_path")
    )

    parser.add_argument(
        '--anchors', type=str, default="model_data/yolo_anchors.txt",
        help='path to anchor definitions, default ' + YOLO.get_defaults("anchors_path")
    )

    parser.add_argument(
        '--classes', type=str, default="model_data/training-usbase.names",
        help='path to class definitions, default ' + YOLO.get_defaults("classes_path")
    )

    parser.add_argument(
        '--gpu_num', type=int,
        help='Number of GPU to use, default ' + str(YOLO.get_defaults("gpu_num"))
    )

    parser.add_argument(
        '--image', default=False, action="store_true",
        help='Image detection mode, will ignore all positional arguments'
    )
    '''
    Command line positional arguments -- for video detection mode
    '''
    parser.add_argument(
        "--input", nargs='?', type=str,required=False,default='./training-usbase/thyroid_inna_right_v2.mp4',
        help = "Video input path"
    )

    parser.add_argument(
        "--output", nargs='?', type=str, default="",
        help = "[Optional] Video output path"
    )

    FLAGS = parser.parse_args()

class NebrusScreenDemo(QtGui.QMainWindow):
    def __init__(self, parent=None):
        super(NebrusScreenDemo, self).__init__(parent)

        #### Create Gui Elements ###########
        self.mainbox = QtGui.QWidget()
        self.screen_resolution = app.desktop().screenGeometry()
        kw = 0.4
        kh = 0.3

        #self.setFixedSize(screen_resolution.width(), int(screen_resolution.height() * 0.936)) # 0.636))
        self.setFixedSize(int(self.screen_resolution.width() * kh), int(self.screen_resolution.height() * kw))  # 0.636))
        self.setCentralWidget(self.mainbox)
        self.mainbox.setLayout(QtGui.QGridLayout())

        # QGridLayout#addWidget(): (QWidget * widget, int fromRow, int fromColumn, int rowSpan, int columnSpan, Qt::Alignment alignment = 0)

        # self.canvas = pg.GraphicsLayoutWidget()
        # self.mainbox.layout().addWidget(self.canvas, 0, 1, 1, 6) # last param = number of buttons + 1

        self.image_view = pg.ImageView()
        self.mainbox.layout().addWidget(self.image_view, 0, 0, 1, 1)

        self.image_width = int(self.screen_resolution.width() - self.screen_resolution.width() * kw)
        self.image_height = int(self.screen_resolution.height() - self.screen_resolution.height() * kh)

        # sct = mss()
        # # pdb.set_trace() ###
        # monitor = self.get_mss_monitor()
        # image = np.array(sct.grab(monitor))
        # image = np.flip(image[:, :, :3], 2)
        # image = np.rot90(image, axes=(-2, -1))
        # self.image_view.setImage(image, xvals=np.linspace(1., 3., image.shape[0]))

        # self.image_view.autoRange()

        self.yolo = YOLO(**vars(FLAGS))
        self.update()

    def detect_screen(self):
        sct = mss()
        # pdb.set_trace() ###
        monitor = {'top': int(self.screen_resolution.height() * 0.3), 'left': int(self.screen_resolution.width() * kw), 'width': int(self.screen_resolution.width() * self.screen_resolution.width() * kw), 'height': int(self.screen_resolution.height() * self.screen_resolution.height() * kh)}
        image = np.array(sct.grab(monitor))
        image = np.flip(image[:, :, :3], 2)
        frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame)
        image.show()

    def detect_img(self, yolo):
        while True:
            img = input('Input image filename:')
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                r_image = yolo.detect_image(image)
                r_image.show()
        yolo.close_session()

    def detect_video(self, yolo, video_path, output_path=""):
        import cv2
        vid = cv2.VideoCapture(video_path)
        if not vid.isOpened():
            raise IOError("Couldn't open webcam or video")
        video_FourCC = int(vid.get(cv2.CAP_PROP_FOURCC))
        video_fps = vid.get(cv2.CAP_PROP_FPS)
        video_size = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                      int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        isOutput = True if output_path != "" else False
        if isOutput:
            print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))
            out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)
        accum_time = 0
        curr_fps = 0
        fps = "FPS: ??"
        prev_time = timer()
        while True:
            return_value, frame = vid.read()
            image = Image.fromarray(frame)
            image = yolo.detect_image(image)
            result = np.asarray(image)
            curr_time = timer()
            exec_time = curr_time - prev_time
            prev_time = curr_time
            accum_time = accum_time + exec_time
            curr_fps = curr_fps + 1
            if accum_time > 1:
                accum_time = accum_time - 1
                fps = "FPS: " + str(curr_fps)
                curr_fps = 0
            cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.50, color=(255, 0, 0), thickness=2)
            cv2.namedWindow("result", cv2.WINDOW_NORMAL)
            cv2.imshow("result", result)
            if isOutput:
                out.write(result)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        yolo.close_session()

    def get_mss_monitor(self):
        kw = 0.4
        kh = 0.3

        monitor = {'top': int(self.screen_resolution.height() * 0.3),
                   'left': int(self.screen_resolution.width() * kw),
                   'width': int(self.screen_resolution.width() - self.screen_resolution.width() * kw),
                   'height': int(self.screen_resolution.height() - self.screen_resolution.height() * kh)}

        return monitor

    def update(self):
        try:
            sct = mss()
            # pdb.set_trace() ###
            monitor = self.get_mss_monitor()

            image = np.array(sct.grab(monitor))
            #image = np.flip(image[:, :, :3], 2)
            #image = np.rot90(image, axes=(-2, -1))

            #frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            #image_ax = Image.fromarray(frame)

            image_ax = Image.fromarray(image)
            tf_image = self.yolo.detect_image(image_ax)
            tf_image = scipy.ndimage.rotate(tf_image, 90)
            tf_image = np.asarray(tf_image)
            #tf_image = np.flip(tf_image[:, :, :3], 2)
            tf_image = np.flipud(tf_image)
            #tf_image = np.rot90(tf_image, axes=(-2, -1))
            self.image_view.setImage(tf_image)
            self.image_view.autoRange()

            QtCore.QTimer.singleShot(33, self.update)
            # self.counter += 1
        except KeyboardInterrupt:
            print("Exiting gracefully...")
            # self.decode_thread.join()
            # self.filter_thread.join()
            # self.update_threaad.terminate()
            # self.multi_pulse_thread.terminate()
        except BaseException as e:
            print(e)
            print("update thread: %s" % (str(e)))
            raise e
            QtCore.QTimer.singleShot(1, self.update)

    def keyPressEvent(self, e):
        print(str(e.key()))
        if e.key() == Qt.Key_F5:
            self.close()
# if FLAGS.image:
#     """
#     Image detection mode, disregard any remaining command line arguments
#     """
#     print("Image detection mode")
#     if "input" in FLAGS:
#         print(" Ignoring remaining command line arguments: " + FLAGS.input + "," + FLAGS.output)
#     detect_img(YOLO(**vars(FLAGS)))
# elif "input" in FLAGS:
#     detect_video(YOLO(**vars(FLAGS)), FLAGS.input, FLAGS.output)
# else:
#     print("Must specify at least video_input_path.  See usage with --help.")

app = None

main_thread = threading.currentThread()

if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    thisapp = NebrusScreenDemo()
    thisapp.show()
    sys.exit(app.exec_())
