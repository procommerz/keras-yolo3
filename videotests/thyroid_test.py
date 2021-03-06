import os
import sys
from threading import Thread

import numpy as np
import copy
from pdb import set_trace

from PyQt5 import QtGui

import videotests.object_tracking_test
import cv2 as cv
from PIL import Image, ImageFont, ImageDraw
import matplotlib
import matplotlib.pyplot as plt

import keras
import tensorflow as tf
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from keras.utils import multi_gpu_model

from videotests.thyroid_test_app import ThyroidTestApp
from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from timeit import default_timer as timer

class DetectBoundingBoxThread(Thread):
    def __init__(self):
        Thread.__init__(self)

class ThyroidTest(videotests.object_tracking_test.ObjectTrackingTest):
    # A 'signal' used to update the video surface in the parent Qt App
    qt_main_video_update_signal = None

    def __init__(self):
        super().__init__()
        self.track_classes = ['thyroid_left_view_right', 'thyroid_left']

    def load_everything(self):
        self.load_box_detection_model()

    def load_box_detection_model(self):
        print("[ThyroidTest] Loading Keral model...")
        self.box_detection_model_path = 'checkpoints/v2/trained_weights_final.h5'
        self.box_detection_anchors_path = 'model_data/yolo_anchors.txt'
        self.box_detection_classes_path = 'model_data/training-usbase.names'

        gpu_num = 1
        self.score = 0.3
        self.iou = 0.45
        self.box_detection_model_image_size = (416, 416)

        # Read anchors
        anchors_path = os.path.expanduser(self.box_detection_anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        self.box_detection_anchors = np.array(anchors).reshape(-1, 2)

        # Read class names
        classes_path = os.path.expanduser(self.box_detection_classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        self.box_detection_class_names = [c.strip() for c in class_names]

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.5 # https://github.com/tensorflow/tensorflow/issues/22623#issuecomment-425702729
        self.sess = tf.Session(config=config)
        K.set_session(self.sess)

        # Read the model
        model_path = os.path.expanduser(self.box_detection_model_path)
        num_anchors = len(self.box_detection_anchors)
        num_classes = len(self.box_detection_class_names)
        is_tiny_version = num_anchors==6 # default setting
        try:
            self.box_detection_yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
            self.yolo_model.load_weights(model_path) # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                num_anchors/len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        self.input_image_shape = K.placeholder(shape=(2, ))

        if gpu_num >=2:
            self.yolo_model = multi_gpu_model(self.yolo_model, gpus=self.gpu_num)

        self.box_detection_boxes, self.box_detection_scores, self.box_detection_classes = self.yolo_eval(self.yolo_model.output, self.box_detection_anchors,
                len(self.box_detection_class_names), self.input_image_shape,
                score_threshold=self.score, iou_threshold=self.iou)

        self.generate_colors()


    def start(self):
        app = QtGui.QApplication(sys.argv)
        self.qtapp = ThyroidTestApp(app=app, testapp=self)
        self.qtapp.show()
        sys.exit(app.exec_())

    def run(self):
        video_path = "training-usbase/thyroid_right_1.mp4"
        self.crop_style = "ipad_sd"

        cap = cv.VideoCapture(video_path)

        # params for ShiTomasi corner detection
        feature_params = dict(maxCorners = 100,
                              qualityLevel = 0.3,
                              minDistance = 7,
                              blockSize = 7)

        # Parameters for lucas kanade optical flow
        lk_params = dict(winSize  = (15,15),
                         maxLevel = 2,
                         criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
        # Create some random colors
        color = np.random.randint(0, 255, (100, 3))

        # rewind before analysis
        cap.set(cv.CAP_PROP_POS_MSEC, 0)

        # Take first frame and find corners in it
        p0, old_gray, old_frame = self.select_features_to_track(cap)

        # Create a mask image for drawing purposes
        mask = np.zeros_like(old_frame)

        # self.update_window_layout()

        frames_till_feature_change = 0

        detect_frame_fps_period = 10 # every nth frame
        detection_frame_counter = detect_frame_fps_period

        # Registers to placehold last bounding box predictions
        out_boxes = ()
        out_scores = ()
        out_classes = ()

        while(1):
            # if frames_till_feature_change > 100:
            #     p0, old_gray, old_frame = self.select_features_to_track(cap)
            #     frames_till_feature_change = 0
            #
            #     # Create a mask image for drawing purposes
            #     mask = np.zeros_like(old_frame)

            ret, frame = cap.read()

            # Rewind to start
            if frame is None:
                print("Rewinding")
                cap.set(cv.CAP_PROP_POS_MSEC, 0)
                continue

            frame = self.crop_frame(frame)
            keras_image = Image.fromarray(frame)

            detection_frame_counter -= 1

            # Detect bounding boxes
            if detection_frame_counter == 0:
                out_boxes, out_scores, out_classes = self.detect_image(keras_image)
                detection_frame_counter = detect_frame_fps_period

                # TODO: Detect features only inside the frame via select_features_to_track

            # p1, st, err, frame_gray = self.update_tracked_features(frame, old_gray, p0)
            #
            # # Select good points
            # good_new = p1[st==1]
            # good_old = p0[st==1]
            #
            # # draw feature tracks
            # for i,(new,old) in enumerate(zip(good_new, good_old)):
            #     a,b = new.ravel()
            #     c,d = old.ravel()
            #     mask = cv.line(mask, (a, b), (c, d), color[i].tolist(), 2)
            #     frame = cv.circle(frame, (a, b), 5, color[i].tolist(), -1)

            # Draw bounding boxes
            if len(out_boxes) > 0:
                frame = np.asarray(self.draw_and_grab_frames_bounding_boxes(frame, keras_image, out_boxes, out_scores, out_classes))

            main_img = cv.add(frame, mask)
            self.qt_main_video_update_signal.emit(self.convert_frame_to_qt_image(main_img))

            contrast_alpha = 3
            contrast_beta = 0

            for class_name in self.current_detected_objects:
                if class_name in self.track_classes:
                    i = 0

                    if len(self.current_detected_objects[class_name]) > 0:
                        img = np.asarray(self.current_detected_objects[class_name][0])
                        img = cv.add(img, np.zeros(shape=img.shape, dtype=img.dtype))
                        original_img = np.copy(img)

                        # Increase contrast of the  image
                        # for x in range(img.shape[0]):
                        #     for y in range(img.shape[1]):
                        #         for c in range(img.shape[2]):
                        #             img[x,y,c] = np.clip(contrast_alpha * img[x,y,c] + contrast_beta, 0, 255)
                        # img = np.zeros(img.shape, img.dtype)
                        img = cv.fastNlMeansDenoising(img, img, h=9, templateWindowSize=7, searchWindowSize=21)
                        img = cv.convertScaleAbs(img, alpha=contrast_alpha, beta=contrast_beta)
                        # img[:,:] = np.clip(contrast_alpha * img[:,:] + contrast_beta, 0, 255)

                        self.draw_feature_controur(class_name, img)
                        qt_original_img = self.convert_frame_to_qt_image(original_img)
                        qt_img = self.convert_frame_to_qt_image(img)
                        self.qt_feature_frame_update_signal.emit(class_name, qt_original_img, qt_img)

            # Notify parent QT window

            # Now update the previous frame and previous points
            # old_gray = frame_gray.copy()
            # p0 = good_new.reshape(-1, 1, 2)

            frames_till_feature_change += 1

    def process_main_and_features(self, frame, mask):
        main_img = cv.add(frame, mask)
        print("Emitting...")
        self.qt_main_video_update_signal.emit(self.convert_frame_to_qt_image(main_img))

        # cv.imshow('frame', img)

        # Update detected objects windows
        for class_name in self.current_detected_objects:
            if class_name in self.track_classes:
                i = 0

                if len(self.current_detected_objects[class_name]) > 0:
                    img = np.asarray(self.current_detected_objects[class_name][0])
                    img = cv.add(img, np.zeros(shape=img.shape, dtype=img.dtype))
                    self.qt_feature_frame_update_signal.emit(class_name, self.convert_frame_to_qt_image(img))

    def draw_feature_controur(self, class_name, image):
        image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        ret, thresh = cv.threshold(image_gray, 127, 255, 0)
        contours, hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        # set_trace()
        cv.drawContours(image, contours, -1, (0,255,0), 3)

    def convert_frame_to_qt_image(self, img):
        h, w, ch = img.shape
        bytes_per_line = ch * w
        return QtGui.QImage(img, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)

    def select_features_to_track(self, cap):
        # params for ShiTomasi corner detection
        feature_params = dict( maxCorners = 100,
                               qualityLevel = 0.3,
                               minDistance = 7,
                               blockSize = 7 )

        # Take first frame and find corners in it
        ret, old_frame = cap.read()
        old_frame = self.crop_frame(old_frame)
        old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
        p0 = cv.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

        return (p0, old_gray, old_frame)

    def update_tracked_features(self, frame, old_gray, p0):
        # Parameters for lucas kanade optical flow
        lk_params = dict( winSize  = (15,15),
                          maxLevel = 2,
                          criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # calculate optical flow
        p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        return (p1, st, err, frame_gray)
