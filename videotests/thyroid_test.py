import os
import numpy as np
from pdb import set_trace
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
from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from timeit import default_timer as timer

class ThyroidTest(videotests.object_tracking_test.ObjectTrackingTest):
    def __init__(self):
        super().__init__()
        self.track_classes = ['thyroid_left_view_right', 'thyroid_left', 'trachea_front']
        self.load_box_detection_model()

    def load_box_detection_model(self):
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
        config.gpu_options.per_process_gpu_memory_fraction = 0.82 # https://github.com/tensorflow/tensorflow/issues/22623#issuecomment-425702729
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


    def run(self):
        video_path = "training-usbase/thyroid_right_1.mp4"
        self.crop_style = "ipad_sd"

        cap = cv.VideoCapture(video_path)

        # params for ShiTomasi corner detection
        feature_params = dict( maxCorners = 100,
                               qualityLevel = 0.3,
                               minDistance = 7,
                               blockSize = 7 )

        # Parameters for lucas kanade optical flow
        lk_params = dict( winSize  = (15,15),
                          maxLevel = 2,
                          criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
        # Create some random colors
        color = np.random.randint(0,255,(100,3))

        # rewind before analysis
        cap.set(cv.CAP_PROP_POS_MSEC, 0)

        # Take first frame and find corners in it
        p0, old_gray, old_frame = self.select_features_to_track(cap)

        # Create a mask image for drawing purposes
        mask = np.zeros_like(old_frame)

        self.update_window_layout()

        frames_till_feature_change = 0

        detect_frame_fps_period = 10 # every nth frame
        detection_frame_counter = detect_frame_fps_period

        # Registers to placehold last bounding box predictions
        out_boxes = ()
        out_scores = ()
        out_classes = ()

        while(1):
            if frames_till_feature_change > 100:
                p0, old_gray, old_frame = self.select_features_to_track(cap)
                frames_till_feature_change = 0

                # Create a mask image for drawing purposes
                mask = np.zeros_like(old_frame)

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
                # print("Detected %d boxes" % (len(out_boxes)))

                # TODO: Detect features only inside the frame via select_features_to_track

            p1, st, err, frame_gray = self.update_tracked_features(frame, old_gray, p0)

            # Select good points
            good_new = p1[st==1]
            good_old = p0[st==1]

            # Draw bounding boxes
            if len(out_boxes) > 0:
                frame = np.asarray(self.draw_and_grab_frames_bounding_boxes(frame, keras_image, out_boxes, out_scores, out_classes))

            # draw feature tracks
            for i,(new,old) in enumerate(zip(good_new, good_old)):
                a,b = new.ravel()
                c,d = old.ravel()
                mask = cv.line(mask, (a,b),(c,d), color[i].tolist(), 2)
                frame = cv.circle(frame,(a,b),5,color[i].tolist(),-1)

            img = cv.add(frame, mask)

            cv.imshow('frame', img)

            # Update detected objects windows
            for class_name in self.current_detected_objects:
                if class_name in self.track_classes:
                    i = 0
                    for child_img in self.current_detected_objects[class_name]:
                        cv.imshow("%s_%d" % (class_name, i), child_img)
                        i += 1

            if len(self.current_detected_objects) > 0:
                self.update_window_layout()

            # Now update the previous frame and previous points
            old_gray = frame_gray.copy()
            p0 = good_new.reshape(-1,1,2)

            k = cv.waitKey(30) & 0xff
            if k == ord('q'):
                break

            frames_till_feature_change += 1

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

    def update_window_layout(self):
        origin = (30, 30)
        main_size = (700,700)

        cv.namedWindow('frame', cv.WINDOW_NORMAL)
        cv.resizeWindow('frame', main_size[0], main_size[1])
        cv.moveWindow('frame', origin[0], origin[1])

        win_sizes = dict()
        prev_shape = None

        displayed_feature_window_count = 0
        prev_offset = [0, 0]

        for class_name in self.track_classes:
            winname = '%s_0' % class_name

            if class_name in self.current_detected_objects:
                offset = [main_size[0], 0]

                if displayed_feature_window_count >= 1:
                    offset[1] = prev_offset[1] + prev_shape[0] + 50

                prev_offset = offset

                cv.namedWindow(winname, cv.WINDOW_NORMAL)
                win_sizes[class_name] = (self.current_detected_objects[class_name][0].shape[1], self.current_detected_objects[class_name][0].shape[0])
                cv.resizeWindow(winname, win_sizes[class_name][0], win_sizes[class_name][1])
                cv.moveWindow(winname, origin[0] + offset[0], origin[1] + offset[1])

                prev_shape = self.current_detected_objects[class_name][0].shape

                displayed_feature_window_count += 1

        self.window_layout_needs_update = False