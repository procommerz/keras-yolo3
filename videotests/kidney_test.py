import numpy as np
import cv2 as cv
from PIL import Image, ImageFont, ImageDraw
import matplotlib
import matplotlib.pyplot as plt

class KidneyTest(object):
    def __init__(self):
        pass

    def run(self):
        video_path = "training-usbase/kidney_right_lateral_denis.mp4"
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
        cap.set(cv.CAP_PROP_POS_MSEC, 10000)

        # Take first frame and find corners in it
        p0, old_gray, old_frame = self.select_features_to_track(cap)

        # Create a mask image for drawing purposes
        mask = np.zeros_like(old_frame)

        cv.namedWindow('frame', cv.WINDOW_NORMAL)
        cv.resizeWindow('frame', 700, 700)

        frames_till_feature_change = 0

        while(1):
            if frames_till_feature_change > 100:
                p0, old_gray, old_frame = self.select_features_to_track(cap)
                frames_till_feature_change = 0

                # Create a mask image for drawing purposes
                mask = np.zeros_like(old_frame)

            ret,frame = cap.read()
            frame = self.crop_frame(frame)

            p1, st, err, frame_gray = self.update_tracked_features(frame, old_gray, p0)

            # Select good points
            good_new = p1[st==1]
            good_old = p0[st==1]

            # draw the tracks
            for i,(new,old) in enumerate(zip(good_new, good_old)):
                a,b = new.ravel()
                c,d = old.ravel()
                mask = cv.line(mask, (a,b),(c,d), color[i].tolist(), 2)
                frame = cv.circle(frame,(a,b),5,color[i].tolist(),-1)

            img = cv.add(frame,mask)
            cv.imshow('frame', img)
            k = cv.waitKey(30) & 0xff
            if k == 27:
                break
            # Now update the previous frame and previous points
            old_gray = frame_gray.copy()
            p0 = good_new.reshape(-1,1,2)

            frames_till_feature_change += 1

    # Crops the frame to hide the UI
    # of the recorded app
    def crop_frame(self, frame):
        height = frame.shape[0]
        width = frame.shape[1]
        # Hide the 'B' left side indicator
        cv.circle(frame, (530,310), 10, (0,0,0), thickness=50, lineType=-1)
        #return frame
        return frame[280:height-300, 0:width-100]

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